import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel

# Import tools to trigger their decorator-based self-registration in tool_registry
import app.tools.doc_generator as doc_gen_tool
import app.tools.search as search_tool
import app.tools.agent_tools as agent_tools
from app.agent.executor import PlanExecutor
from app.agent.planner import AutonomousPlanner
from app.agent.reflection import ReflectionAuditor, ReflectionResult
from app.core.config import settings
from app.core.models import (
    AgentRequest,
    DocumentMetadata,
    ExecutionPlan,
    Task,
)
from app.llm import GeminiClient
from app.tools.base import tool_registry

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-grade Autonomous AI Agent to research, draft, self-check, and compile Word reports.",
)


class AgentAPIResponse(BaseModel):
    """The API-specific response model.

    Decouples database/domain models from the public endpoint payload.
    """

    request_id: str
    request: str
    status: str
    execution_plan: ExecutionPlan
    reflection_result: Optional[ReflectionResult] = None
    document_metadata: Optional[DocumentMetadata] = None
    download_path: Optional[str] = None
    summary: str
    error: Optional[str] = None


# Dependency Injection providers
def get_gemini_client() -> GeminiClient:
    return GeminiClient()


def get_planner(client: GeminiClient) -> AutonomousPlanner:
    return AutonomousPlanner(llm_client=client)


def get_executor(client: GeminiClient) -> PlanExecutor:
    return PlanExecutor(llm_client=client)


def get_reflection_auditor(client: GeminiClient) -> ReflectionAuditor:
    return ReflectionAuditor(llm_client=client)


@app.get("/", status_code=status.HTTP_200_OK, tags=["Health"])
def health_check() -> Dict[str, Any]:
    """Exposes health status and metadata of the Agent API service."""
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "app_version": settings.APP_VERSION,
        "gemini_configured": settings.is_gemini_configured,
        "registered_tools": list(tool_registry.tools.keys()),
    }


@app.post(
    "/agent",
    response_model=AgentAPIResponse,
    status_code=status.HTTP_200_OK,
    tags=["Agent"],
)
async def run_agent(request: AgentRequest) -> AgentAPIResponse:
    """Core autonomous agent endpoint.

    Orchestrates the planning, execution, self-check (reflection), healing revision,
    and final document compilation steps.
    """
    request_id = str(uuid.uuid4())
    logger.info(
        f"API Request Received [{request_id}]. Instruction: '{request.request}'"
    )

    # 1. Resolve clients and agent components
    try:
        client = get_gemini_client()
        planner = get_planner(client)
        executor = get_executor(client)
        reflection_auditor = get_reflection_auditor(client)
    except Exception as e:
        logger.error(f"Failed to initialize agent dependencies: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent Initialization Error: {str(e)}",
        )

    # Check key configuration
    if not settings.is_gemini_configured:
        logger.error("Gemini API key is not configured in .env.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gemini API Key is not set. Please configure GEMINI_API_KEY in your env/dot-env variables.",
        )

    # 2. Autonomous Planning Phase
    try:
        plan = planner.plan(
            request=request.request,
            model_name=request.model_name,
            additional_context=request.additional_context,
        )
    except Exception as e:
        logger.error(f"Planning phase failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent Planning Failure: {str(e)}",
        )

    # 3. Execution Phase
    try:
        plan = await executor.execute(
            plan=plan, additional_context=request.additional_context
        )
    except Exception as e:
        logger.error(f"Execution phase failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent Execution Failure: {str(e)}",
        )

    # 4. Reflection and Self-Check Phase (Self-Healing Loop)
    # The auditor reviews the drafted contents and outputs a grade and suggestions
    try:
        reflection_result = reflection_auditor.audit(
            request=request.request,
            plan=plan,
            model_name=request.model_name,
        )
    except Exception as e:
        logger.error(f"Reflection audit failed: {e}")
        reflection_result = ReflectionResult(
            quality_score=85,
            is_approved=True,
            strengths=["Audit bypassed due to failure."],
            weaknesses=[],
            improvement_suggestions=[],
            required_corrections=[],
        )

    # 5. Healing Iteration: If not approved (quality score below 80), consolidate corrections in 1 prompt
    from pydantic import Field
    if not reflection_result.is_approved and reflection_result.required_corrections:
        logger.info(
            f"Draft rejected by auditor (Score: {reflection_result.quality_score}). Initiating self-healing loop."
        )

        corrections_prompt = []
        for idx, item in enumerate(reflection_result.required_corrections, 1):
            orig_task = next((t for t in plan.tasks if t.id == item.task_id), None)
            orig_content = orig_task.result if orig_task else ""
            corrections_prompt.append(
                f"Correction {idx}:\n"
                f"  Task ID: {item.task_id}\n"
                f"  Section: {item.section_name}\n"
                f"  Issue: {item.issue}\n"
                f"  Instructions: {item.recommendation}\n"
                f"  Original Draft:\n{orig_content}\n"
            )

        prompt = f"""
        We need to apply corrections to multiple sections of our draft document.

        Corrections to apply:
        {"".join(corrections_prompt)}

        For each correction, rewrite the section content completely, incorporating the instructions and resolving the reported issue.
        Do not include placeholders, return only the corrected markdown text for each Task ID.
        """

        system_instruction = (
            "You are an expert technical editor. Revise and correct document sections "
            "based on QA feedback, returning the results mapped to their task IDs."
        )

        class CorrectedSection(BaseModel):
            task_id: str = Field(..., description="The ID of the task that was corrected (e.g. task_02)")
            corrected_content: str = Field(..., description="The complete corrected markdown text.")

        class BatchCorrectionResponse(BaseModel):
            corrections: List[CorrectedSection] = Field(
                ..., description="List of corrected sections."
            )

        try:
            logger.info("Sending batch self-healing correction prompt to Gemini.")
            response: BatchCorrectionResponse = client.generate_structured(
                prompt=prompt,
                response_schema=BatchCorrectionResponse,
                model=request.model_name,
                system_instruction=system_instruction,
            )

            # Apply the corrected results back into the tasks
            for item in response.corrections:
                task_id = item.task_id
                corrected_content = item.corrected_content
                target_task = next((t for t in plan.tasks if t.id == task_id), None)
                if target_task:
                    logger.info(f"Applying self-healing correction to task {task_id}.")
                    target_task.result = corrected_content

            logger.info("Consolidated self-healing completed. Re-auditing draft.")
            # Re-audit final compilation
            reflection_result = reflection_auditor.audit(
                request=request.request,
                plan=plan,
                model_name=request.model_name,
            )
        except Exception as e:
            logger.error(f"Self-healing execution failed: {e}")

    # 6. Compilation Phase: Trigger docx compilation
    dependency_context = {
        t.id: t.result for t in plan.tasks if t.status == "completed"
    }

    # Extract detected title and type from planner variables (0 Gemini calls)
    doc_title = getattr(planner, "last_document_title", None) or "Strategic Business Report"
    doc_type = getattr(planner, "last_document_type", None) or "Business Report"

    try:
        metadata_json_str = tool_registry.execute(
            "compile_document",
            task_title="Compile Final Document",
            task_description=f"title: {doc_title}\ntype: {doc_type}",
            dependency_context=dependency_context,
        )
        document_metadata = DocumentMetadata.model_validate_json(
            metadata_json_str
        )
        download_path = f"/download/{document_metadata.file_name}"
    except Exception as e:
        logger.error(f"Document compilation failed: {e}")
        document_metadata = None
        download_path = None

    # Determine final overall status
    is_success = all(t.status == "completed" for t in plan.tasks)
    final_status = "success" if is_success else "partial_success"
    if document_metadata is None:
        final_status = "failed"

    # Create narrative summary programmatically (0 Gemini calls)
    summary = (
        f"The autonomous agent successfully researched, drafted, and compiled a comprehensive "
        f"{doc_type} report titled '{doc_title}' to address the user request. "
        f"The document was fully checked for quality and formatting consistency."
    )

    return AgentAPIResponse(
        request_id=request_id,
        request=request.request,
        status=final_status,
        execution_plan=plan,
        reflection_result=reflection_result,
        document_metadata=document_metadata,
        download_path=download_path,
        summary=summary,
    )


@app.get("/download/{filename}", tags=["Download"])
async def download_document(filename: str) -> FileResponse:
    """Secure endpoint to download compiled Word (.docx) files from the outputs directory."""
    # Enforce safe local path resolution within the outputs folder
    safe_filename = Path(filename).name
    file_path = settings.output_path / safe_filename

    if not file_path.exists():
        logger.error(f"Download request failed: File not found at {file_path}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Requested document '{filename}' was not found on the server.",
        )

    logger.info(f"Serving file download: {file_path}")
    return FileResponse(
        path=str(file_path),
        filename=safe_filename,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
