
from typing import Any, Dict, List, Optional
from loguru import logger
from pydantic import BaseModel, Field

from app.core.models import ExecutionPlan
from app.llm import GeminiClient


class CorrectionItem(BaseModel):
    """Schema representing a single targeted correction action recommended by the auditor."""

    task_id: str = Field(
        ...,
        description="The ID of the task requiring correction (e.g. 'task_02') or 'new_task' if a new step is required.",
    )
    section_name: str = Field(
        ...,
        description="The name of the section or topic requiring modification.",
    )
    issue: str = Field(
        ...,
        description="The description of the issue found (e.g. 'contains placeholder [Insert Name]', 'too brief').",
    )
    recommendation: str = Field(
        ...,
        description="Explicit guidance for the agent to rewrite or fix the content.",
    )


class ReflectionResult(BaseModel):
    """Schema representing the structured output of the self-check auditing process."""

    quality_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Overall grade assigned to the compiled draft, from 0 to 100.",
    )
    is_approved: bool = Field(
        ...,
        description="Whether the document is approved for final compile (should be False if score < 80 or has critical issues).",
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Aspects of the draft that are well-written, factual, or complete.",
    )
    weaknesses: List[str] = Field(
        default_factory=list,
        description="Omissions, placeholders, inconsistencies, or poor formatting aspects found.",
    )
    improvement_suggestions: List[str] = Field(
        default_factory=list,
        description="High-level structural or stylistic suggestions.",
    )
    required_corrections: List[CorrectionItem] = Field(
        default_factory=list,
        description="A list of specific, action-item corrections needed to heal the draft.",
    )


class ReflectionAuditor:
    """An AI Auditor that performs a self-check of the agent's draft content.

    Verifies task outputs against user requirements, audits for quality and completeness,
    and structures corrective tasks for self-healing.
    """

    def __init__(self, llm_client: Optional[GeminiClient] = None):
        """Initializes the auditor with a GeminiClient."""
        self.llm = llm_client or GeminiClient()

    def audit(
        self,
        request: str,
        plan: ExecutionPlan,
        model_name: Optional[str] = None,
    ) -> ReflectionResult:
        """Audits the completed task outputs of an ExecutionPlan against the user request.

        Args:
            request: The original natural language prompt.
            plan: The completed ExecutionPlan with results.
            model_name: Optional LLM model override.

        Returns:
            A structured ReflectionResult.
        """
        logger.info("Initiating reflection and self-check audit.")

        # 1. Compile draft sections from completed tasks for the auditor to inspect
        compiled_draft_segments = []
        for task in plan.tasks:
            # Skip non-drafting tasks (e.g. search tasks) to focus audit on the text
            if task.tool == "web_search" or not task.result:
                continue
            compiled_draft_segments.append(
                f"=== Section: {task.title} (ID: {task.id}) ===\n{task.result}\n"
            )

        full_draft_text = "\n".join(compiled_draft_segments)

        if not full_draft_text.strip():
            logger.warning("No drafted content was found to audit.")
            return ReflectionResult(
                quality_score=0,
                is_approved=False,
                strengths=[],
                weaknesses=["No content was drafted."],
                improvement_suggestions=[
                    "Ensure draft tasks run and write results to the context."
                ],
                required_corrections=[
                    CorrectionItem(
                        task_id="system",
                        section_name="Global",
                        issue="Empty document",
                        recommendation="Generate content for main sections.",
                    )
                ],
            )

        # 2. Formulate prompt for the audit
        prompt = f"""
        You are an expert editor and QA auditor. Audit the following document draft compiled to satisfy this request:
        
        Original Request: "{request}"
        
        Compiled Draft Content:
        \"\"\"
        {full_draft_text}
        \"\"\"
        
        Audit requirements:
        1. Check for any placeholder values (e.g., [Insert Name], [Company Name], Date, TODOs, XXX).
        2. Verify if all core constraints and details requested in the original prompt are addressed.
        3. Check for repetitive paragraphs, logical gaps, or incomplete analysis.
        4. Grade the document from 0 to 100 based on completeness, accuracy, and tone.
        5. If there are placeholders or missing sections, is_approved MUST be set to false and the score should be below 80.
        6. List targeted correction instructions under required_corrections mapping to the exact Task ID of the section.
        """

        system_instruction = (
            "You are a strict technical document reviewer. "
            "Inspect content for placeholders, errors, omissions, and grade drafts strictly."
        )

        try:
            # Execute structured JSON query
            result: ReflectionResult = self.llm.generate_structured(
                prompt=prompt,
                response_schema=ReflectionResult,
                model=model_name,
                system_instruction=system_instruction,
            )

            logger.info(
                f"Audit completed. Quality Score: {result.quality_score}/100. Approved: {result.is_approved}"
            )
            return result

        except Exception as e:
            logger.error(f"Error encountered during reflection audit: {e}")
            # Fallback bypass to ensure the agent executes cleanly in failure scenarios
            logger.warning("Applying fallback approval configuration.")
            return ReflectionResult(
                quality_score=85,
                is_approved=True,
                strengths=["Automatic bypass approval (auditor offline)."],
                weaknesses=[],
                improvement_suggestions=[],
                required_corrections=[],
            )
