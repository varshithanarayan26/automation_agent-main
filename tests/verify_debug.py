import sys
import os
# Insert project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
import httpx
from loguru import logger
from pathlib import Path
import json

from app.llm import GeminiClient
from app.agent.planner import AutonomousPlanner
from app.agent.executor import PlanExecutor
from app.agent.reflection import ReflectionAuditor
from app.tools.doc_generator import DocxBuilder, compile_document_tool
from app.core.models import ExecutionPlan, Task
from app.core.config import settings

async def main():
    logger.info("=== DEBUG VERIFICATION TESTS ===")

    # 1. Test importing app.llm & creating GeminiClient
    logger.info("1. Testing import & creation of GeminiClient...")
    try:
        client = GeminiClient()
        logger.info("GeminiClient instantiated successfully.")
    except Exception as e:
        logger.error(f"Failed to instantiate client: {e}")
        return

    # 2. Test one simple prompt ("Hello")
    logger.info("2. Testing one simple Gemini prompt ('Hello')...")
    try:
        response = client.generate_text("Hello")
        logger.info(f"Gemini Response: '{response}'")
    except Exception as e:
        logger.warning(f"Simple prompt failed as expected with auth error: {e}")

    # 3. Test planner
    logger.info("3. Testing planner...")
    planner = AutonomousPlanner(llm_client=client)
    try:
        plan = planner.plan("Create a cyber report.")
        logger.info(f"Planner succeeded. Plan tasks count: {len(plan.tasks)}")
    except Exception as e:
        logger.warning(f"Planner live run failed as expected with auth error: {e}")

    # 4. Test executor
    logger.info("4. Testing executor...")
    executor = PlanExecutor(llm_client=client)
    # Build a mock plan to test executor execution loop
    mock_plan = ExecutionPlan(
        tasks=[
            Task(id="task_01", title="Mock Search", description="Query test", tool="web_search"),
            Task(id="task_02", title="Mock Draft", description="Draft section text", tool="draft_section", dependencies=["task_01"])
        ]
    )
    try:
        logger.info("Running executor on plan...")
        updated_plan = await executor.execute(mock_plan)
        logger.info("Executor completed mock execution.")
        for task in updated_plan.tasks:
            logger.info(f"Task {task.id} Status: {task.status}")
    except Exception as e:
        logger.error(f"Executor crashed unexpectedly: {e}")

    # 5. Test reflection auditor
    logger.info("5. Testing reflection auditor...")
    auditor = ReflectionAuditor(llm_client=client)
    try:
        ref_res = auditor.audit("Request summary", mock_plan)
        logger.info(f"Auditor completed. Approved: {ref_res.is_approved}, Score: {ref_res.quality_score}")
    except Exception as e:
        logger.warning(f"Auditor live run failed as expected with auth error: {e}")

    # 6. Test document generation (including locked-file PermissionError check)
    logger.info("6. Testing document generation...")
    try:
        builder = DocxBuilder(title="Initial Report", doc_type="Test SOP")
        builder.add_cover_page(subtitle="Initial verification")
        builder.add_markdown_content("# Introduction\nThis is a test introduction.")
        builder.add_header_footer()
        
        # Test locked-file save mechanism
        test_filename = "strategic_business_report.docx"
        out_dir = settings.output_path
        target_path = out_dir / test_filename
        
        # Ensure directory exists and create a file
        target_path.touch(exist_ok=True)
        
        # Mock save to raise PermissionError on the target_path but succeed on other paths
        original_save = builder.doc.save
        def mock_save(path_str):
            if Path(path_str).resolve() == target_path.resolve():
                logger.info(f"Mocking PermissionError for locked file path: {path_str}")
                raise PermissionError("[WinError 32] The process cannot access the file because it is being used by another process")
            return original_save(path_str)
        builder.doc.save = mock_save
        
        try:
            # Now trigger docx save, it should hit PermissionError, generate unique name, and save
            saved_path = builder.save(test_filename)
            logger.info(f"Saved file path returned: {saved_path}")
            
            if saved_path != target_path:
                logger.info(f"Locked file bypass working! Document saved to unique fallback: {saved_path.name}")
            else:
                logger.error("Failed to trigger locked file bypass.")
        finally:
            # Restore original save method
            builder.doc.save = original_save
                
    except Exception as e:
        logger.error(f"Document generation test failed: {e}")

    # 7. Test FastAPI /agent endpoint
    logger.info("7. Testing FastAPI /agent endpoint...")
    try:
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            payload = {"request": "Create onboarding report."}
            r = await http_client.post("http://127.0.0.1:8001/agent", json=payload)
            logger.info(f"POST /agent Status Code: {r.status_code}")
            logger.info(f"POST /agent Response keys: {list(r.json().keys())}")
            logger.info(f"POST /agent Final Status: {r.json().get('status')}")
    except Exception as e:
        logger.error(f"FastAPI /agent endpoint test query failed: {e}")

    logger.info("=== DEBUG VERIFICATION COMPLETED ===")

if __name__ == "__main__":
    asyncio.run(main())
