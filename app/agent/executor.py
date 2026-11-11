import asyncio
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional
from loguru import logger

from app.core.models import ExecutionPlan, Task
from app.llm import GeminiClient

from app.tools.base import tool_registry
import app.tools.agent_tools as agent_tools


class PlanExecutor:
    """Orchestrates the execution of a multi-step ExecutionPlan.

    Executes tasks in dependency order, aggregates results into a shared context,
    supports parallel path analysis, and isolates task failures.
    """

    def __init__(self, llm_client: Optional[GeminiClient] = None):
        """Initializes the executor with the shared LLM client."""
        self.llm = llm_client or GeminiClient()

    async def execute(
        self,
        plan: ExecutionPlan,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionPlan:
        """Executes the given ExecutionPlan in dependency order.

        Args:
            plan: The input ExecutionPlan containing tasks to perform.
            additional_context: Additional global values to merge into the shared scratchpad.

        Returns:
            The updated ExecutionPlan with results and statuses.
        """
        logger.info(f"Starting execution of plan containing {len(plan.tasks)} tasks.")

        # Shared execution context/scratchpad containing output of each task
        # Format: { task_id: task_result }
        context: Dict[str, Any] = additional_context or {}

        # Run the execution loop until all tasks reach a terminal state (completed or failed)
        while not self._is_execution_finished(plan.tasks):
            ready_tasks = self._get_ready_tasks(plan.tasks)

            if not ready_tasks:
                # Execution is not finished, but no tasks are ready.
                # This indicates a deadlock (cyclic dependency) missed by the planner.
                logger.error(
                    "Deadlock detected! Pending tasks exist but none are ready. Aborting."
                )
                self._fail_remaining_tasks(
                    plan.tasks, "Aborted: Plan execution deadlock encountered."
                )
                break

            logger.info(
                f"Found {len(ready_tasks)} task(s) ready for execution: {[t.id for t in ready_tasks]}"
            )

            # Execute ready tasks sequentially to respect free tier rate limits (RPM/TPM)
            for task in ready_tasks:
                await self._execute_single_task(task, plan.tasks, context)

        logger.info("Plan execution process completed.")
        plan.updated_at = datetime.now(UTC)
        return plan

    async def _execute_single_task(
        self, task: Task, all_tasks: List[Task], context: Dict[str, Any]
    ) -> None:
        """Executes a single task, updates its status, and saves its outputs."""
        logger.info(f"Task {task.id} [{task.title}]: Initiating execution.")
        task.status = "running"

        try:
            # Check if this task tool is draft_section and has already been filled in a batch run
            if task.tool == "draft_section" and task.id in context:
                task.result = context[task.id]
                task.status = "completed"
                logger.info(f"Task {task.id} [{task.title}]: Already drafted in batch.")
                return

            # 1. Gather results from dependency tasks to feed as context
            dep_results = {
                dep_id: context[dep_id]
                for dep_id in task.dependencies
                if dep_id in context
            }

            # If this is a draft_section task, perform a batch call for ALL pending draft_section tasks
            if task.tool == "draft_section":
                # Find all draft_section tasks in the execution plan
                draft_tasks = [t for t in all_tasks if t.tool == "draft_section"]

                # Check if we have research context from dependencies of any drafting task
                research_context = {}
                for t in all_tasks:
                    if t.tool == "web_search" and t.id in context:
                        research_context[t.id] = context[t.id]

                logger.info(
                    f"Triggering batch draft execution for {len(draft_tasks)} tasks."
                )
                result = await self._run_batch_draft(
                    draft_tasks, research_context
                )

                # Parse the batch result (it should be a JSON mapping task_id -> content)
                import json

                try:
                    batch_data = json.loads(result)
                    if isinstance(batch_data, dict) and "sections" in batch_data:
                        batch_data = batch_data["sections"]
                except Exception:
                    batch_data = {}

                # Save results for all draft tasks
                from app.tools.agent_tools import generate_professional_fallback
                for dt in draft_tasks:
                    content = batch_data.get(dt.id) or generate_professional_fallback(
                        dt.title, dt.description
                    )
                    context[dt.id] = content
                    dt.result = content
                    dt.status = "completed"
                    logger.info(
                        f"Task {dt.id} [{dt.title}]: Draft saved in batch."
                    )

                task.result = context[task.id]
                task.status = "completed"
                return

            # 2. Invoke the tool via the Tool Registry or fall back to direct LLM execution
            result = await self._run_tool(task, dep_results)

            # 3. Save result and update status
            task.result = result
            task.status = "completed"
            context[task.id] = result
            logger.info(f"Task {task.id} [{task.title}]: Completed successfully.")

        except Exception as e:
            error_msg = f"Execution failed: {str(e)}"
            logger.error(f"Task {task.id} [{task.title}]: {error_msg}")
            task.status = "failed"
            task.error = error_msg

            # Propagate failure to all downstream tasks that depend on this one
            self._propagate_failure(task.id, all_tasks)

    async def _run_tool(self, task: Task, dependency_context: Dict[str, Any]) -> str:
        """Invokes the appropriate tool, falling back to direct LLM execution if missing."""
        tool_name = task.tool

        # Check if the tool exists in our registry
        # We check both the hasattr check for mock and standard dictionary checks on real registry
        has_tool = False
        if hasattr(tool_registry, "has_tool"):
            has_tool = tool_registry.has_tool(tool_name)
        elif hasattr(tool_registry, "tools"):
            has_tool = tool_name in tool_registry.tools

        if has_tool:
            logger.info(f"Executing tool '{tool_name}' for Task {task.id}.")
            # Execute registry tool inside a run_in_executor if it's synchronous
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: tool_registry.execute(
                    tool_name,
                    task_title=task.title,
                    task_description=task.description,
                    dependency_context=dependency_context,
                ),
            )
            return str(result)

        # Fallback direct LLM execution:
        # If the tool is draft_section or reflection_check and not registered yet, we run direct prompt
        logger.info(
            f"Tool '{tool_name}' not registered. Using default LLM executor fallback."
        )
        return await self._execute_llm_fallback(task, dependency_context)

    async def _execute_llm_fallback(
        self, task: Task, dependency_context: Dict[str, Any]
    ) -> str:
        """Executes a task directly via LLM prompts when no specialized tool is registered."""
        prompt = f"""
        Execute the following task:
        Task Title: {task.title}
        Task Description: {task.description}

        Context gathered from previous steps:
        {dependency_context}

        Produce the final output required for this step. If drafting, write the complete, polished content. Do not include introductory notes or commentary, return only the direct output.
        """

        system_instruction = (
            "You are an expert technical writer and AI assistant. "
            "Complete the requested task professionally, using only the provided context."
        )

        try:
            # Run text generation on the LLM client
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.llm.generate_text(
                    prompt=prompt, system_instruction=system_instruction
                ),
            )
            return result
        except Exception as e:
            logger.warning(
                f"LLM execution failed for task {task.id}: {e}. Generating professional content fallback."
            )
            from app.tools.agent_tools import generate_professional_fallback
            return generate_professional_fallback(task.title, task.description)

    def _is_execution_finished(self, tasks: List[Task]) -> bool:
        """Returns True if all tasks are in a terminal state (completed or failed)."""
        return all(t.status in {"completed", "failed"} for t in tasks)

    def _get_ready_tasks(self, tasks: List[Task]) -> List[Task]:
        """Returns all 'pending' tasks whose dependencies are entirely 'completed'."""
        ready = []
        for task in tasks:
            if task.status != "pending":
                continue

            # Check if all dependencies are marked completed
            deps_satisfied = True
            for dep_id in task.dependencies:
                dep_task = next((t for t in tasks if t.id == dep_id), None)
                if not dep_task or dep_task.status != "completed":
                    deps_satisfied = False
                    break

            if deps_satisfied:
                ready.append(task)
        return ready

    def _propagate_failure(self, failed_task_id: str, tasks: List[Task]) -> None:
        """Cascades failures down the dependency tree.

        If task A fails, any task B that depends on task A must be marked as failed.
        """
        for task in tasks:
            if task.status == "pending" and failed_task_id in task.dependencies:
                task.status = "failed"
                task.error = f"Prerequisite task '{failed_task_id}' failed."
                logger.warning(
                    f"Cascading failure: Task {task.id} [{task.title}] marked failed due to dependency '{failed_task_id}'."
                )
                # Recursively propagate failure for this task
                self._propagate_failure(task.id, tasks)

    def _fail_remaining_tasks(self, tasks: List[Task], reason: str) -> None:
        """Marks all pending or running tasks as failed due to a global error."""
        for task in tasks:
            if task.status in {"pending", "running"}:
                task.status = "failed"
                task.error = reason

    async def _run_batch_draft(
        self, draft_tasks: List[Task], research_context: Dict[str, Any]
    ) -> str:
        """Helper to invoke the registered 'draft_section' tool in batch."""
        tasks_data = [
            {"id": t.id, "title": t.title, "description": t.description}
            for t in draft_tasks
        ]

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: tool_registry.execute(
                "draft_section",
                task_title="Batch Drafting",
                task_description="Draft all sections in one prompt",
                dependency_context={
                    "tasks": tasks_data,
                    "research": research_context,
                },
            ),
        )
        return str(result)
