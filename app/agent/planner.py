from datetime import datetime, UTC
from typing import Any, Dict, List, Optional, TypedDict
from langgraph.graph import START, END, StateGraph
from loguru import logger
from pydantic import BaseModel, Field

from app.core.models import ExecutionPlan, Task
from app.llm import GeminiClient

# Define intermediate schemas for LLM structured output parsing during planning


class RawTask(BaseModel):
    """Temporary schema for task parsing from LLM response."""

    id: str = Field(
        ...,
        description="Unique task identifier, sequentially numbered (e.g., 'task_01', 'task_02').",
    )
    title: str = Field(
        ...,
        description="A short, action-oriented title for the task (e.g., 'Research CRM features').",
    )
    description: str = Field(
        ...,
        description="Detailed description of what this task should draft, research, or audit.",
    )
    tool: str = Field(
        ...,
        description="The tool required for this task. Must be one of: 'web_search', 'draft_section', 'reflection_check'.",
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of task IDs that this task depends on. (e.g., a drafting task depends on its research task).",
    )


class RawDocDetails(BaseModel):
    """Temporary schema for document metadata detection."""

    document_type: str = Field(
        ...,
        description="The class of document (e.g., 'Product Specification', 'Business Proposal', 'SOP', 'Meeting Minutes').",
    )
    document_title: str = Field(
        ...,
        description="A professional, descriptive title for the document.",
    )
    suggested_sections: List[str] = Field(
        ...,
        description="List of main heading titles suggested for the document outline.",
    )


# LangGraph state definition for the planning graph
class PlannerState(TypedDict):
    request: str
    model_name: Optional[str]
    additional_context: Optional[Dict[str, Any]]
    document_type: Optional[str]
    document_title: Optional[str]
    suggested_sections: Optional[List[str]]
    raw_tasks: Optional[List[RawTask]]
    execution_plan: Optional[ExecutionPlan]


class AutonomousPlanner:
    """Uses a LangGraph state machine to analyze requirements, detect document types,

    and construct a valid directed acyclic graph (DAG) of execution tasks (TODO list).
    """

    def __init__(self, llm_client: Optional[GeminiClient] = None):
        """Initializes the planner with an LLM client and compiles the LangGraph workflow."""
        self.llm = llm_client or GeminiClient()

        # Build and compile the state graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Constructs the LangGraph state graph for the planning workflow."""
        builder = StateGraph(PlannerState)

        # Register nodes
        builder.add_node("detect_document_type", self._node_detect_document_type)
        builder.add_node("generate_task_plan", self._node_generate_task_plan)
        builder.add_node("compile_and_validate", self._node_compile_and_validate)

        # Wire graph edges
        builder.add_edge(START, "detect_document_type")
        builder.add_edge("detect_document_type", "generate_task_plan")
        builder.add_edge("generate_task_plan", "compile_and_validate")
        builder.add_edge("compile_and_validate", END)

        return builder.compile()

    def _node_detect_document_type(self, state: PlannerState) -> Dict[str, Any]:
        """LangGraph node: Analyzes the request and context to auto-detect document details."""
        logger.info("Planner Node: Detecting document type and outline.")

        prompt = f"""
        Analyze the following user request and additional context. 
        Determine the most appropriate business document type (e.g. SOP, Product Specification, Proposal, Business Report) and a professional title. 
        Provide a list of recommended main sections/headings for this document type.

        Original Request: {state['request']}
        Context: {state.get('additional_context') or "None provided"}
        """

        system_instruction = (
            "You are an expert business analyst and technical writer. "
            "Detect document types and outline appropriate structures."
        )

        try:
            details: RawDocDetails = self.llm.generate_structured(
                prompt=prompt,
                response_schema=RawDocDetails,
                model=state.get("model_name"),
                system_instruction=system_instruction,
            )

            logger.info(
                f"Detected document type: '{details.document_type}' with title: '{details.document_title}'"
            )

            return {
                "document_type": details.document_type,
                "document_title": details.document_title,
                "suggested_sections": details.suggested_sections,
            }
        except Exception as e:
            logger.error(f"Error in document type detection node: {e}")
            request_text = state.get("request", "Project Document")
            
            # Professional document type heuristics
            doc_type = "Strategic Plan"
            lower_req = request_text.lower()
            if any(k in lower_req for k in ["sop", "procedure", "standard operating"]):
                doc_type = "Standard Operating Procedure"
            elif any(k in lower_req for k in ["proposal", "bid"]):
                doc_type = "Business Proposal"
            elif any(k in lower_req for k in ["plan", "roadmap"]):
                doc_type = "Strategic Plan"
            elif any(k in lower_req for k in ["report", "audit"]):
                doc_type = "Business Report"
            elif any(k in lower_req for k in ["case", "study"]):
                doc_type = "Case Study"
            elif any(k in lower_req for k in ["spec", "specification"]):
                doc_type = "Technical Specification"

            # Derive clean professional title
            doc_title = request_text
            for prefix in ["create a", "write a", "generate a", "create", "write", "generate"]:
                if doc_title.lower().startswith(prefix):
                    doc_title = doc_title[len(prefix):].strip()
            doc_title = doc_title.title()

            # Dynamic suggested sections
            sections = ["Executive Summary", "Market Context & Objectives", "Strategic Recommendations", "Conclusion"]
            if "sop" in doc_type.lower() or "procedure" in doc_type.lower():
                sections = ["Purpose & Scope", "Roles & Responsibilities", "Procedure Workflow", "Compliance & Audit"]
            elif "specification" in doc_type.lower() or "technical" in doc_type.lower():
                sections = ["System Architecture Overview", "Functional Requirements", "Interface & Data Specifications", "Verification & Testing"]

            return {
                "document_type": doc_type,
                "document_title": doc_title,
                "suggested_sections": sections,
            }

    def _node_generate_task_plan(self, state: PlannerState) -> Dict[str, Any]:
        """LangGraph node: Generates the list of tasks required to create the document."""
        logger.info("Planner Node: Generating task execution plan.")

        sections_str = ", ".join(
            [f"'{s}'" for s in state.get("suggested_sections", [])]
        )

        prompt = f"""
        You are planning the creation of a document of type: '{state.get('document_type')}' 
        titled '{state.get('document_title')}' to satisfy this request: "{state['request']}"

        We have decided on the following structural sections: {sections_str}
        Context: {state.get('additional_context') or "None"}

        Create a plan (list of tasks) to build this document step-by-step.
        
        Available tools are:
        1. 'web_search': Use this to perform external research (e.g. competitors, best practices, standards).
        2. 'draft_section': Use this to write the content of a single specific section.
        3. 'reflection_check': Use this to audit and check the drafted sections for completeness, style, and placeholders.

        Rules for Task Generation:
        - Include tasks to perform necessary research first using 'web_search'.
        - Include separate 'draft_section' tasks for EACH section of the document.
        - Ensure a 'reflection_check' task is added after all drafting is complete to review the final draft.
        - Map out dependencies cleanly (e.g. drafting tasks should depend on the research tasks; reflection should depend on all drafting tasks).
        - Use unique IDs (task_01, task_02, etc.).
        """

        system_instruction = (
            "You are an expert project planner. Break down document writing projects "
            "into logical, dependency-resolved task lists utilizing available tools."
        )

        class TaskList(BaseModel):
            tasks: List[RawTask]

        try:
            response: TaskList = self.llm.generate_structured(
                prompt=prompt,
                response_schema=TaskList,
                model=state.get("model_name"),
                system_instruction=system_instruction,
            )

            logger.info(f"Generated {len(response.tasks)} raw planning tasks.")
            return {"raw_tasks": response.tasks}
        except Exception as e:
            logger.error(f"Error in task plan generation node: {e}")
            suggested_sections = state.get("suggested_sections") or ["Introduction", "Analysis", "Recommendations"]
            
            fallback_tasks = []
            # 1. Research task
            fallback_tasks.append(
                RawTask(
                    id="task_01",
                    title="Research",
                    description=f"Perform search queries regarding the topic: {state.get('request')}",
                    tool="web_search",
                )
            )
            
            # 2. Drafting tasks for each suggested section
            dep = ["task_01"]
            for idx, section in enumerate(suggested_sections, 2):
                task_id = f"task_{idx:02d}"
                fallback_tasks.append(
                    RawTask(
                        id=task_id,
                        title=section,
                        description=f"Draft the {section} section of the document based on research",
                        tool="draft_section",
                        dependencies=dep.copy(),
                    )
                )
                
            # 3. Final review task
            review_task_id = f"task_{(len(suggested_sections) + 2):02d}"
            fallback_tasks.append(
                RawTask(
                    id=review_task_id,
                    title="Review",
                    description="Audit draft and make necessary corrections",
                    tool="reflection_check",
                    dependencies=[f"task_{(len(suggested_sections) + 1):02d}"],
                )
            )
            return {"raw_tasks": fallback_tasks}

    def _node_compile_and_validate(
        self, state: PlannerState
    ) -> Dict[str, Any]:
        """LangGraph node: Validates constraints (acyclic dependencies) and compiles ExecutionPlan."""
        logger.info("Planner Node: Validating and compiling task graph.")

        raw_tasks = state.get("raw_tasks") or []
        valid_tasks: List[Task] = []

        # Convert RawTask to Domain Task model
        for rt in raw_tasks:
            valid_tasks.append(
                Task(
                    id=rt.id,
                    title=rt.title,
                    description=rt.description,
                    tool=rt.tool,
                    dependencies=rt.dependencies,
                    status="pending",
                )
            )

        # 1. Verify dependency existence
        task_ids = {t.id for t in valid_tasks}
        for task in valid_tasks:
            # Filter dependencies that do not exist to prevent deadlocks
            task.dependencies = [dep for dep in task.dependencies if dep in task_ids]

        # 2. Cycle detection (Topological safety check)
        if self._has_cycles(valid_tasks):
            logger.warning(
                "Cycle detected in task dependencies! Clearing dependencies to guarantee progress."
            )
            # Safe recovery: clear dependencies and enforce sequential fallback
            for i, task in enumerate(valid_tasks):
                if i > 0:
                    task.dependencies = [valid_tasks[i - 1].id]
                else:
                    task.dependencies = []

        # Compile final Pydantic plan
        now = datetime.now(UTC)
        execution_plan = ExecutionPlan(
            tasks=valid_tasks,
            created_at=now,
            updated_at=now,
        )

        return {"execution_plan": execution_plan}

    def _has_cycles(self, tasks: List[Task]) -> bool:
        """Determines if there are cyclic dependencies in the task graph using DFS."""
        adj = {t.id: t.dependencies for t in tasks}
        visited = {t.id: 0 for t in tasks}  # 0=unvisited, 1=visiting, 2=visited

        def dfs(node_id: str) -> bool:
            if node_id not in visited:
                return False
            if visited[node_id] == 1:
                return True  # Cycle found
            if visited[node_id] == 2:
                return False

            visited[node_id] = 1
            for neighbor in adj.get(node_id, []):
                if dfs(neighbor):
                    return True
            visited[node_id] = 2
            return False

        for task in tasks:
            if visited[task.id] == 0:
                if dfs(task.id):
                    return True
        return False

    def plan(
        self,
        request: str,
        model_name: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionPlan:
        """Runs the LangGraph planning workflow end-to-end.

        Args:
            request: Natural language instructions.
            model_name: Optional override for model.
            additional_context: Additional variables dictionary.

        Returns:
            A validated ExecutionPlan.
        """
        logger.info(f"Starting planning phase for request: '{request}'")

        initial_state: PlannerState = {
            "request": request,
            "model_name": model_name,
            "additional_context": additional_context,
            "document_type": None,
            "document_title": None,
            "suggested_sections": None,
            "raw_tasks": None,
            "execution_plan": None,
        }

        # Invoke the compiled state graph
        final_state = self.graph.invoke(initial_state)

        plan = final_state.get("execution_plan")
        if not plan:
            raise RuntimeError("Failed to generate a valid ExecutionPlan.")

        # Store document details to bypass extra LLM queries in main.py
        self.last_document_title = final_state.get("document_title")
        self.last_document_type = final_state.get("document_type")

        logger.info(
            f"Successfully compiled plan with {len(plan.tasks)} tasks."
        )
        return plan
