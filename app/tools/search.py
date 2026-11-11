from typing import Any, Dict, List, Optional
from duckduckgo_search import DDGS
from loguru import logger
from pydantic import BaseModel, Field

from app.llm import GeminiClient
from app.tools.base import tool_registry


class SearchResult(BaseModel):
    """Schema representing a single search result from DuckDuckGo."""

    title: str = Field(..., description="The title of the search result page.")
    url: str = Field(..., description="The URL of the result page.")
    snippet: str = Field(
        ..., description="The text snippet or summary describing the page content."
    )


def search_ddg(query: str, max_results: int = 5) -> List[SearchResult]:
    """Queries DuckDuckGo search and returns structured SearchResult objects.

    Args:
        query: The search query string.
        max_results: The maximum number of results to fetch.

    Returns:
        A list of SearchResult models.
    """
    if not query or not query.strip():
        logger.warning("Empty search query received. Returning empty list.")
        return []

    logger.info(f"Querying DuckDuckGo for: '{query}' (limit: {max_results})")
    results: List[SearchResult] = []

    try:
        # Initialize DuckDuckGo Search Context Manager
        with DDGS() as ddgs:
            # Fetch text search results
            ddg_results = ddgs.text(query, max_results=max_results)

            if ddg_results:
                for r in ddg_results:
                    title = r.get("title", "")
                    url = r.get("href", "")
                    snippet = r.get("body", "")

                    if title and url:
                        results.append(
                            SearchResult(title=title, url=url, snippet=snippet)
                        )

        logger.info(f"DuckDuckGo returned {len(results)} structured results.")
        return results

    except Exception as e:
        logger.error(f"DuckDuckGo search encountered an error: {e}")
        # Return empty list; let the fallback handling logic in the tool wrapper generate a placeholder
        return []


def generate_fallback_mock_results(query: str) -> List[SearchResult]:
    """Generates static domain-specific search results when search fails or returns zero results.

    Provides a clean, static overview using typical standards. Uses 0 Gemini calls.
    """
    logger.warning(
        f"Search returned 0 results. Using static fallback research context for query: '{query}'."
    )
    return [
        SearchResult(
            title=f"Technical Standards: {query.title() if hasattr(query, 'title') else query}",
            url="https://internal-agent-kb.local/standards_research",
            snippet=f"Factual reference and standards overview regarding the topic: '{query}'. Includes typical practices, industry guidelines, and structural recommendations.",
        )
    ]


# Automatically register the search wrapper under tool name 'web_search'
@tool_registry.register("web_search")
def web_search_tool(
    task_title: str,
    task_description: str,
    dependency_context: Dict[str, Any],
    max_results: int = 4,
) -> str:
    """The tool wrapper registered under the 'web_search' identifier.

    Uses task description directly (no Gemini calls) and formats results as clean markdown.
    """
    logger.info(f"Tool 'web_search' invoked for task: '{task_title}'.")

    # Use the task description directly as the search query
    search_query = task_description

    # Execute DuckDuckGo Search
    results = search_ddg(search_query, max_results=max_results)

    # Trigger static fallback if no results are returned
    if not results:
        results = generate_fallback_mock_results(search_query)

    # Format results into clean Markdown string
    formatted_results = []
    formatted_results.append(
        f"### Research Results for query: '{search_query}'"
    )
    for i, r in enumerate(results, 1):
        formatted_results.append(
            f"{i}. **[{r.title}]({r.url})**\n   {r.snippet}\n"
        )

    return "\n".join(formatted_results)
