from typing import Callable, Dict, Any
from loguru import logger


class ToolRegistry:
    """A production-grade, thread-safe registry for dynamic tool registration and execution.

    Follows Single Responsibility and Open-Closed principles (SOLID), allowing new tools
    to be added via decorators without modifying the executor or registry core.
    """

    def __init__(self) -> None:
        """Initializes the tool registry maps."""
        self._tools: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """A decorator to register a function as an agent tool.

        Args:
            name: The unique string identifier for the tool.

        Returns:
            The decorator function wrapping the tool function.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._tools:
                logger.warning(
                    f"Overwriting existing tool registration: '{name}' (old: {self._tools[name].__name__}, new: {func.__name__})"
                )
            self._tools[name] = func
            logger.info(
                f"Successfully registered tool: '{name}' linked to function: '{func.__name__}'."
            )
            return func

        return decorator

    def execute(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Executes a registered tool by its name.

        Args:
            name: The unique name of the tool to execute.
            *args: Position arguments for the tool.
            **kwargs: Keyword arguments for the tool.

        Returns:
            The output returned by the tool function.

        Raises:
            KeyError: If the tool is not registered.
            Exception: Re-raises any error thrown during tool execution.
        """
        if not self.has_tool(name):
            error_msg = f"Tool Execution Failure: Tool '{name}' is not registered in the Tool Registry."
            logger.error(error_msg)
            raise KeyError(error_msg)

        tool_func = self._tools[name]
        logger.debug(
            f"Invoking tool '{name}' via registry handler '{tool_func.__name__}'."
        )

        try:
            return tool_func(*args, **kwargs)
        except Exception as e:
            logger.error(
                f"Error encountered during tool '{name}' execution: {str(e)}"
            )
            raise e

    def has_tool(self, name: str) -> bool:
        """Checks if a tool name is registered in the registry.

        Args:
            name: Tool name string.

        Returns:
            True if the tool is registered, False otherwise.
        """
        return name in self._tools

    @property
    def tools(self) -> Dict[str, Callable[..., Any]]:
        """Returns a read-only copy of the registered tools mapping."""
        return self._tools.copy()


# Singleton global instance of the Tool Registry to be used across the application
tool_registry = ToolRegistry()
