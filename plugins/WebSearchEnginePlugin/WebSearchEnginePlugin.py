class WebSearchEnginePlugin:
    """
    A search engine plugin.
    """

    from semantic_kernel.orchestration.kernel_context import KernelContext
    from semantic_kernel.plugin_definition import (
        kernel_function,
        kernel_function_context_parameter,
    )

    def __init__(self, connector) -> None:
        self._connector = connector

    @kernel_function(description="Performs a web search for a given query", name="searchAsync")
    @kernel_function_context_parameter(
        name="query",
        description="The search query",
    )
    async def search(self, query: str, context: KernelContext) -> str:
        query = query or context.variables.get("query")[1]
        result = await self._connector.search(query, num_results=5, offset=0)
        return str(result)