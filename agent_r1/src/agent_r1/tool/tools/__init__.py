def _default_tool(name):
    if name == "search":
        from agent_r1.tool.tools.search_tool import SearchTool
        return SearchTool()
    elif name == "wiki_search":
        from agent_r1.tool.tools.wiki_search_tool import WikiSearchTool
        return WikiSearchTool()
    elif name == "python":
        from agent_r1.tool.tools.python_tool import PythonTool
        return PythonTool()
    else:
        raise NotImplementedError(f"Tool {name} not implemented")