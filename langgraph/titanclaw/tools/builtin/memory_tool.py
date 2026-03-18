"""Memory/workspace tools — mirrors src/tools/builtin/memory.rs.

Two usage modes
---------------
1. **Module-level tools** (``memory_search_tool``, ``memory_write_tool``,
   ``memory_read_tool``): use a simple in-memory dict — suitable for
   quick testing without any external dependencies.

2. **Workspace-bound tools** via ``create_memory_tools(workspace)``:
   pass a ``Workspace`` (or ``SupermemoryWorkspace``) instance to get
   tools that delegate to it.  With ``SupermemoryWorkspace`` the
   ``memory_search`` tool gains real semantic search via Supermemory.

Example::

    from titanclaw.memory import SupermemoryWorkspace
    from titanclaw.tools.builtin.memory_tool import create_memory_tools

    ws = SupermemoryWorkspace(api_key=os.environ["SUPERMEMORY_API_KEY"])
    tools = create_memory_tools(ws)
    for t in tools:
        registry.register(t)
"""

from __future__ import annotations

import time
from typing import Any

from titanclaw.tools.registry import ToolDefinition, ToolResult

# ---------------------------------------------------------------------------
# In-process memory store (used by the module-level tool singletons)
# ---------------------------------------------------------------------------
_store: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Module-level tool singletons (keyword search, in-memory only)
# ---------------------------------------------------------------------------


async def _search_execute(params: dict[str, Any], ctx: Any = None) -> ToolResult:
    start = time.monotonic()
    query = params.get("query", "").lower()
    limit = int(params.get("limit", 10))
    results = [
        f"{path}: {content[:200]}"
        for path, content in _store.items()
        if query in content.lower() or query in path.lower()
    ][:limit]
    duration = (time.monotonic() - start) * 1000
    output = "\n---\n".join(results) if results else "No results found."
    return ToolResult(output=output, duration_ms=duration)


memory_search_tool = ToolDefinition(
    name="memory_search",
    description="Search persistent workspace memory using full-text search.",
    parameters_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Maximum results (default: 10)"},
        },
        "required": ["query"],
    },
    execute=_search_execute,
    requires_approval=False,
    requires_sanitization=True,
)


async def _write_execute(params: dict[str, Any], ctx: Any = None) -> ToolResult:
    start = time.monotonic()
    path = params.get("path", "")
    content = params.get("content", "")
    if not path:
        return ToolResult(output="Path is required", duration_ms=0, error=True)
    _store[path] = content
    duration = (time.monotonic() - start) * 1000
    return ToolResult(output=f"Saved {len(content)} bytes to '{path}'", duration_ms=duration)


memory_write_tool = ToolDefinition(
    name="memory_write",
    description="Write content to a named path in workspace memory.",
    parameters_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Memory path (e.g. 'notes/todo.md')"},
            "content": {"type": "string", "description": "Content to store"},
        },
        "required": ["path", "content"],
    },
    execute=_write_execute,
    requires_approval=False,
    requires_sanitization=False,
)


async def _read_execute(params: dict[str, Any], ctx: Any = None) -> ToolResult:
    start = time.monotonic()
    path = params.get("path", "")
    content = _store.get(path)
    duration = (time.monotonic() - start) * 1000
    if content is None:
        return ToolResult(output=f"Not found: '{path}'", duration_ms=duration, error=True)
    return ToolResult(output=content, duration_ms=duration)


memory_read_tool = ToolDefinition(
    name="memory_read",
    description="Read content from a named path in workspace memory.",
    parameters_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Memory path to read"},
        },
        "required": ["path"],
    },
    execute=_read_execute,
    requires_approval=False,
    requires_sanitization=True,
)


# ---------------------------------------------------------------------------
# Factory: workspace-bound tools (supports SupermemoryWorkspace)
# ---------------------------------------------------------------------------


def create_memory_tools(workspace: Any) -> list[ToolDefinition]:
    """
    Return memory tools bound to ``workspace``.

    When ``workspace`` is a ``SupermemoryWorkspace``, ``memory_search``
    uses Supermemory's semantic/vector search.  ``memory_write`` and
    ``memory_read`` use the workspace's local store (with Supermemory as
    a mirror for future recall).

    Parameters
    ----------
    workspace:
        A ``Workspace`` or ``SupermemoryWorkspace`` instance.

    Returns
    -------
    list[ToolDefinition]
        Three tools: ``memory_search``, ``memory_write``, ``memory_read``.
    """

    async def _ws_search(params: dict[str, Any], ctx: Any = None) -> ToolResult:
        start = time.monotonic()
        query = params.get("query", "")
        limit = int(params.get("limit", 10))
        if not query:
            return ToolResult(output="Query is required", duration_ms=0, error=True)
        hits = await workspace.search(query, limit=limit)
        duration = (time.monotonic() - start) * 1000
        if not hits:
            return ToolResult(output="No results found.", duration_ms=duration)
        lines = [f"{path}: {snippet}" for path, snippet, _score in hits]
        return ToolResult(output="\n---\n".join(lines), duration_ms=duration)

    async def _ws_write(params: dict[str, Any], ctx: Any = None) -> ToolResult:
        start = time.monotonic()
        path = params.get("path", "")
        content = params.get("content", "")
        if not path:
            return ToolResult(output="Path is required", duration_ms=0, error=True)
        await workspace.write(path, content)
        duration = (time.monotonic() - start) * 1000
        return ToolResult(output=f"Saved {len(content)} bytes to '{path}'", duration_ms=duration)

    async def _ws_read(params: dict[str, Any], ctx: Any = None) -> ToolResult:
        start = time.monotonic()
        path = params.get("path", "")
        content = await workspace.read(path)
        duration = (time.monotonic() - start) * 1000
        if content is None:
            return ToolResult(output=f"Not found: '{path}'", duration_ms=duration, error=True)
        return ToolResult(output=content, duration_ms=duration)

    return [
        ToolDefinition(
            name="memory_search",
            description="Search persistent workspace memory using semantic search.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Maximum results (default: 10)"},
                },
                "required": ["query"],
            },
            execute=_ws_search,
            requires_approval=False,
            requires_sanitization=True,
        ),
        ToolDefinition(
            name="memory_write",
            description="Write content to a named path in workspace memory.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Memory path (e.g. 'notes/todo.md')"},
                    "content": {"type": "string", "description": "Content to store"},
                },
                "required": ["path", "content"],
            },
            execute=_ws_write,
            requires_approval=False,
            requires_sanitization=False,
        ),
        ToolDefinition(
            name="memory_read",
            description="Read content from a named path in workspace memory.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Memory path to read"},
                },
                "required": ["path"],
            },
            execute=_ws_read,
            requires_approval=False,
            requires_sanitization=True,
        ),
    ]
