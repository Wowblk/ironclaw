"""
Supermemory-backed workspace — cloud semantic memory.

Extends the base ``Workspace`` with Supermemory cloud storage so that
``search()`` uses real vector/semantic search instead of keyword matching.
``write()`` and ``append()`` mirror content to both the local in-memory
store (for fast ``read()`` / ``tree()``) and to Supermemory (for semantic
recall).

Usage::

    from ironclaw.memory import SupermemoryWorkspace

    ws = SupermemoryWorkspace(
        api_key=os.environ["SUPERMEMORY_API_KEY"],
        container_tag="ironclaw",      # namespace per user / deployment
    )

Requires the ``supermemory`` package::

    pip install supermemory

Fall-back behaviour
-------------------
Every Supermemory call is wrapped in a try/except.  If the API is
unavailable or the key is missing, the method falls back silently to the
in-memory implementation so the agent keeps running.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ironclaw.memory.workspace import Workspace

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Sentinel so we only log the import-error once.
_IMPORT_WARNED = False


def _get_async_client(api_key: str) -> Any:
    """Import and return an ``AsyncSupermemory`` client."""
    global _IMPORT_WARNED  # noqa: PLW0603
    try:
        from supermemory import AsyncSupermemory  # type: ignore[import-untyped]

        return AsyncSupermemory(api_key=api_key)
    except ImportError:
        if not _IMPORT_WARNED:
            logger.warning(
                "supermemory package not installed — falling back to in-memory workspace. "
                "Run: pip install supermemory"
            )
            _IMPORT_WARNED = True
        return None


class SupermemoryWorkspace(Workspace):
    """
    Workspace backed by Supermemory cloud for semantic search.

    Inherits all CRUD from ``Workspace`` (in-memory dict) and overrides
    ``write``, ``append``, and ``search`` to also talk to Supermemory.

    Parameters
    ----------
    api_key:
        Supermemory API key (from console.supermemory.ai).
    container_tag:
        Namespace tag that scopes memories — use per-user or per-deployment
        values to keep memories isolated.  Defaults to ``"ironclaw"``.
    """

    # Prefix used to embed the path inside stored content so we can
    # reconstruct it on retrieval.
    _PATH_PREFIX = "__path__:"
    _PATH_SEP = "\n\n"

    def __init__(self, api_key: str, container_tag: str = "ironclaw") -> None:
        super().__init__()
        self._api_key = api_key
        self._container_tag = container_tag
        # Lazy-initialised client — created once on first use.
        self._client: Any | None = None

    # ------------------------------------------------------------------
    # Client access
    # ------------------------------------------------------------------

    def _get_client(self) -> Any | None:
        if self._client is None:
            self._client = _get_async_client(self._api_key)
        return self._client

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode(self, path: str, content: str) -> str:
        """Wrap content with path metadata for round-trip decoding."""
        return f"{self._PATH_PREFIX}{path}{self._PATH_SEP}{content}"

    def _decode(self, raw: str) -> tuple[str, str]:
        """
        Extract (path, snippet) from stored content.

        Falls back to (``"memory"``, raw[:300]) for content written
        without the path prefix (e.g. by older code).
        """
        if raw.startswith(self._PATH_PREFIX):
            rest = raw[len(self._PATH_PREFIX):]
            if self._PATH_SEP in rest:
                path, body = rest.split(self._PATH_SEP, 1)
                return path.strip(), body[:300]
        return "memory", raw[:300]

    # ------------------------------------------------------------------
    # Overrides: write / append  (mirror to Supermemory)
    # ------------------------------------------------------------------

    async def write(self, path: str, content: str, metadata: dict | None = None) -> None:
        """Write to local store and push to Supermemory."""
        await super().write(path, content, metadata)
        await self._push(path, content)

    async def append(self, path: str, content: str) -> None:
        """Append to local store and push the chunk to Supermemory."""
        await super().append(path, content)
        await self._push(path, content)

    async def _push(self, path: str, content: str) -> None:
        """Push a content chunk to Supermemory (best-effort)."""
        client = self._get_client()
        if client is None:
            return
        try:
            await client.add(
                content=self._encode(path, content),
                container_tag=self._container_tag,
            )
            logger.debug("Supermemory push: %s (%d bytes)", path, len(content))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Supermemory push failed (local store intact): %s", exc)

    # ------------------------------------------------------------------
    # Override: search  (semantic via Supermemory)
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[tuple[str, str, float]]:
        """
        Semantic search via Supermemory.

        Returns list of (path, snippet, score) tuples, highest first.
        Falls back to the parent keyword-search when Supermemory is
        unavailable.
        """
        client = self._get_client()
        if client is None:
            return await super().search(query, limit)

        try:
            response = await client.search.memories(
                q=query,
                container_tag=self._container_tag,
            )
            # The SDK returns response.results (list of memory objects).
            raw_results = getattr(response, "results", None) or []
            results: list[tuple[str, str, float]] = []
            total = len(raw_results)
            for rank, item in enumerate(raw_results[:limit]):
                raw = getattr(item, "content", "") or ""
                path, snippet = self._decode(raw)
                # Score decays linearly from 1.0 to ~0 by rank.
                score = (total - rank) / max(total, 1)
                results.append((path, snippet, score))
            logger.debug(
                "Supermemory search '%s': %d results", query, len(results)
            )
            return results
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Supermemory search failed, falling back to local: %s", exc
            )
            return await super().search(query, limit)
