"""
Persistent workspace memory — mirrors src/workspace/.

Provides hybrid search (full-text + semantic) over stored documents.
The default backend is an in-memory store; swap for a PostgreSQL +
pgvector backend for production.

When ``base_dir`` is supplied the workspace is backed by the local
filesystem — each document maps to a file under that directory.  This is
what makes ``load_identity_context`` actually useful: place AGENTS.md,
SOUL.md, etc. in the workspace directory and they will be injected into
every system prompt.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WorkspaceDocument:
    """A stored document in the workspace."""

    path: str
    content: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


class Workspace:
    """
    Persistent memory with hybrid search.

    Mirrors the Rust ``Workspace`` struct.  In production, back this with
    PostgreSQL + pgvector for proper vector search.  The in-memory
    implementation supports all four tool operations: search, write, read,
    and tree listing.

    Parameters
    ----------
    base_dir:
        Optional path to a directory on disk.  When set, all reads/writes
        are transparently backed by the filesystem so that identity files
        (AGENTS.md, SOUL.md, …) placed there persist across restarts.
        The in-memory cache is still used to avoid repeated disk reads
        within a single session.
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self._docs: dict[str, WorkspaceDocument] = {}
        self._base_dir: Path | None = Path(base_dir) if base_dir else None
        if self._base_dir:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            logger.debug("Workspace base_dir: %s", self._base_dir)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Filesystem helpers
    # ------------------------------------------------------------------

    def _fs_path(self, path: str) -> Path | None:
        """Return the absolute filesystem path for *path*, or None if no base_dir."""
        if self._base_dir is None:
            return None
        # Prevent path traversal
        rel = Path(path)
        if rel.is_absolute() or ".." in rel.parts:
            return None
        return self._base_dir / rel

    async def _fs_read(self, path: str) -> str | None:
        """Read *path* from the filesystem backend, or return None."""
        fpath = self._fs_path(path)
        if fpath is None or not fpath.exists():
            return None
        try:
            return fpath.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Workspace fs_read failed for %s: %s", path, exc)
            return None

    async def _fs_write(self, path: str, content: str) -> None:
        """Write *content* to *path* in the filesystem backend."""
        fpath = self._fs_path(path)
        if fpath is None:
            return
        fpath.parent.mkdir(parents=True, exist_ok=True)
        try:
            fpath.write_text(content, encoding="utf-8")
        except OSError as exc:
            logger.warning("Workspace fs_write failed for %s: %s", path, exc)

    async def _fs_delete(self, path: str) -> bool:
        """Delete *path* from the filesystem. Returns True if it existed."""
        fpath = self._fs_path(path)
        if fpath is None or not fpath.exists():
            return False
        try:
            fpath.unlink()
            return True
        except OSError as exc:
            logger.warning("Workspace fs_delete failed for %s: %s", path, exc)
            return False

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def write(self, path: str, content: str, metadata: dict | None = None) -> None:
        """Write or overwrite a document at path."""
        doc = self._docs.get(path)
        if doc:
            doc.content = content
            doc.updated_at = datetime.now(timezone.utc)
            if metadata:
                doc.metadata.update(metadata)
        else:
            self._docs[path] = WorkspaceDocument(
                path=path,
                content=content,
                metadata=metadata or {},
            )
        await self._fs_write(path, content)
        logger.debug("Workspace write: %s (%d bytes)", path, len(content))

    async def read(self, path: str) -> str | None:
        """Read a document. Returns None if not found."""
        doc = self._docs.get(path)
        if doc:
            return doc.content
        # Fall back to filesystem
        content = await self._fs_read(path)
        if content is not None:
            self._docs[path] = WorkspaceDocument(path=path, content=content)
        return content

    async def delete(self, path: str) -> bool:
        """Delete a document. Returns True if it existed."""
        in_memory = path in self._docs
        if in_memory:
            del self._docs[path]
        on_disk = await self._fs_delete(path)
        return in_memory or on_disk

    async def append(self, path: str, content: str) -> None:
        """
        Append content to an existing document, or create it if absent.

        Used by the compaction system to write to daily log files
        (e.g. ``daily/2026-03-16.md``).  Mirrors the Rust ``Workspace::append``.
        """
        existing = await self.read(path) or ""
        new_content = existing + content
        doc = self._docs.get(path)
        if doc:
            doc.content = new_content
            doc.updated_at = datetime.now(timezone.utc)
        else:
            self._docs[path] = WorkspaceDocument(path=path, content=new_content)
        await self._fs_write(path, new_content)
        logger.debug("Workspace append: %s (+%d bytes)", path, len(content))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[tuple[str, str, float]]:
        """
        Full-text search over workspace documents.

        Returns list of (path, snippet, score) tuples, highest score first.
        Production implementation should use RRF over FTS + pgvector.
        """
        query_lower = query.lower()
        results: list[tuple[str, str, float]] = []

        for path, doc in self._docs.items():
            content_lower = doc.content.lower()
            # Simple TF-style scoring: count query term occurrences
            score = content_lower.count(query_lower) + (1.0 if query_lower in path.lower() else 0)
            if score > 0:
                snippet = doc.content[:300]
                results.append((path, snippet, float(score)))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:limit]

    # ------------------------------------------------------------------
    # Tree listing
    # ------------------------------------------------------------------

    async def tree(self, prefix: str = "") -> list[str]:
        """List all document paths, optionally filtered by prefix."""
        paths: set[str] = {p for p in self._docs if p.startswith(prefix)}
        # Also enumerate files from the filesystem backend
        if self._base_dir and self._base_dir.exists():
            for fpath in self._base_dir.rglob("*"):
                if fpath.is_file():
                    rel = str(fpath.relative_to(self._base_dir))
                    if rel.startswith(prefix):
                        paths.add(rel)
        return sorted(paths)

    # ------------------------------------------------------------------
    # Identity files (injected into system prompt)
    # ------------------------------------------------------------------

    async def load_identity_context(self) -> str:
        """
        Build the identity/context block from well-known identity files.

        Mirrors the Rust ``Workspace::identity_context()`` — reads
        AGENTS.md, SOUL.md, USER.md, IDENTITY.md, MEMORY.md and
        concatenates them into a string for the system prompt.
        """
        identity_paths = ["AGENTS.md", "SOUL.md", "USER.md", "IDENTITY.md", "MEMORY.md"]
        parts: list[str] = []

        for path in identity_paths:
            content = await self.read(path)
            if content:
                parts.append(f"## {path}\n\n{content}")

        return "\n\n".join(parts)
