"""Per-job bearer token store.

Each Docker sandbox job receives a unique bearer token that it uses to
authenticate requests to the orchestrator.  Tokens are generated with
``secrets.token_urlsafe`` and are never reused across jobs.

Mirrors ``src/orchestrator/auth.rs`` in the Rust ironclaw codebase.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass, field


@dataclass
class JobToken:
    """A bearer token bound to a single sandbox job."""

    job_id: str
    token: str
    created_at: float = field(default_factory=time.time)
    session_id: str = ""


class TokenStore:
    """
    Thread-safe store for per-job bearer tokens.

    Usage::

        store = TokenStore()
        token = store.issue("job-123", session_id="sess-abc")
        # Pass token to the container; container authenticates with it.
        job_id = store.verify(token)  # returns "job-123" or None
        store.revoke("job-123")       # called after job completes
    """

    def __init__(self) -> None:
        self._tokens: dict[str, JobToken] = {}   # token → JobToken
        self._by_job: dict[str, str] = {}        # job_id → token
        self._lock = asyncio.Lock()

    async def issue(self, job_id: str, session_id: str = "") -> str:
        """Generate and store a new token for *job_id*.  Returns the token string."""
        token = secrets.token_urlsafe(32)
        entry = JobToken(job_id=job_id, token=token, session_id=session_id)
        async with self._lock:
            # Revoke any existing token for this job
            old_token = self._by_job.get(job_id)
            if old_token:
                self._tokens.pop(old_token, None)
            self._tokens[token] = entry
            self._by_job[job_id] = token
        return token

    async def verify(self, token: str) -> JobToken | None:
        """Return the ``JobToken`` for *token*, or ``None`` if invalid."""
        async with self._lock:
            return self._tokens.get(token)

    async def revoke(self, job_id: str) -> None:
        """Remove all tokens associated with *job_id*."""
        async with self._lock:
            token = self._by_job.pop(job_id, None)
            if token:
                self._tokens.pop(token, None)

    async def revoke_all(self) -> None:
        """Remove all tokens (call on server shutdown)."""
        async with self._lock:
            self._tokens.clear()
            self._by_job.clear()

    # Sync variant for use outside async contexts (e.g. from thread pool)
    def verify_sync(self, token: str) -> JobToken | None:
        return self._tokens.get(token)
