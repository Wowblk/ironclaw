"""Container orphan reaper.

Periodically scans for Docker containers created by the titanclaw sandbox
that are still running beyond the configured lifetime threshold, and forcibly
removes them.

Why
---
``docker run --rm`` removes a container only if the process exits normally.
If the Python process is killed (SIGKILL, OOM), the container is left running
forever.  The reaper is the backstop that prevents resource leaks.

Container identification
------------------------
Every container created by ``DockerSandbox`` is labelled with::

    titanclaw.sandbox = "true"
    titanclaw.session = <session_id | "unknown">

The reaper only removes containers that carry the ``titanclaw.sandbox=true``
label, so it never touches unrelated containers.

Mirrors ``SandboxManager`` reaper logic in ``src/sandbox/manager.rs`` with:
* ``reaper_interval_secs``  = 300  (5 minutes)
* ``orphan_threshold_secs`` = 600  (10 minutes)

Usage::

    reaper = ContainerOrphanReaper(interval_secs=300, threshold_secs=600)
    await reaper.start()
    # ... runs in background ...
    await reaper.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Label key applied to every sandbox container (value = "true")
SANDBOX_LABEL = "titanclaw.sandbox"
SESSION_LABEL = "titanclaw.session"

_DEFAULT_INTERVAL = 300     # seconds between reap passes
_DEFAULT_THRESHOLD = 600    # seconds a container may run before it's an orphan


class ContainerOrphanReaper:
    """
    Background asyncio task that kills stale sandbox containers.

    Parameters
    ----------
    interval_secs:
        How often (in seconds) to scan for orphans.  Default 300 (5 min).
    threshold_secs:
        A container running longer than this many seconds is considered an
        orphan and is forcibly removed.  Default 600 (10 min).
    """

    def __init__(
        self,
        interval_secs: int = _DEFAULT_INTERVAL,
        threshold_secs: int = _DEFAULT_THRESHOLD,
    ) -> None:
        self.interval_secs = interval_secs
        self.threshold_secs = threshold_secs
        self._task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background reaper task."""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._loop(), name="titanclaw-orphan-reaper")
        logger.info(
            "ContainerOrphanReaper started (interval=%ds, threshold=%ds)",
            self.interval_secs,
            self.threshold_secs,
        )

    async def stop(self) -> None:
        """Stop the background reaper task gracefully."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.debug("ContainerOrphanReaper stopped")

    # ------------------------------------------------------------------
    # Reaper loop
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.interval_secs)
                await self._reap_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning("OrphanReaper pass failed: %s", exc)

    async def _reap_once(self) -> None:
        """List sandbox containers; remove any that have exceeded the threshold."""
        containers = await self._list_sandbox_containers()
        now = time.time()
        reaped = 0
        for cid, started_at, session in containers:
            age = now - started_at
            if age >= self.threshold_secs:
                removed = await self._remove_container(cid)
                if removed:
                    logger.info(
                        "Reaped orphaned sandbox container %s (session=%s, age=%.0fs)",
                        cid[:12],
                        session,
                        age,
                    )
                    reaped += 1
        if reaped:
            logger.info("OrphanReaper pass: removed %d container(s)", reaped)
        else:
            logger.debug("OrphanReaper pass: no orphans found")

    # ------------------------------------------------------------------
    # Docker helpers
    # ------------------------------------------------------------------

    async def _list_sandbox_containers(self) -> list[tuple[str, float, str]]:
        """
        Return list of (container_id, start_unix_ts, session_id) for all
        running containers with the titanclaw.sandbox label.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "ps",
                "--filter", f"label={SANDBOX_LABEL}=true",
                "--format", "{{json .}}",
                "--no-trunc",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        except (FileNotFoundError, asyncio.TimeoutError):
            return []

        results: list[tuple[str, float, str]] = []
        for line in stdout.decode(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue

            cid = row.get("ID", "")
            if not cid:
                continue

            # Parse the started-at timestamp via docker inspect
            started_ts = await self._get_started_at(cid)
            if started_ts is None:
                continue

            # Extract session label
            labels_str = row.get("Labels", "")
            session = _extract_label(labels_str, SESSION_LABEL) or "unknown"
            results.append((cid, started_ts, session))

        return results

    async def _get_started_at(self, container_id: str) -> float | None:
        """Return Unix timestamp when the container was started, or None on error."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "inspect",
                "--format", "{{.State.StartedAt}}",
                container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        except (FileNotFoundError, asyncio.TimeoutError):
            return None

        raw = stdout.decode(errors="replace").strip()
        if not raw:
            return None

        # Format: 2024-01-15T12:34:56.789Z or 2024-01-15T12:34:56.789012345Z
        return _parse_docker_time(raw)

    async def _remove_container(self, container_id: str) -> bool:
        """Force-remove a container.  Return True on success."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", container_id,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.communicate(), timeout=15)
            return proc.returncode == 0
        except (FileNotFoundError, asyncio.TimeoutError):
            return False

    # ------------------------------------------------------------------
    # Immediate cleanup (called on sandbox shutdown)
    # ------------------------------------------------------------------

    async def reap_all(self) -> int:
        """
        Immediately remove **all** titanclaw sandbox containers regardless
        of age.  Used during graceful shutdown.

        Returns the number of containers removed.
        """
        containers = await self._list_sandbox_containers()
        count = 0
        for cid, _, session in containers:
            if await self._remove_container(cid):
                logger.info("Shutdown reap: removed container %s (session=%s)", cid[:12], session)
                count += 1
        return count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_docker_time(s: str) -> float | None:
    """
    Parse a Docker ISO-8601 timestamp to a Unix float.

    Docker returns timestamps like:
        ``2024-01-15T12:34:56.789012345Z``

    Python's ``datetime.fromisoformat`` (3.11+) handles most formats; we strip
    nanoseconds for older Python versions.
    """
    import datetime
    s = s.rstrip("Z")
    # Truncate sub-microsecond precision (Docker emits nanoseconds)
    if "." in s:
        base, frac = s.rsplit(".", 1)
        s = base + "." + frac[:6]
    try:
        dt = datetime.datetime.fromisoformat(s).replace(tzinfo=datetime.timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def _extract_label(labels_str: str, key: str) -> str | None:
    """
    Extract a label value from Docker's comma-separated label string.

    Format: ``key1=val1,key2=val2``
    """
    for pair in labels_str.split(","):
        pair = pair.strip()
        if pair.startswith(key + "="):
            return pair[len(key) + 1:]
    return None
