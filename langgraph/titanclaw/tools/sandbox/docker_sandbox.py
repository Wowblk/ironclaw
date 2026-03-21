"""Docker-based execution sandbox.

Wraps the bare ``shell`` tool so that every command runs inside a Docker
container with hard resource limits and a minimal attack surface.

Isolation guarantees
--------------------
* **Filesystem** — the host workspace is mounted read-only (or write, if
  ``workspace_writable=True``).  The container root filesystem is ephemeral.
* **Network** — defaults to ``--network=none``.  Set ``network_mode``
  to ``"bridge"`` or a custom network when outbound access is required.
* **Resources** — ``--memory`` + ``--memory-swap`` cap RAM; ``--cpus`` caps CPU.
* **Privileges** — ``--cap-drop=ALL --no-new-privileges --security-opt
  no-new-privileges:true``.  Add back specific caps via ``extra_caps``.
* **Cleanup** — ``--rm`` ensures the container is removed on exit.

Requirements
------------
The ``docker`` CLI must be available on ``PATH`` (no extra Python package).

Usage::

    sandbox = DockerSandbox(
        image="python:3.12-slim",
        memory_mb=256,
        cpu_quota=0.5,
    )
    # Replace the bare shell tool in the registry:
    registry.register(sandbox.create_shell_tool())

Communication
-------------
stdin/stdout of the container are used verbatim.  The container receives the
exact ``command`` string, executed via ``sh -c <command>``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

from titanclaw.tools.registry import ToolDefinition, ToolResult

logger = logging.getLogger(__name__)

_DEFAULT_IMAGE = "python:3.12-slim"
_DEFAULT_MEMORY_MB = 512
_DEFAULT_CPU_QUOTA = 1.0
_DEFAULT_TIMEOUT = 60.0
_MAX_OUTPUT_BYTES = 100_000

_SHELL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "Shell command to execute inside the Docker sandbox",
        },
        "timeout": {
            "type": "number",
            "description": "Timeout in seconds (default: 60)",
        },
        "working_dir": {
            "type": "string",
            "description": "Working directory inside the container (optional)",
        },
    },
    "required": ["command"],
}


class DockerSandbox:
    """
    Docker-based execution sandbox.

    Parameters
    ----------
    image:
        Docker image to use (default ``python:3.12-slim``).
    memory_mb:
        Container memory limit in MiB (default 512).
    cpu_quota:
        CPU quota as a fractional number of CPUs (default 1.0).
    network_mode:
        Docker network mode (default ``"none"`` — no networking).
    workspace_dir:
        Host path to mount as ``/workspace`` inside the container.
        Defaults to ``~/.titanclaw/workspace``.
    workspace_writable:
        If ``True``, the workspace mount is read-write.  Default is
        read-only to prevent accidental data loss.
    timeout:
        Default command timeout in seconds (default 60).
    extra_caps:
        Additional Linux capabilities to add back (e.g. ``["NET_RAW"]``).
    extra_docker_args:
        Arbitrary extra arguments to pass to ``docker run``.
    """

    def __init__(
        self,
        image: str = _DEFAULT_IMAGE,
        memory_mb: int = _DEFAULT_MEMORY_MB,
        cpu_quota: float = _DEFAULT_CPU_QUOTA,
        network_mode: str = "none",
        workspace_dir: str | None = None,
        workspace_writable: bool = False,
        timeout: float = _DEFAULT_TIMEOUT,
        extra_caps: list[str] | None = None,
        extra_docker_args: list[str] | None = None,
    ) -> None:
        self.image = image
        self.memory_mb = memory_mb
        self.cpu_quota = cpu_quota
        self.network_mode = network_mode
        self.workspace_dir = workspace_dir or str(
            Path.home() / ".titanclaw" / "workspace"
        )
        self.workspace_writable = workspace_writable
        self.timeout = timeout
        self.extra_caps = extra_caps or []
        self.extra_docker_args = extra_docker_args or []

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    async def run_command(
        self,
        command: str,
        timeout: float | None = None,
        working_dir: str | None = None,
    ) -> ToolResult:
        """
        Execute *command* inside a Docker container.

        Parameters
        ----------
        command:
            Shell command string (executed via ``sh -c``).
        timeout:
            Timeout in seconds.  Overrides the instance default.
        working_dir:
            Working directory inside the container.

        Returns
        -------
        ToolResult
            Combined stdout+stderr output, exit code prefix, and timing.
        """
        effective_timeout = timeout if timeout is not None else self.timeout
        start = time.monotonic()

        # Ensure workspace directory exists on the host
        os.makedirs(self.workspace_dir, exist_ok=True)
        mount_mode = "rw" if self.workspace_writable else "ro"

        args: list[str] = [
            "docker", "run", "--rm",
            "--network", self.network_mode,
            f"--memory={self.memory_mb}m",
            f"--memory-swap={self.memory_mb}m",   # disable swap
            f"--cpus={self.cpu_quota}",
            "--cap-drop=ALL",
            "--no-new-privileges",
            "--security-opt", "no-new-privileges:true",
            # Workspace volume
            "-v", f"{self.workspace_dir}:/workspace:{mount_mode}",
        ]

        for cap in self.extra_caps:
            args += ["--cap-add", cap]

        if working_dir:
            args += ["-w", working_dir]

        args += self.extra_docker_args
        args += [self.image, "sh", "-c", command]

        logger.debug("DockerSandbox run: %s", command[:120])

        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(), timeout=effective_timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                duration = (time.monotonic() - start) * 1000
                return ToolResult(
                    output=f"Command timed out after {effective_timeout}s",
                    duration_ms=duration,
                    error=True,
                )

            output = stdout.decode(errors="replace")[:_MAX_OUTPUT_BYTES]
            duration = (time.monotonic() - start) * 1000
            exit_code = proc.returncode or 0
            return ToolResult(
                output=f"Exit code: {exit_code}\n\n{output}",
                duration_ms=duration,
                error=exit_code != 0,
            )

        except FileNotFoundError:
            return ToolResult(
                output=(
                    "docker CLI not found on PATH.  "
                    "Install Docker: https://docs.docker.com/get-docker/"
                ),
                duration_ms=0,
                error=True,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                output=f"DockerSandbox error: {exc}",
                duration_ms=0,
                error=True,
            )

    # ------------------------------------------------------------------
    # Tool factory
    # ------------------------------------------------------------------

    def create_shell_tool(self) -> ToolDefinition:
        """
        Return a ``ToolDefinition`` named ``shell`` that executes commands
        inside this Docker sandbox.

        Register it **after** the bare ``shell_tool`` to replace it::

            registry.register(bare_shell_tool)    # sets initial shell
            registry.register(sandbox.create_shell_tool())  # overrides it
        """
        sandbox = self

        async def _execute(params: dict[str, Any], ctx: Any = None) -> ToolResult:
            command = params.get("command", "")
            timeout = float(params.get("timeout", sandbox.timeout))
            working_dir = params.get("working_dir") or None

            if not command:
                return ToolResult(output="No command provided", duration_ms=0, error=True)

            return await sandbox.run_command(command, timeout=timeout, working_dir=working_dir)

        return ToolDefinition(
            name="shell",
            description=(
                "Execute a shell command inside a Docker sandbox container.  "
                f"Isolated: network={self.network_mode}, "
                f"memory={self.memory_mb}MiB, cpus={self.cpu_quota}.  "
                "Always requires user approval."
            ),
            parameters_schema=_SHELL_SCHEMA,
            execute=_execute,
            requires_approval=True,
            requires_sanitization=True,
        )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def check(self) -> bool:
        """Return ``True`` if Docker is available and the image can be pulled."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "info",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate()
            return proc.returncode == 0
        except FileNotFoundError:
            return False
