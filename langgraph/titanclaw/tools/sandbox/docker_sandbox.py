"""Docker-based execution sandbox.

Wraps the bare ``shell`` tool so that every command runs inside a Docker
container with hard resource limits and a minimal attack surface.

Isolation guarantees
--------------------
* **Filesystem** — the host workspace is mounted read-only (or read-write for
  ``WorkspaceWrite`` / ``FullAccess`` policies).  The container root filesystem
  is read-only by default; ``/tmp`` is a size-limited ``tmpfs``.
* **Privileges** — ``--cap-drop=ALL --no-new-privileges``.  Container runs as
  non-root (UID/GID 1000).
* **Resources** — ``--memory`` + ``--memory-swap`` cap RAM; ``--cpus`` caps CPU.
* **Network** — defaults to ``--network=none``.  When ``allowed_domains`` is
  non-empty the sandbox starts a local HTTP proxy (see ``network_proxy``) and
  routes container traffic through it, allowing only the listed hosts.
* **Credentials** — when using the network proxy, API credentials are injected
  at the proxy level and *never* passed into the container.
* **Cleanup** — ``--rm`` ensures the container is removed on exit.

Sandbox policies
----------------
``ReadOnly``
    Workspace mounted read-only.  Network via proxy allowlist (or ``none``).
    This is the safe default.

``WorkspaceWrite``
    Workspace mounted read-write.  Network via proxy allowlist (or ``none``).
    Use when the tool must write output files.

``FullAccess``
    Direct host execution — **NO SANDBOX**.  Requires
    *both* ``policy=FullAccess`` and the environment variable
    ``SANDBOX_ALLOW_FULL_ACCESS=true``.  Logged at WARNING level.

Requirements
------------
The ``docker`` CLI must be available on ``PATH`` (no extra Python package).

Usage::

    sandbox = DockerSandbox(
        image="python:3.12-slim",
        memory_mb=256,
        cpu_quota=0.5,
        policy=SandboxPolicy.READ_ONLY,
        allowed_domains=["pypi.org", "*.github.com"],
    )
    registry.register(sandbox.create_shell_tool())
"""

from __future__ import annotations

import asyncio
import enum
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
_MAX_RETRIES = 2  # total retry attempts for transient errors

# /tmp inside the container is a size-limited tmpfs (matches Rust 512 MiB default)
_TMPFS_TMP = "/tmp:rw,noexec,nosuid,size=512m"

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

# Errors that are safe to retry (transient infrastructure failures)
_TRANSIENT_ERRORS = (
    "docker: Error response from daemon",
    "cannot connect to the Docker daemon",
    "container exited with code 125",  # docker run startup failure
)


class SandboxPolicy(str, enum.Enum):
    """
    Three-tier isolation model (mirrors ``SandboxPolicy`` in ``src/sandbox/config.rs``).

    ``READ_ONLY``
        Safe default.  Workspace mounted read-only; no network except via
        explicit domain allowlist proxy.

    ``WORKSPACE_WRITE``
        Workspace mounted read-write; same network restrictions as ReadOnly.

    ``FULL_ACCESS``
        **DANGEROUS — no container isolation**.  Requires
        ``policy=FullAccess`` AND ``SANDBOX_ALLOW_FULL_ACCESS=true`` env var.
    """

    READ_ONLY = "read_only"
    WORKSPACE_WRITE = "workspace_write"
    FULL_ACCESS = "full_access"


def _is_transient_error(output: str) -> bool:
    lo = output.lower()
    return any(sig.lower() in lo for sig in _TRANSIENT_ERRORS)


class DockerSandbox:
    """
    Docker-based execution sandbox.

    Parameters
    ----------
    image:
        Docker image to use (default ``python:3.12-slim``).
    memory_mb:
        Container memory limit in MiB (default 512).  Swap is disabled.
    cpu_quota:
        CPU quota as a fractional number of CPUs (default 1.0).
    policy:
        Isolation policy: ``READ_ONLY``, ``WORKSPACE_WRITE``, or
        ``FULL_ACCESS``.  Defaults to ``READ_ONLY``.
    network_mode:
        Docker network mode.  Ignored (overridden to ``"bridge"``) when
        ``allowed_domains`` is non-empty.  Default ``"none"``.
    workspace_dir:
        Host path to mount as ``/workspace``.  Defaults to
        ``~/.titanclaw/workspace``.
    allowed_domains:
        Whitelist of host patterns (exact or ``*.example.com``).  When
        non-empty a local HTTP proxy is started and container traffic is
        routed through it; only the listed hosts are reachable.  Implies
        ``network_mode="bridge"``.
    credential_mappings:
        Credential injection rules.  List of dicts with keys
        ``host_pattern``, ``secret_env_var``, and ``location``
        (``"bearer"`` | ``"header:<name>"`` | ``"query:<param>"``).
        Secrets are resolved from host environment variables and injected
        at the proxy layer — never passed into the container.
    timeout:
        Default command timeout in seconds (default 60).
    extra_caps:
        Additional Linux capabilities to re-add (e.g. ``["NET_RAW"]``).
    extra_docker_args:
        Arbitrary extra arguments appended to ``docker run``.
    """

    def __init__(
        self,
        image: str = _DEFAULT_IMAGE,
        memory_mb: int = _DEFAULT_MEMORY_MB,
        cpu_quota: float = _DEFAULT_CPU_QUOTA,
        policy: SandboxPolicy = SandboxPolicy.READ_ONLY,
        network_mode: str = "none",
        workspace_dir: str | None = None,
        allowed_domains: list[str] | None = None,
        credential_mappings: list[dict[str, str]] | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        extra_caps: list[str] | None = None,
        extra_docker_args: list[str] | None = None,
    ) -> None:
        self.image = image
        self.memory_mb = memory_mb
        self.cpu_quota = cpu_quota
        self.policy = policy
        self.network_mode = network_mode
        self.workspace_dir = str(
            Path(workspace_dir or Path.home() / ".titanclaw" / "workspace").expanduser().resolve()
        )
        self.allowed_domains = allowed_domains or []
        self.credential_mappings = credential_mappings or []
        self.timeout = timeout
        self.extra_caps = extra_caps or []
        self.extra_docker_args = extra_docker_args or []

        # Proxy server, started lazily on first call when allowed_domains is set
        self._proxy: Any | None = None          # HttpProxyServer instance
        self._proxy_port: int | None = None
        self._proxy_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Proxy lifecycle
    # ------------------------------------------------------------------

    async def _ensure_proxy(self) -> int | None:
        """Start the HTTP proxy if allowed_domains is configured; return its port."""
        if not self.allowed_domains:
            return None
        async with self._proxy_lock:
            if self._proxy_port is not None:
                return self._proxy_port
            try:
                from titanclaw.tools.sandbox.network_proxy import HttpProxyServer
                proxy = HttpProxyServer(
                    allowed_domains=self.allowed_domains,
                    credential_mappings=self.credential_mappings,
                )
                port = await proxy.start()
                self._proxy = proxy
                self._proxy_port = port
                logger.info(
                    "DockerSandbox network proxy started on port %d "
                    "(allowed: %s)",
                    port,
                    ", ".join(self.allowed_domains[:5]),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to start network proxy: %s — falling back to --network=none", exc)
                self._proxy_port = -1   # sentinel: failed
        return self._proxy_port if self._proxy_port and self._proxy_port > 0 else None

    async def stop_proxy(self) -> None:
        """Stop the background proxy server (call on application shutdown)."""
        if self._proxy is not None:
            await self._proxy.stop()
            self._proxy = None
            self._proxy_port = None

    # ------------------------------------------------------------------
    # FullAccess guard
    # ------------------------------------------------------------------

    def _check_full_access(self) -> bool:
        """
        Return True if FullAccess is legitimately enabled (double opt-in).

        Mirrors the guard in ``src/sandbox/manager.rs`` lines 210–231.
        """
        if self.policy != SandboxPolicy.FULL_ACCESS:
            return False
        if not os.environ.get("SANDBOX_ALLOW_FULL_ACCESS", "").lower() in ("true", "1", "yes"):
            logger.error(
                "DockerSandbox policy=FULL_ACCESS requested but "
                "SANDBOX_ALLOW_FULL_ACCESS env var is not set.  "
                "Downgrading to WORKSPACE_WRITE for safety.  "
                "Set SANDBOX_ALLOW_FULL_ACCESS=true to explicitly enable "
                "unsandboxed execution."
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    async def run_command(
        self,
        command: str,
        timeout: float | None = None,
        working_dir: str | None = None,
    ) -> ToolResult:
        """Execute *command* with up to ``_MAX_RETRIES`` retries on transient errors."""
        last_result: ToolResult | None = None
        delay = 2.0
        for attempt in range(_MAX_RETRIES + 1):
            result = await self._run_once(command, timeout=timeout, working_dir=working_dir)
            if not result.error or not _is_transient_error(result.output):
                return result
            last_result = result
            if attempt < _MAX_RETRIES:
                logger.warning(
                    "DockerSandbox transient error (attempt %d/%d), retrying in %.0fs: %s",
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    delay,
                    result.output[:120],
                )
                await asyncio.sleep(delay)
                delay *= 2  # exponential backoff: 2s → 4s
        return last_result  # type: ignore[return-value]

    async def _run_once(
        self,
        command: str,
        timeout: float | None = None,
        working_dir: str | None = None,
    ) -> ToolResult:
        effective_timeout = timeout if timeout is not None else self.timeout
        start = time.monotonic()

        # --- FullAccess: direct host execution (no container) ---
        if self.policy == SandboxPolicy.FULL_ACCESS and self._check_full_access():
            logger.warning(
                "DockerSandbox FULL_ACCESS: executing directly on host (no isolation): %s",
                command[:80],
            )
            try:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=working_dir,
                )
                try:
                    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=effective_timeout)
                except asyncio.TimeoutError:
                    proc.kill()
                    return ToolResult(
                        output=f"Command timed out after {effective_timeout}s",
                        duration_ms=(time.monotonic() - start) * 1000,
                        error=True,
                    )
                output = stdout.decode(errors="replace")[:_MAX_OUTPUT_BYTES]
                exit_code = proc.returncode or 0
                return ToolResult(
                    output=f"Exit code: {exit_code}\n\n{output}",
                    duration_ms=(time.monotonic() - start) * 1000,
                    error=exit_code != 0,
                )
            except Exception as exc:  # noqa: BLE001
                return ToolResult(output=f"Host execution error: {exc}", duration_ms=0, error=True)

        # --- Sandboxed container execution ---
        os.makedirs(self.workspace_dir, exist_ok=True)

        effective_policy = (
            SandboxPolicy.WORKSPACE_WRITE
            if self.policy == SandboxPolicy.FULL_ACCESS
            else self.policy
        )
        workspace_writable = effective_policy == SandboxPolicy.WORKSPACE_WRITE

        # Determine network settings
        proxy_port = await self._ensure_proxy()
        if proxy_port:
            actual_network = "bridge"
            proxy_env = [
                "-e", f"HTTP_PROXY=http://host-gateway:{proxy_port}",
                "-e", f"HTTPS_PROXY=http://host-gateway:{proxy_port}",
                "-e", f"http_proxy=http://host-gateway:{proxy_port}",
                "-e", f"https_proxy=http://host-gateway:{proxy_port}",
                "--add-host=host-gateway:host-gateway",
            ]
        else:
            actual_network = self.network_mode
            proxy_env = []

        mount_mode = "rw" if workspace_writable else "ro"

        args: list[str] = [
            "docker", "run", "--rm",
            # Resource limits
            "--network", actual_network,
            f"--memory={self.memory_mb}m",
            f"--memory-swap={self.memory_mb}m",   # disable swap
            f"--cpus={self.cpu_quota}",
            # Privilege restrictions
            "--cap-drop=ALL",
            "--no-new-privileges",
            "--security-opt", "no-new-privileges:true",
            # Non-root execution (mirrors Rust UID 1000:1000)
            "--user=1000:1000",
            # Read-only root filesystem
            "--read-only",
            # /tmp as size-limited tmpfs (replaces the now-read-only /tmp on rootfs)
            "--tmpfs", _TMPFS_TMP,
            # Workspace volume
            "-v", f"{self.workspace_dir}:/workspace:{mount_mode}",
        ]

        args += proxy_env

        for cap in self.extra_caps:
            args += ["--cap-add", cap]

        if working_dir:
            args += ["-w", working_dir]

        args += self.extra_docker_args
        args += [self.image, "sh", "-c", command]

        logger.debug("DockerSandbox(%s) run: %s", effective_policy.value, command[:120])

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
                return ToolResult(
                    output=f"Command timed out after {effective_timeout}s",
                    duration_ms=(time.monotonic() - start) * 1000,
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

            registry.register(shell_tool)
            registry.register(sandbox.create_shell_tool())
        """
        sandbox = self
        net_info = (
            f"allowed domains: {', '.join(self.allowed_domains[:3])}{'…' if len(self.allowed_domains) > 3 else ''}"
            if self.allowed_domains
            else f"network={self.network_mode}"
        )

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
                f"Execute a shell command inside a Docker sandbox container "
                f"(policy={self.policy.value}, "
                f"memory={self.memory_mb}MiB, cpus={self.cpu_quota}, "
                f"{net_info}).  Always requires user approval."
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
        """Return ``True`` if Docker is available and the daemon is running."""
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
