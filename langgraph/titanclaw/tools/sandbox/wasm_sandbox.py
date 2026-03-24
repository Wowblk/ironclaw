"""WASM/WASI execution sandbox using wasmtime.

Loads ``.wasm`` WASI modules from a directory and exposes each one as a
first-class registered tool.

Isolation guarantees
--------------------
* **Filesystem** — only the workspace directory (and a per-call temp directory
  for stdin/stdout pipes) are preopened.  The module cannot access the rest of
  the host filesystem.
* **Network** — WASI provides no socket API; modules have zero network access.
* **CPU** — ``wasmtime`` fuel metering caps total instruction count.  A module
  that loops forever is killed once it exhausts the fuel budget.
* **Memory** — the WASM linear memory is bounded by the module's declared
  maximum (or an explicit cap via the ``wasmtime.Config`` limits).
* **Syscalls** — WASI exposes only a narrow, well-defined set of capabilities.
  Modules cannot make arbitrary syscalls.

WASM tool protocol
------------------
Each ``.wasm`` WASI module must follow a simple JSON stdio protocol:

1. The sandbox writes a single UTF-8 JSON line to the module's **stdin**::

       {"params": {"arg1": "value", ...}}

2. The module writes a single UTF-8 JSON line to **stdout**::

       {"output": "result text", "error": false}

The tool's name is taken from the filename (without extension).
The tool description is read from a matching ``<name>.description.txt`` file in
the same directory, or defaults to ``"WASM tool: <name>"``.
The parameter schema is read from a matching ``<name>.schema.json`` file, or
defaults to an open ``{"type": "object", "additionalProperties": true}`` schema.

Requirements
------------
``wasmtime-py>=25`` — install with::

    pip install "titanclaw[wasm-sandbox]"

Usage::

    sandbox = WasmSandbox(tools_dir="~/.titanclaw/wasm-tools", fuel=500_000_000)
    for tool in sandbox.load_tools():
        registry.register(tool)

Building a WASM tool
--------------------
Any language that targets ``wasm32-wasi`` works.  A minimal Rust example::

    // src/main.rs
    use std::io::{self, BufRead, Write};

    fn main() {
        let stdin = io::stdin();
        let line = stdin.lock().lines().next().unwrap().unwrap();
        let req: serde_json::Value = serde_json::from_str(&line).unwrap();
        let params = &req["params"];
        // ... process params ...
        let out = serde_json::json!({"output": "Hello!", "error": false});
        println!("{}", out);
    }

Compile with ``cargo build --target wasm32-wasi --release``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from titanclaw.tools.registry import ToolDefinition, ToolResult

logger = logging.getLogger(__name__)

_DEFAULT_FUEL = 1_000_000_000       # ~1 billion instructions
_DEFAULT_TOOLS_DIR = "~/.titanclaw/wasm-tools"
_DEFAULT_TIMEOUT = 30.0             # wall-clock seconds (fuel is the primary CPU guard)
_DEFAULT_MAX_MEMORY_MB = 64         # WASM linear memory cap (Rust default is 10 MiB; we allow more)
_DEFAULT_MAX_TABLE_ELEMENTS = 10_000  # WASM table entries (matches Rust limits.rs)
_MAX_OUTPUT_BYTES = 100_000

_OPEN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
}


class WasmSandbox:
    """
    WASM/WASI-based tool sandbox.

    Parameters
    ----------
    tools_dir:
        Directory to scan for ``.wasm`` tool modules.
    fuel:
        Initial fuel units per call (default 1 billion).  Each WebAssembly
        instruction consumes one unit.  Set to ``0`` to disable fuel metering.
    max_memory_mb:
        Maximum WASM linear memory in MiB (default 64).  Memory growth beyond
        this is denied by the ``WasmResourceLimiter``.
    max_table_elements:
        Maximum number of WASM table entries (default 10,000).  Prevents
        unbounded indirect-call-table growth.
    workspace_dir:
        Host path that is preopened (read-write) inside the sandbox.
        Defaults to ``~/.titanclaw/workspace``.
    timeout:
        Wall-clock timeout per call in seconds (default 30).  The fuel cap
        is the primary CPU guard; this is a fallback.
    """

    def __init__(
        self,
        tools_dir: str = _DEFAULT_TOOLS_DIR,
        fuel: int = _DEFAULT_FUEL,
        max_memory_mb: int = _DEFAULT_MAX_MEMORY_MB,
        max_table_elements: int = _DEFAULT_MAX_TABLE_ELEMENTS,
        workspace_dir: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.tools_dir = str(Path(tools_dir).expanduser().resolve())
        self.fuel = fuel
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_table_elements = max_table_elements
        self.workspace_dir = str(
            Path(workspace_dir or Path.home() / ".titanclaw" / "workspace").expanduser().resolve()
        )
        self.timeout = timeout

        # Lazy-initialised wasmtime Engine (shared, thread-safe)
        self._engine: Any | None = None

    # ------------------------------------------------------------------
    # Resource limiter
    # ------------------------------------------------------------------

    def _make_resource_limiter(self) -> Any:
        """
        Build a wasmtime ``ResourceLimiter`` that caps linear memory growth
        and table element growth.

        Mirrors ``WasmResourceLimiter`` in ``src/tools/wasm/limits.rs``.

        The limiter is installed on the ``Store`` via ``store.limiter()``.
        wasmtime calls ``memory_growing`` / ``table_growing`` before each
        allocation; returning ``False`` traps the module.
        """
        max_bytes = self.max_bytes = self.max_memory_bytes
        max_table = self.max_table_elements

        try:
            import wasmtime

            class _Limiter(wasmtime.ResourceLimiter):
                def memory_growing(
                    self,
                    current: int,
                    desired: int,
                    maximum: int | None,
                ) -> bool:
                    if desired > max_bytes:
                        logger.debug(
                            "WASM memory growth denied: %d bytes requested, limit %d bytes",
                            desired,
                            max_bytes,
                        )
                        return False
                    return True

                def table_growing(
                    self,
                    current: int,
                    desired: int,
                    maximum: int | None,
                ) -> bool:
                    if desired > max_table:
                        logger.debug(
                            "WASM table growth denied: %d elements requested, limit %d",
                            desired,
                            max_table,
                        )
                        return False
                    return True

            return _Limiter()
        except (ImportError, AttributeError):
            # wasmtime not installed or old version without ResourceLimiter —
            # fall back gracefully (fuel still provides CPU protection)
            return None

    # ------------------------------------------------------------------
    # wasmtime engine
    # ------------------------------------------------------------------

    def _get_engine(self) -> Any:
        """Return (or lazily create) the wasmtime Engine."""
        if self._engine is not None:
            return self._engine
        try:
            import wasmtime
        except ImportError as exc:
            raise ImportError(
                "wasmtime-py is required for the WASM sandbox.\n"
                "Install it with: pip install \"titanclaw[wasm-sandbox]\""
            ) from exc

        cfg = wasmtime.Config()
        if self.fuel > 0:
            cfg.consume_fuel = True

        self._engine = wasmtime.Engine(cfg)
        return self._engine

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def _run_wasm_sync(
        self,
        wasm_path: str,
        params: dict[str, Any],
    ) -> ToolResult:
        """
        Execute a WASM module synchronously (called from a thread pool).

        Writes ``{"params": params}`` to the module's stdin and reads the
        JSON response from stdout.
        """
        start = time.monotonic()
        try:
            import wasmtime
        except ImportError as exc:
            raise ImportError(
                "wasmtime-py is required for the WASM sandbox.\n"
                "Install it with: pip install \"titanclaw[wasm-sandbox]\""
            ) from exc

        engine = self._get_engine()

        # Write stdin payload to a temp file (WASI preopens a temp dir)
        with tempfile.TemporaryDirectory(prefix="titanclaw-wasm-") as tmp:
            stdin_path = os.path.join(tmp, "stdin.json")
            stdout_path = os.path.join(tmp, "stdout.json")

            with open(stdin_path, "w") as f:
                json.dump({"params": params}, f)

            # Build WASI config
            wasi_cfg = wasmtime.WasiConfig()
            wasi_cfg.stdin_file = stdin_path
            wasi_cfg.stdout_file = stdout_path
            wasi_cfg.stderr_file = os.devnull  # suppress module stderr

            # Preopened directories: workspace (read-write) + temp (read-write)
            os.makedirs(self.workspace_dir, exist_ok=True)
            wasi_cfg.preopen_dir(self.workspace_dir, "/workspace")
            wasi_cfg.preopen_dir(tmp, "/tmp")

            store = wasmtime.Store(engine)
            store.set_wasi(wasi_cfg)
            if self.fuel > 0:
                store.add_fuel(self.fuel)

            # Attach resource limiter (memory + table caps)
            limiter = self._make_resource_limiter()
            if limiter is not None:
                store.limiter(limiter)

            # Compile and instantiate
            linker = wasmtime.Linker(engine)
            linker.define_wasi()

            module = wasmtime.Module.from_file(engine, wasm_path)
            instance = linker.instantiate(store, module)
            exports = instance.exports(store)

            # Call the WASI entry point
            start_fn = exports.get("_start")
            if start_fn is None:
                raise RuntimeError("WASM module has no '_start' export (not a WASI module?)")

            try:
                start_fn(store)
            except wasmtime.ExitTrap as e:
                # A WASI process exit is normal (code 0 = success)
                if e.code != 0:
                    duration = (time.monotonic() - start) * 1000
                    return ToolResult(
                        output=f"WASM module exited with code {e.code}",
                        duration_ms=duration,
                        error=True,
                    )
            except wasmtime.Trap as exc:
                duration = (time.monotonic() - start) * 1000
                msg = str(exc)
                if "fuel" in msg.lower() or "out of fuel" in msg.lower():
                    return ToolResult(
                        output=f"WASM module exceeded CPU fuel limit ({self.fuel:,} instructions)",
                        duration_ms=duration,
                        error=True,
                    )
                return ToolResult(
                    output=f"WASM trap: {exc}",
                    duration_ms=duration,
                    error=True,
                )

            # Read stdout
            if os.path.exists(stdout_path):
                with open(stdout_path) as f:
                    raw = f.read(_MAX_OUTPUT_BYTES)
            else:
                raw = ""

            duration = (time.monotonic() - start) * 1000

            if not raw.strip():
                return ToolResult(output="(no output)", duration_ms=duration)

            # Try to parse JSON response
            try:
                resp = json.loads(raw.strip().splitlines()[-1])
                output = str(resp.get("output", raw))[:_MAX_OUTPUT_BYTES]
                error = bool(resp.get("error", False))
            except (json.JSONDecodeError, AttributeError):
                output = raw[:_MAX_OUTPUT_BYTES]
                error = False

            return ToolResult(output=output, duration_ms=duration, error=error)

    async def run_wasm(
        self,
        wasm_path: str,
        params: dict[str, Any],
        timeout: float | None = None,
    ) -> ToolResult:
        """
        Execute a WASM module asynchronously.

        Runs ``_run_wasm_sync`` in the default executor (thread pool) to avoid
        blocking the event loop during compilation or long-running execution.
        """
        effective_timeout = timeout if timeout is not None else self.timeout
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self._run_wasm_sync, wasm_path, params),
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError:
            return ToolResult(
                output=f"WASM module timed out after {effective_timeout}s",
                duration_ms=effective_timeout * 1000,
                error=True,
            )
        return result

    # ------------------------------------------------------------------
    # Tool loading
    # ------------------------------------------------------------------

    def _load_tool_meta(self, wasm_path: Path) -> tuple[str, str, dict[str, Any]]:
        """
        Return (name, description, parameters_schema) for a .wasm file.

        Side-car files:
        * ``<name>.description.txt`` — one-line or multi-line description
        * ``<name>.schema.json``     — JSON Schema for parameters
        """
        name = wasm_path.stem
        stem = wasm_path.with_suffix("")

        desc_file = stem.with_suffix(".description.txt")
        description = (
            desc_file.read_text(encoding="utf-8").strip()
            if desc_file.is_file()
            else f"WASM sandbox tool: {name}"
        )

        schema_file = stem.with_suffix(".schema.json")
        schema: dict[str, Any] = (
            json.loads(schema_file.read_text(encoding="utf-8"))
            if schema_file.is_file()
            else _OPEN_SCHEMA
        )

        return name, description, schema

    def load_tools(self) -> list[ToolDefinition]:
        """
        Scan ``tools_dir`` for ``.wasm`` files and return a ``ToolDefinition``
        for each one.

        Missing or unreadable files are skipped with a warning.
        """
        tools_path = Path(self.tools_dir)
        if not tools_path.is_dir():
            logger.info("WASM tools directory %s does not exist — no tools loaded", self.tools_dir)
            return []

        definitions: list[ToolDefinition] = []
        for wasm_file in sorted(tools_path.glob("*.wasm")):
            try:
                tool = self._make_tool(wasm_file)
                definitions.append(tool)
                logger.info("Loaded WASM tool: %s (%s)", tool.name, wasm_file.name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load WASM tool %s: %s", wasm_file, exc)

        return definitions

    def load_tool(self, wasm_path: str) -> ToolDefinition:
        """Load a single WASM tool from an explicit path."""
        return self._make_tool(Path(wasm_path))

    def _make_tool(self, wasm_file: Path) -> ToolDefinition:
        name, description, schema = self._load_tool_meta(wasm_file)
        wasm_path_str = str(wasm_file)
        sandbox = self

        async def _execute(params: dict[str, Any], ctx: Any = None) -> ToolResult:
            return await sandbox.run_wasm(wasm_path_str, params)

        return ToolDefinition(
            name=name,
            description=description,
            parameters_schema=schema,
            execute=_execute,
            requires_approval=False,   # WASM is sandboxed; lower friction
            requires_sanitization=True,
        )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def check(self) -> bool:
        """Return ``True`` if wasmtime-py is importable."""
        try:
            import wasmtime  # noqa: F401
            return True
        except ImportError:
            return False
