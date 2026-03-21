"""Tool sandbox layer.

Two backends are available:

Docker sandbox (``titanclaw.tools.sandbox.docker_sandbox``)
    Runs shell commands inside a Docker container with hard resource limits,
    dropped capabilities, and an optional network block.
    Install dependency: ``pip install "titanclaw[docker-sandbox]"`` (no extra
    Python package needed — uses the ``docker`` CLI).

WASM sandbox (``titanclaw.tools.sandbox.wasm_sandbox``)
    Loads ``.wasm`` WASI modules from a directory and registers each one as a
    first-class tool.  Modules run with restricted filesystem access, no
    network, and a configurable fuel cap.
    Install dependency: ``pip install "titanclaw[wasm-sandbox]"``

Usage example::

    from titanclaw.tools.sandbox.docker_sandbox import DockerSandbox
    from titanclaw.tools.sandbox.wasm_sandbox import WasmSandbox

    docker = DockerSandbox(image="python:3.12-slim", memory_mb=512)
    registry.register(docker.create_shell_tool())   # replaces bare shell tool

    wasm = WasmSandbox(tools_dir="~/.titanclaw/wasm-tools")
    for tool in wasm.load_tools():
        registry.register(tool)
"""

from titanclaw.tools.sandbox.docker_sandbox import DockerSandbox
from titanclaw.tools.sandbox.wasm_sandbox import WasmSandbox

__all__ = ["DockerSandbox", "WasmSandbox"]
