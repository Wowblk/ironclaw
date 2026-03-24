"""Orchestrator — internal HTTP API for sandbox container ↔ host communication.

Mirrors ``src/orchestrator/`` in the Rust ironclaw codebase.

Components
----------
``OrchestratorServer``
    Minimal asyncio HTTP server exposing:

    * ``POST /v1/chat``    — proxy LLM calls from containers (auth required)
    * ``POST /v1/events``  — ingest container progress events (auth required)
    * ``GET  /v1/health``  — liveness check (no auth)

``TokenStore``
    Manages per-job bearer tokens.  Each ``DockerSandbox`` job receives a
    unique token via ``ORCHESTRATOR_TOKEN`` env var; the container uses it
    to authenticate orchestrator requests.

Quick start::

    from titanclaw.orchestrator import OrchestratorServer

    server = OrchestratorServer(llm=my_langchain_llm)
    port = await server.start()

    # When launching a Docker container job:
    token = await server.token_store.issue(job_id="job-42")

    # Inject into container:
    #   -e ORCHESTRATOR_URL=http://host-gateway:{port}
    #   -e ORCHESTRATOR_TOKEN={token}

    await server.stop()
"""

from titanclaw.orchestrator.auth import JobToken, TokenStore
from titanclaw.orchestrator.server import OrchestratorServer

__all__ = ["JobToken", "OrchestratorServer", "TokenStore"]
