"""Orchestrator HTTP API server.

Provides an internal HTTP API that sandbox containers can call to:

* Invoke the LLM without holding an API key themselves (``POST /v1/chat``)
* Report progress events (``POST /v1/events``)
* Check liveness (``GET /v1/health``)

Security
--------
Every endpoint (except ``/v1/health``) requires a ``Authorization: Bearer
<token>`` header.  Tokens are issued by ``TokenStore`` (one per job) and
injected into the container via the ``ORCHESTRATOR_TOKEN`` environment
variable.  The container never has direct LLM API access.

This mirrors ``src/orchestrator/api.rs`` + ``src/orchestrator/auth.rs`` in
the Rust ironclaw codebase.

Architecture
------------
The server is a minimal asyncio HTTP/1.1 server — no extra dependencies.  It
uses the same ``asyncio.start_server`` / ``StreamReader`` approach as
``network_proxy.py``.

Endpoints
---------
``GET /v1/health``
    Returns ``{"status": "ok"}``.  No authentication required.

``POST /v1/chat``
    Proxy a chat completion to the host LLM.

    Request body::

        {
          "messages": [{"role": "user", "content": "..."}],
          "system":   "optional system prompt (string)",
          "max_tokens": 1024
        }

    Response body::

        {
          "content":     "...",
          "stop_reason": "end_turn",
          "usage":       {"input_tokens": 10, "output_tokens": 50}
        }

``POST /v1/events``
    Ingest a progress event from the container.

    Request body::

        {
          "event_type": "progress" | "status" | "error" | "output",
          "message":    "human-readable description",
          "data":       {}
        }

    Response body::  ``{"ok": true}``

Usage::

    server = OrchestratorServer(llm=langchain_llm)
    port = await server.start()

    token = await server.token_store.issue("job-42", session_id="sess-1")
    # In DockerSandbox.run_command():
    #   -e ORCHESTRATOR_URL=http://host-gateway:{port}
    #   -e ORCHESTRATOR_TOKEN={token}

    await server.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from titanclaw.orchestrator.auth import TokenStore

logger = logging.getLogger(__name__)

_READ_TIMEOUT = 30.0
_MAX_BODY_BYTES = 1 << 20   # 1 MiB


class OrchestratorServer:
    """
    Minimal async HTTP/1.1 orchestrator server.

    Parameters
    ----------
    llm:
        A LangChain chat model instance (``ChatAnthropic``, ``ChatOpenAI``,
        etc.).  Used to serve ``POST /v1/chat`` requests.
    bind_host:
        Interface to listen on.  Defaults to ``"127.0.0.1"`` (localhost only);
        containers reach it via the ``host-gateway`` Docker DNS alias.
    on_event:
        Optional async callback ``(job_id, event_type, message, data) → None``
        called whenever a container posts to ``/v1/events``.
    """

    def __init__(
        self,
        llm: Any,
        bind_host: str = "127.0.0.1",
        on_event: Any = None,
    ) -> None:
        self.llm = llm
        self.bind_host = bind_host
        self.on_event = on_event
        self.token_store = TokenStore()
        self._server: asyncio.AbstractServer | None = None
        self._port: int | None = None

    @property
    def port(self) -> int | None:
        return self._port

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> int:
        """Start listening; return the bound port."""
        self._server = await asyncio.start_server(
            self._handle_client,
            host=self.bind_host,
            port=0,
        )
        self._port = self._server.sockets[0].getsockname()[1]
        logger.info("OrchestratorServer listening on %s:%d", self.bind_host, self._port)
        return self._port

    async def stop(self) -> None:
        """Shut down the server and revoke all tokens."""
        await self.token_store.revoke_all()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self._port = None
        logger.debug("OrchestratorServer stopped")

    # ------------------------------------------------------------------
    # Connection dispatch
    # ------------------------------------------------------------------

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            await self._dispatch(reader, writer)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Orchestrator client error: %s", exc)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:  # noqa: BLE001
                pass

    async def _dispatch(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        # Request line
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=_READ_TIMEOUT)
        except asyncio.TimeoutError:
            return
        if not request_line:
            return

        parts = request_line.decode(errors="replace").split()
        if len(parts) < 2:
            return
        method, path = parts[0].upper(), parts[1].split("?")[0]

        # Headers
        headers = await self._read_headers(reader)
        content_length = int(headers.get("content-length", "0") or "0")

        # Body
        body_bytes = b""
        if content_length > 0:
            try:
                body_bytes = await asyncio.wait_for(
                    reader.readexactly(min(content_length, _MAX_BODY_BYTES)),
                    timeout=_READ_TIMEOUT,
                )
            except (asyncio.TimeoutError, asyncio.IncompleteReadError):
                pass

        body = {}
        if body_bytes:
            try:
                body = json.loads(body_bytes)
            except json.JSONDecodeError:
                pass

        # Route
        if method == "GET" and path == "/v1/health":
            await self._respond(writer, 200, {"status": "ok", "ts": time.time()})
            return

        # Auth check for all other endpoints
        job_token = await self._authenticate(headers)
        if job_token is None:
            await self._respond(writer, 401, {"error": "unauthorized"})
            return

        if method == "POST" and path == "/v1/chat":
            await self._handle_chat(writer, job_token.job_id, body)
        elif method == "POST" and path == "/v1/events":
            await self._handle_event(writer, job_token.job_id, body)
        else:
            await self._respond(writer, 404, {"error": "not found"})

    # ------------------------------------------------------------------
    # Endpoint handlers
    # ------------------------------------------------------------------

    async def _handle_chat(
        self,
        writer: asyncio.StreamWriter,
        job_id: str,
        body: dict[str, Any],
    ) -> None:
        """Proxy a chat request to the host LLM."""
        messages_raw = body.get("messages", [])
        system = body.get("system")
        max_tokens = body.get("max_tokens")

        if not messages_raw:
            await self._respond(writer, 400, {"error": "messages is required"})
            return

        try:
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

            lc_messages = []
            if system:
                lc_messages.append(SystemMessage(content=system))
            for m in messages_raw:
                role = m.get("role", "user")
                content = m.get("content", "")
                if role == "user":
                    lc_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    lc_messages.append(AIMessage(content=content))
                elif role == "system":
                    lc_messages.append(SystemMessage(content=content))

            # Bind max_tokens if provided
            llm = self.llm
            if max_tokens:
                try:
                    llm = llm.bind(max_tokens=max_tokens)
                except Exception:  # noqa: BLE001
                    pass

            response: AIMessage = await llm.ainvoke(lc_messages)

            content_text = (
                response.content
                if isinstance(response.content, str)
                else json.dumps(response.content)
            )
            usage = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = {
                    "input_tokens": response.usage_metadata.get("input_tokens", 0),
                    "output_tokens": response.usage_metadata.get("output_tokens", 0),
                }

            logger.debug(
                "Orchestrator /v1/chat (job=%s): %d in / %d out tokens",
                job_id,
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
            )

            await self._respond(writer, 200, {
                "content": content_text,
                "stop_reason": getattr(response, "response_metadata", {}).get(
                    "stop_reason", "end_turn"
                ),
                "usage": usage,
            })

        except Exception as exc:  # noqa: BLE001
            logger.warning("Orchestrator /v1/chat error (job=%s): %s", job_id, exc)
            await self._respond(writer, 500, {"error": str(exc)})

    async def _handle_event(
        self,
        writer: asyncio.StreamWriter,
        job_id: str,
        body: dict[str, Any],
    ) -> None:
        """Ingest a progress event from the container."""
        event_type = body.get("event_type", "status")
        message = body.get("message", "")
        data = body.get("data", {})

        logger.info(
            "Container event [job=%s type=%s]: %s",
            job_id,
            event_type,
            message[:200],
        )

        if self.on_event is not None:
            try:
                await self.on_event(job_id, event_type, message, data)
            except Exception as exc:  # noqa: BLE001
                logger.warning("on_event callback error: %s", exc)

        await self._respond(writer, 200, {"ok": True})

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _read_headers(self, reader: asyncio.StreamReader) -> dict[str, str]:
        headers: dict[str, str] = {}
        while True:
            try:
                line = await asyncio.wait_for(reader.readline(), timeout=_READ_TIMEOUT)
            except asyncio.TimeoutError:
                break
            if not line or line in (b"\r\n", b"\n"):
                break
            decoded = line.decode(errors="replace").strip()
            if ":" in decoded:
                k, _, v = decoded.partition(":")
                headers[k.strip().lower()] = v.strip()
        return headers

    async def _authenticate(self, headers: dict[str, str]) -> Any:
        """
        Extract and verify the bearer token from the Authorization header.
        Returns the ``JobToken`` or ``None``.
        """
        auth = headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            return None
        token = auth[7:].strip()
        return await self.token_store.verify(token)

    async def _respond(
        self,
        writer: asyncio.StreamWriter,
        status: int,
        body: dict[str, Any],
    ) -> None:
        payload = json.dumps(body).encode()
        status_text = {200: "OK", 400: "Bad Request", 401: "Unauthorized",
                       404: "Not Found", 500: "Internal Server Error"}.get(status, "Unknown")
        headers = (
            f"HTTP/1.1 {status} {status_text}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        )
        writer.write(headers.encode() + payload)
        await writer.drain()
