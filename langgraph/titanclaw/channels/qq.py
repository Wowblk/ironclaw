"""QQ channel adapter — OneBot v11 HTTP callback.

Receives events via an OneBot v11-compatible HTTP POST callback and sends
replies through the OneBot HTTP API.

Compatible implementations
--------------------------
Any OneBot v11 implementation works:

* `NapCat <https://github.com/NapNeko/NapCatQQ>`_
* `LLOneBot <https://github.com/LLOneBot/LLOneBot>`_
* `OpenShamrock <https://github.com/whitechi73/OpenShamrock>`_
* `go-cqhttp <https://github.com/Mrs4s/go-cqhttp>`_

Setup checklist
---------------
1. Start your OneBot v11 implementation and configure its HTTP POST push
   to point at::

       https://<your-host>:<port>/qq/events

2. Set the OneBot HTTP API base URL in the environment::

       ONEBOT_API_URL=http://localhost:3000   # default

3. Optionally set an access token for request authentication::

       ONEBOT_ACCESS_TOKEN=...

4. Set the QQ bot's own user ID so self-messages can be ignored::

       ONEBOT_SELF_ID=12345678

Thread IDs
----------
* Private chat: ``qq-private-{user_id}``
* Group chat:   ``qq-group-{group_id}``

Message format
--------------
The adapter accepts both CQ-code strings and segment arrays.  Only ``text``
segments (and the plain string form) trigger agent invocation; images and
other media are currently ignored.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, Response
from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger(__name__)

_QQ_MSG_LIMIT = 4500  # conservative limit for a single QQ message

# Strip CQ codes from incoming messages (e.g. [CQ:at,qq=12345])
_CQ_CODE_RE = re.compile(r"\[CQ:[^\]]+\]")


def _strip_cq(text: str) -> str:
    return _CQ_CODE_RE.sub("", text).strip()


def _split_message(text: str, limit: int = _QQ_MSG_LIMIT) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return chunks


def _extract_text(message: Any) -> str:
    """
    Extract plain text from an OneBot v11 message.

    *message* can be a plain string (CQ code form) or a list of segment
    dicts.  Returns the concatenated text of all ``type=text`` segments,
    with CQ codes stripped.
    """
    if isinstance(message, str):
        return _strip_cq(message)
    if isinstance(message, list):
        parts = [
            seg.get("data", {}).get("text", "")
            for seg in message
            if seg.get("type") == "text"
        ]
        return "".join(parts).strip()
    return ""


class QQAdapter:
    """
    QQ channel adapter (OneBot v11).

    Receives OneBot v11 HTTP POST events and invokes the LangGraph agent
    for each private or group text message.

    Parameters
    ----------
    graph:
        Compiled LangGraph graph.
    tool_registry:
        Tool registry for ``available_tools``.
    onebot_api_url:
        Base URL of the OneBot HTTP API (e.g. ``http://localhost:3000``).
    access_token:
        Optional bearer token.  When set, incoming requests must carry a
        matching ``Authorization: Bearer <token>`` header, and outbound API
        calls include it too.
    self_id:
        The bot's own QQ number.  Messages from this ID are ignored.
    system_prompt:
        Optional system prompt injected into every agent invocation.
    """

    def __init__(
        self,
        graph: Any,
        tool_registry: Any,
        onebot_api_url: str = "http://localhost:3000",
        access_token: str | None = None,
        self_id: int | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._graph = graph
        self._tool_registry = tool_registry
        self._onebot_api_url = onebot_api_url.rstrip("/")
        self._access_token = access_token
        self._self_id = self_id
        self._system_prompt = system_prompt

    # ------------------------------------------------------------------
    # Agent helpers
    # ------------------------------------------------------------------

    def _tool_defs(self) -> list[Any]:
        from titanclaw.state import ToolDefinition

        return [
            ToolDefinition(
                name=t.name,
                description=t.description,
                parameters=t.parameters_schema,
            )
            for t in self._tool_registry.list_tools()
        ]

    async def _invoke(self, text: str, thread_id: str) -> str:
        state_input: dict[str, Any] = {
            "messages": [HumanMessage(content=text)],
            "available_tools": self._tool_defs(),
        }
        if self._system_prompt:
            state_input["system_prompt"] = self._system_prompt
        result = await self._graph.ainvoke(
            state_input,
            config={"configurable": {"thread_id": thread_id}},
        )
        messages = result.get("messages", [])
        last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if last_ai is None:
            return "(no response)"
        content = last_ai.content
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in content
            )
        return content

    # ------------------------------------------------------------------
    # OneBot send helpers
    # ------------------------------------------------------------------

    def _api_headers(self) -> dict[str, str]:
        if self._access_token:
            return {"Authorization": f"Bearer {self._access_token}"}
        return {}

    async def _send_private(self, user_id: int, text: str) -> None:
        async with httpx.AsyncClient() as client:
            for chunk in _split_message(text):
                await client.post(
                    f"{self._onebot_api_url}/send_private_msg",
                    headers=self._api_headers(),
                    json={"user_id": user_id, "message": chunk},
                    timeout=15,
                )

    async def _send_group(self, group_id: int, text: str) -> None:
        async with httpx.AsyncClient() as client:
            for chunk in _split_message(text):
                await client.post(
                    f"{self._onebot_api_url}/send_group_msg",
                    headers=self._api_headers(),
                    json={"group_id": group_id, "message": chunk},
                    timeout=15,
                )

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    async def _handle_event(self, event: dict) -> None:
        post_type = event.get("post_type")
        if post_type != "message":
            return

        # Ignore self-messages
        sender_id = event.get("user_id")
        if self._self_id and sender_id == self._self_id:
            return

        msg_type = event.get("message_type", "")
        raw_message = event.get("message", "")
        text = _extract_text(raw_message)
        if not text:
            return

        if msg_type == "private":
            thread_id = f"qq-private-{sender_id}"
            logger.info(
                "QQ private message from user=%s thread=%s: %.80s",
                sender_id, thread_id, text,
            )
            try:
                response = await self._invoke(text, thread_id)
            except Exception as exc:
                logger.exception("Agent error for thread %s", thread_id)
                await self._send_private(sender_id, f"出错了：{exc}")
                return
            await self._send_private(sender_id, response)

        elif msg_type == "group":
            group_id = event.get("group_id")
            thread_id = f"qq-group-{group_id}"
            logger.info(
                "QQ group message from user=%s group=%s thread=%s: %.80s",
                sender_id, group_id, thread_id, text,
            )
            try:
                response = await self._invoke(text, thread_id)
            except Exception as exc:
                logger.exception("Agent error for thread %s", thread_id)
                await self._send_group(group_id, f"出错了：{exc}")
                return
            await self._send_group(group_id, response)

    # ------------------------------------------------------------------
    # ASGI app
    # ------------------------------------------------------------------

    @property
    def asgi_app(self) -> FastAPI:
        """Return a FastAPI app exposing ``POST /qq/events``."""
        api = FastAPI(title="TitanClaw QQ Gateway", docs_url=None, redoc_url=None)

        @api.post("/qq/events")
        async def qq_events(
            request: Request,
            authorization: str | None = Header(default=None),
        ) -> Response:
            # Verify access token if configured
            if self._access_token:
                expected = f"Bearer {self._access_token}"
                if authorization != expected:
                    raise HTTPException(status_code=401, detail="unauthorized")

            try:
                event = await request.json()
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc

            try:
                await self._handle_event(event)
            except Exception:
                logger.exception("QQ event handling error")
                # Always return 200 to prevent OneBot from retrying
            return Response(content="{}", media_type="application/json")

        @api.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        return api
