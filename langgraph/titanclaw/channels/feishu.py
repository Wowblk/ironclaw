"""Feishu (Lark / 飞书) channel adapter — Events API v2.

Receives ``im.message.receive_v1`` events via HTTP POST webhook and replies
using the Feishu Messaging API.

Setup checklist
---------------
1. Create a Feishu app at https://open.feishu.cn/app
2. Under **事件订阅 (Event Subscriptions)**, set the Request URL to::

       https://<your-host>:<port>/feishu/events

3. Subscribe to the ``im.message.receive_v1`` event.
4. Add API permissions: ``im:message`` and ``im:message:send_as_bot``.
5. Set environment variables::

       FEISHU_APP_ID=cli_...
       FEISHU_APP_SECRET=...
       FEISHU_VERIFICATION_TOKEN=...

Message encryption (optional)
------------------------------
If you enable **message encryption** in the Feishu app settings, also set::

    FEISHU_ENCRYPT_KEY=...

The encrypt key is the plaintext key shown in the console — the adapter
derives the AES-256 key internally.

Requires ``cryptography`` for encrypted mode::

    pip install "titanclaw[feishu]"

Thread IDs
----------
* P2P (direct message): ``feishu-p2p-{sender_open_id}``
* Group chat:           ``feishu-group-{chat_id}``

Dependencies (encrypted mode only)
-------------------------------------
``cryptography>=42``  — install with ``pip install "titanclaw[feishu]"``
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger(__name__)

_FEISHU_API = "https://open.feishu.cn/open-apis"
_FEISHU_MSG_LIMIT = 4000  # chars per message


def _split_message(text: str, limit: int = _FEISHU_MSG_LIMIT) -> list[str]:
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


def _decrypt_feishu(encrypt_key: str, ciphertext: str) -> dict:
    """Decrypt a Feishu AES-256-CBC encrypted event body."""
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.backends import default_backend
    except ImportError as exc:
        raise ImportError(
            "cryptography is required for Feishu message encryption.\n"
            "Install it with: pip install \"titanclaw[feishu]\""
        ) from exc

    import base64

    key = hashlib.sha256(encrypt_key.encode("utf-8")).digest()
    data = base64.b64decode(ciphertext)
    iv = data[:16]
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    raw = decryptor.update(data[16:]) + decryptor.finalize()
    # Strip PKCS#7 padding
    pad_len = raw[-1]
    raw = raw[:-pad_len]
    return json.loads(raw.decode("utf-8"))


class FeishuAdapter:
    """
    Feishu (Lark) channel adapter.

    Receives events via the Feishu Events API v2 and invokes the LangGraph
    agent for each incoming message, replying via the Messaging API.

    Parameters
    ----------
    graph:
        Compiled LangGraph graph.
    tool_registry:
        Tool registry for ``available_tools``.
    app_id:
        Feishu app ID (``cli_...``).
    app_secret:
        Feishu app secret.
    verification_token:
        Verification token from the Event Subscriptions console page.
    encrypt_key:
        Optional AES encrypt key. Set only when message encryption is enabled.
    system_prompt:
        Optional system prompt injected into every agent invocation.
    """

    def __init__(
        self,
        graph: Any,
        tool_registry: Any,
        app_id: str,
        app_secret: str,
        verification_token: str,
        encrypt_key: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._graph = graph
        self._tool_registry = tool_registry
        self._app_id = app_id
        self._app_secret = app_secret
        self._verification_token = verification_token
        self._encrypt_key = encrypt_key
        self._system_prompt = system_prompt

        # Tenant access token cache: (token, expire_at)
        self._token_cache: tuple[str, float] | None = None

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    async def _get_tenant_token(self) -> str:
        """Return a valid tenant_access_token, refreshing if needed."""
        now = time.monotonic()
        if self._token_cache and self._token_cache[1] > now + 60:
            return self._token_cache[0]

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{_FEISHU_API}/auth/v3/tenant_access_token/internal",
                json={"app_id": self._app_id, "app_secret": self._app_secret},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

        token = data["tenant_access_token"]
        expire_in = data.get("expire", 7200)
        self._token_cache = (token, now + expire_in)
        return token

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
    # Feishu Messaging API
    # ------------------------------------------------------------------

    async def _send_message(self, receive_id: str, receive_id_type: str, text: str) -> None:
        token = await self._get_tenant_token()
        async with httpx.AsyncClient() as client:
            for chunk in _split_message(text):
                await client.post(
                    f"{_FEISHU_API}/im/v1/messages",
                    params={"receive_id_type": receive_id_type},
                    headers={"Authorization": f"Bearer {token}"},
                    json={
                        "receive_id": receive_id,
                        "msg_type": "text",
                        "content": json.dumps({"text": chunk}),
                    },
                    timeout=15,
                )

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    async def _parse_event(self, raw: dict) -> dict:
        """Decrypt payload if encrypted, then return the event dict."""
        if "encrypt" in raw:
            if not self._encrypt_key:
                raise ValueError(
                    "Received an encrypted Feishu event but FEISHU_ENCRYPT_KEY is not set."
                )
            return _decrypt_feishu(self._encrypt_key, raw["encrypt"])
        return raw

    async def _handle_event(self, body: dict) -> dict:
        """Process a single parsed Feishu event. Returns the HTTP response body."""
        event_type = body.get("type") or (body.get("header") or {}).get("event_type", "")

        # URL verification challenge
        if event_type == "url_verification":
            token = body.get("token", "")
            if token != self._verification_token:
                raise HTTPException(status_code=403, detail="token mismatch")
            return {"challenge": body.get("challenge", "")}

        # Token verification for regular events
        header = body.get("header", {})
        if header.get("token", "") != self._verification_token:
            raise HTTPException(status_code=403, detail="token mismatch")

        if event_type != "im.message.receive_v1":
            return {}

        event = body.get("event", {})
        message = event.get("message", {})
        sender = event.get("sender", {})
        sender_id = sender.get("sender_id", {})

        msg_type = message.get("message_type", "")
        if msg_type != "text":
            return {}  # ignore non-text messages

        # Parse text content
        try:
            content_obj = json.loads(message.get("content", "{}"))
            text = content_obj.get("text", "").strip()
        except json.JSONDecodeError:
            text = ""

        # Remove @bot mention if present
        text = text.replace("\ue058", "").strip()
        if not text:
            return {}

        chat_type = message.get("chat_type", "p2p")
        chat_id = message.get("chat_id", "")
        open_id = sender_id.get("open_id", "")

        if chat_type == "p2p":
            thread_id = f"feishu-p2p-{open_id}"
            receive_id = open_id
            receive_id_type = "open_id"
        else:
            thread_id = f"feishu-group-{chat_id}"
            receive_id = chat_id
            receive_id_type = "chat_id"

        logger.info(
            "Feishu message chat_type=%s thread=%s: %.80s", chat_type, thread_id, text
        )

        try:
            response = await self._invoke(text, thread_id)
        except Exception as exc:
            logger.exception("Agent error for thread %s", thread_id)
            await self._send_message(receive_id, receive_id_type, f"出错了：{exc}")
            return {}

        await self._send_message(receive_id, receive_id_type, response)
        return {}

    # ------------------------------------------------------------------
    # ASGI app
    # ------------------------------------------------------------------

    @property
    def asgi_app(self) -> FastAPI:
        """Return a FastAPI app exposing ``POST /feishu/events``."""
        api = FastAPI(title="TitanClaw Feishu Gateway", docs_url=None, redoc_url=None)

        @api.post("/feishu/events")
        async def feishu_events(request: Request) -> Response:
            raw = await request.json()
            try:
                body = await self._parse_event(raw)
                result = await self._handle_event(body)
            except HTTPException:
                raise
            except Exception as exc:
                logger.exception("Feishu event handling error")
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            return Response(
                content=json.dumps(result),
                media_type="application/json",
            )

        @api.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        return api
