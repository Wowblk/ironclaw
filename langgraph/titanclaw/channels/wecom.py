"""WeCom (企业微信) channel adapter — Callback API.

Receives messages via the WeCom Callback URL (企业内部应用 → 接收消息) and
replies using the WeCom Message API.

Setup checklist
---------------
1. Log in at https://work.weixin.qq.com and open your internal app.
2. Under **企业可信IP / 接收消息**, configure the Callback URL::

       https://<your-host>:<port>/wecom/events

3. Fill in **Token** and **EncodingAESKey** on that page; copy them to env vars.
4. Set environment variables::

       WECOM_CORP_ID=ww...
       WECOM_CORP_SECRET=...
       WECOM_AGENT_ID=1000002          # app agent ID
       WECOM_TOKEN=...                 # callback token
       WECOM_ENCODING_AES_KEY=...      # 43-char base64 key

Message encryption
------------------
WeCom mandates AES-256-CBC encryption for all callback messages.
Requires the ``cryptography`` package::

    pip install "titanclaw[wecom]"

Thread IDs
----------
* Individual (ToUserName == FromUserName): ``wecom-{from_user}``
* Shared callback (group scenario not applicable here)

Supported message types
-----------------------
Only ``text`` messages trigger agent invocation; other types are acknowledged
but silently ignored.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import struct
import time
import xml.etree.ElementTree as ET
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, Response
from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger(__name__)

_WECOM_API = "https://qyapi.weixin.qq.com/cgi-bin"
_WECOM_MSG_LIMIT = 2048  # WeCom text message character limit


def _split_message(text: str, limit: int = _WECOM_MSG_LIMIT) -> list[str]:
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


# ---------------------------------------------------------------------------
# WeCom AES crypto (企业微信消息加解密)
# ---------------------------------------------------------------------------

def _wecom_aes_key(encoding_aes_key: str) -> bytes:
    """Derive the 32-byte AES key from the 43-char base64 EncodingAESKey."""
    return base64.b64decode(encoding_aes_key + "=")


def _wecom_decrypt(aes_key: bytes, ciphertext_b64: str) -> tuple[str, str]:
    """
    Decrypt a WeCom AES-256-CBC encrypted message.

    Returns ``(plaintext_xml, corp_id)``.
    """
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.backends import default_backend
    except ImportError as exc:
        raise ImportError(
            "cryptography is required for WeCom message decryption.\n"
            "Install it with: pip install \"titanclaw[wecom]\""
        ) from exc

    data = base64.b64decode(ciphertext_b64)
    iv = aes_key[:16]
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    raw = decryptor.update(data) + decryptor.finalize()

    # Strip PKCS#7 padding
    pad_len = raw[-1]
    raw = raw[:-pad_len]

    # WeCom format: 16 random bytes | 4-byte big-endian msg_len | msg | corp_id
    msg_len = struct.unpack(">I", raw[16:20])[0]
    msg = raw[20 : 20 + msg_len].decode("utf-8")
    corp_id = raw[20 + msg_len :].decode("utf-8")
    return msg, corp_id


def _wecom_encrypt(aes_key: bytes, corp_id: str, plaintext: str) -> str:
    """
    Encrypt a WeCom reply message.

    Returns base64-encoded ciphertext.
    """
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.backends import default_backend
    except ImportError as exc:
        raise ImportError(
            "cryptography is required for WeCom message encryption.\n"
            "Install it with: pip install \"titanclaw[wecom]\""
        ) from exc

    import os

    random_bytes = os.urandom(16)
    msg_bytes = plaintext.encode("utf-8")
    corp_id_bytes = corp_id.encode("utf-8")
    msg_len_bytes = struct.pack(">I", len(msg_bytes))
    content = random_bytes + msg_len_bytes + msg_bytes + corp_id_bytes

    # PKCS#7 padding to AES block size (32 bytes for WeCom)
    block_size = 32
    pad_len = block_size - len(content) % block_size
    content += bytes([pad_len] * pad_len)

    iv = aes_key[:16]
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(content) + encryptor.finalize()
    return base64.b64encode(ciphertext).decode("utf-8")


def _wecom_signature(token: str, timestamp: str, nonce: str, *extras: str) -> str:
    """Compute WeCom SHA1 signature over sorted fields."""
    parts = sorted([token, timestamp, nonce, *extras])
    return hashlib.sha1("".join(parts).encode("utf-8")).hexdigest()


class WeComAdapter:
    """
    WeCom (企业微信) channel adapter.

    Handles URL verification (GET) and incoming message callbacks (POST),
    invokes the LangGraph agent, and sends replies via the Message API.

    Parameters
    ----------
    graph:
        Compiled LangGraph graph.
    tool_registry:
        Tool registry for ``available_tools``.
    corp_id:
        WeCom corp ID (``ww...``).
    corp_secret:
        WeCom app secret.
    agent_id:
        WeCom app agent ID (integer).
    token:
        Callback token set in the WeCom app console.
    encoding_aes_key:
        43-character base64 EncodingAESKey from the console.
    system_prompt:
        Optional system prompt injected into every agent invocation.
    """

    def __init__(
        self,
        graph: Any,
        tool_registry: Any,
        corp_id: str,
        corp_secret: str,
        agent_id: int,
        token: str,
        encoding_aes_key: str,
        system_prompt: str | None = None,
    ) -> None:
        self._graph = graph
        self._tool_registry = tool_registry
        self._corp_id = corp_id
        self._corp_secret = corp_secret
        self._agent_id = agent_id
        self._token = token
        self._aes_key = _wecom_aes_key(encoding_aes_key)
        self._system_prompt = system_prompt

        # Access token cache: (token_str, expire_at)
        self._access_token_cache: tuple[str, float] | None = None

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    async def _get_access_token(self) -> str:
        now = time.monotonic()
        if self._access_token_cache and self._access_token_cache[1] > now + 60:
            return self._access_token_cache[0]

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{_WECOM_API}/gettoken",
                params={"corpid": self._corp_id, "corpsecret": self._corp_secret},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

        if data.get("errcode", 0) != 0:
            raise RuntimeError(f"WeCom gettoken error: {data}")

        token_str = data["access_token"]
        expire_in = data.get("expires_in", 7200)
        self._access_token_cache = (token_str, now + expire_in)
        return token_str

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
    # WeCom Message API
    # ------------------------------------------------------------------

    async def _send_message(self, to_user: str, text: str) -> None:
        token = await self._get_access_token()
        async with httpx.AsyncClient() as client:
            for chunk in _split_message(text):
                resp = await client.post(
                    f"{_WECOM_API}/message/send",
                    params={"access_token": token},
                    json={
                        "touser": to_user,
                        "msgtype": "text",
                        "agentid": self._agent_id,
                        "text": {"content": chunk},
                    },
                    timeout=15,
                )
                data = resp.json()
                if data.get("errcode", 0) != 0:
                    logger.error("WeCom send_message error: %s", data)

    # ------------------------------------------------------------------
    # Signature verification
    # ------------------------------------------------------------------

    def _verify_signature(
        self, msg_signature: str, timestamp: str, nonce: str, *extras: str
    ) -> bool:
        expected = _wecom_signature(self._token, timestamp, nonce, *extras)
        return hmac.compare_digest(expected, msg_signature)

    # ------------------------------------------------------------------
    # ASGI app
    # ------------------------------------------------------------------

    @property
    def asgi_app(self) -> FastAPI:
        """Return a FastAPI app exposing GET/POST ``/wecom/events``."""
        api = FastAPI(title="TitanClaw WeCom Gateway", docs_url=None, redoc_url=None)

        @api.get("/wecom/events")
        async def wecom_verify(
            msg_signature: str = Query(...),
            timestamp: str = Query(...),
            nonce: str = Query(...),
            echostr: str = Query(...),
        ) -> Response:
            """URL verification handshake."""
            if not self._verify_signature(msg_signature, timestamp, nonce, echostr):
                raise HTTPException(status_code=403, detail="signature mismatch")
            # Decrypt the echostr and return it as plain text
            try:
                plaintext, _ = _wecom_decrypt(self._aes_key, echostr)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return Response(content=plaintext, media_type="text/plain")

        @api.post("/wecom/events")
        async def wecom_callback(
            request: Request,
            msg_signature: str = Query(...),
            timestamp: str = Query(...),
            nonce: str = Query(...),
        ) -> Response:
            """Incoming message callback."""
            body_bytes = await request.body()
            body_text = body_bytes.decode("utf-8")

            # Parse the outer XML to get the Encrypt element
            try:
                root = ET.fromstring(body_text)
            except ET.ParseError as exc:
                raise HTTPException(status_code=400, detail=f"XML parse error: {exc}") from exc

            encrypt_elem = root.find("Encrypt")
            if encrypt_elem is None or not encrypt_elem.text:
                raise HTTPException(status_code=400, detail="missing Encrypt element")

            encrypt_content = encrypt_elem.text

            # Verify signature over the encrypted content
            if not self._verify_signature(
                msg_signature, timestamp, nonce, encrypt_content
            ):
                raise HTTPException(status_code=403, detail="signature mismatch")

            # Decrypt
            try:
                msg_xml, corp_id = _wecom_decrypt(self._aes_key, encrypt_content)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            if corp_id != self._corp_id:
                raise HTTPException(status_code=403, detail="corp_id mismatch")

            # Parse decrypted message XML
            try:
                msg_root = ET.fromstring(msg_xml)
            except ET.ParseError as exc:
                raise HTTPException(status_code=400, detail=f"inner XML parse error: {exc}") from exc

            def _text(tag: str) -> str:
                el = msg_root.find(tag)
                return el.text or "" if el is not None else ""

            msg_type = _text("MsgType").lower()
            from_user = _text("FromUserName")

            if msg_type != "text":
                # Acknowledge non-text messages without responding
                return Response(content="success", media_type="text/plain")

            content = _text("Content").strip()
            if not content:
                return Response(content="success", media_type="text/plain")

            thread_id = f"wecom-{from_user}"
            logger.info(
                "WeCom message from user=%s thread=%s: %.80s", from_user, thread_id, content
            )

            try:
                response = await self._invoke(content, thread_id)
            except Exception as exc:
                logger.exception("Agent error for thread %s", thread_id)
                await self._send_message(from_user, f"出错了：{exc}")
                return Response(content="success", media_type="text/plain")

            await self._send_message(from_user, response)
            return Response(content="success", media_type="text/plain")

        @api.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        return api
