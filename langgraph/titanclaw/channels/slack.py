"""Slack channel adapter — Events API via slack-bolt (async).

Mirrors the WASM Slack channel in channels-src/slack/ but runs natively in
Python as a standalone ASGI server.

Handles two event types:
* ``app_mention``  — @mention in any channel the bot is invited to
* ``message`` (DM) — direct messages (channel_type == "im")

Usage::

    adapter = SlackAdapter(
        graph=graph,
        tool_registry=tool_registry,
        bot_token="xoxb-...",
        signing_secret="abc123...",
    )
    # Run as ASGI app on port 3000
    import uvicorn
    cfg = uvicorn.Config(adapter.asgi_app, host="0.0.0.0", port=3000)
    await uvicorn.Server(cfg).serve()

Slack setup checklist
---------------------
1. Create a Slack app at https://api.slack.com/apps
2. Enable "Event Subscriptions" and set Request URL to
   ``https://<your-host>:3000/slack/events``
3. Subscribe to bot events: ``app_mention``, ``message.im``
4. Add OAuth scopes: ``chat:write``, ``app_mentions:read``, ``im:history``
5. Install the app to your workspace
6. Copy the Bot Token (xoxb-...) and Signing Secret to env vars

Thread IDs
----------
* DMs:             ``slack-{channel_id}``
* Channel threads: ``slack-{channel_id}-{thread_ts}``

Message splitting
-----------------
Slack's ``chat.postMessage`` ``text`` field supports up to ~3000 characters
without truncation.  Longer responses are split on newline boundaries.

Dependencies
------------
Requires ``slack-bolt>=1.21``.  Install with::

    pip install "titanclaw[slack]"
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger(__name__)

_SLACK_LIMIT = 3000
_BOT_MENTION_RE = re.compile(r"<@[A-Z0-9]+>\s*")


def _strip_mention(text: str) -> str:
    """Remove ``<@BOTID>`` mention prefix from message text."""
    return _BOT_MENTION_RE.sub("", text).strip()


def _split_message(text: str, limit: int = _SLACK_LIMIT) -> list[str]:
    """Split *text* into chunks of at most *limit* characters."""
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


class SlackAdapter:
    """
    Slack channel adapter.

    Receives events via the Slack Events API and invokes the LangGraph agent
    for each app_mention or DM, posting the response back to the same channel
    (and thread, if the message was in a thread).

    Parameters
    ----------
    graph:
        Compiled LangGraph graph (from ``build_agent_graph``).
    tool_registry:
        Tool registry used to build ``available_tools`` for the agent state.
    bot_token:
        Slack Bot OAuth token (``xoxb-...``).
    signing_secret:
        Slack app signing secret for request verification.
    owner_id:
        If set, messages from any other Slack user ID are silently ignored.
    """

    def __init__(
        self,
        graph: Any,
        tool_registry: Any,
        bot_token: str,
        signing_secret: str,
        owner_id: str | None = None,
    ) -> None:
        self._graph = graph
        self._tool_registry = tool_registry
        self._bot_token = bot_token
        self._signing_secret = signing_secret
        self._owner_id = owner_id
        self._bolt_app = self._build_bolt_app()

    # ------------------------------------------------------------------
    # Internal helpers
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
        """Run the agent graph and return the final text response."""
        state_input = {
            "messages": [HumanMessage(content=text)],
            "available_tools": self._tool_defs(),
        }
        result = await self._graph.ainvoke(
            state_input,
            config={"configurable": {"thread_id": thread_id}},
        )
        messages = result.get("messages", [])
        last_ai = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage)), None
        )
        if last_ai is None:
            return "(no response)"
        content = last_ai.content
        if isinstance(content, list):  # Anthropic content blocks
            content = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in content
            )
        return content

    async def _handle(self, event: dict[str, Any], say: Any) -> None:
        """Shared handler for app_mention and DM message events."""
        user_id = event.get("user") or event.get("username", "")
        channel_id = event.get("channel", "")
        thread_ts = event.get("thread_ts")  # non-None when inside a thread
        text_raw = event.get("text", "")

        # Ignore messages from bots (including ourselves)
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            return

        # Owner restriction
        if self._owner_id and user_id != self._owner_id:
            return

        text = _strip_mention(text_raw)
        if not text:
            return

        # Thread ID: prefer thread continuity, fall back to channel DM session
        if thread_ts:
            thread_id = f"slack-{channel_id}-{thread_ts}"
        else:
            thread_id = f"slack-{channel_id}"

        logger.info(
            "Slack message from user=%s channel=%s thread=%s: %.80s",
            user_id,
            channel_id,
            thread_id,
            text,
        )

        try:
            response = await self._invoke(text, thread_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Agent error for thread %s", thread_id)
            await say(
                text=f"Sorry, something went wrong: {exc}",
                thread_ts=thread_ts,
            )
            return

        # Post each chunk in the same thread (or top-level for DMs)
        for chunk in _split_message(response):
            await say(text=chunk, thread_ts=thread_ts)

    # ------------------------------------------------------------------
    # Bolt app construction
    # ------------------------------------------------------------------

    def _build_bolt_app(self) -> Any:
        try:
            from slack_bolt.async_app import AsyncApp
        except ImportError as exc:
            raise ImportError(
                "slack-bolt is required for the Slack channel.\n"
                "Install it with: pip install \"titanclaw[slack]\""
            ) from exc

        bolt = AsyncApp(
            token=self._bot_token,
            signing_secret=self._signing_secret,
        )

        @bolt.event("app_mention")
        async def on_mention(event: dict, say: Any) -> None:
            await self._handle(event, say)

        @bolt.event({"type": "message", "channel_type": "im"})
        async def on_dm(event: dict, say: Any) -> None:
            await self._handle(event, say)

        return bolt

    # ------------------------------------------------------------------
    # ASGI entry point
    # ------------------------------------------------------------------

    @property
    def asgi_app(self) -> Any:
        """Return the Slack Bolt ASGI handler for use with uvicorn.

        Mount at the root; Bolt serves ``POST /slack/events`` internally.
        """
        try:
            from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
        except ImportError as exc:
            raise ImportError(
                "slack-bolt is required for the Slack channel.\n"
                "Install it with: pip install \"titanclaw[slack]\""
            ) from exc

        from fastapi import FastAPI, Request, Response

        api = FastAPI(title="TitanClaw Slack Gateway", docs_url=None, redoc_url=None)
        handler = AsyncSlackRequestHandler(self._bolt_app)

        @api.post("/slack/events")
        async def slack_events(req: Request) -> Response:
            return await handler.handle(req)

        @api.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        return api
