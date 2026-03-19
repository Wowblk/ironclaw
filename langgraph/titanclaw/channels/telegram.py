"""Telegram channel adapter — polling-based bot using python-telegram-bot.

Mirrors the WASM Telegram channel in channels-src/telegram/ but runs natively
in Python without needing a public webhook URL.

Usage::

    adapter = TelegramAdapter(
        graph=graph,
        tool_registry=tool_registry,
        bot_token="123456:ABC-...",
        owner_id=123456789,   # optional: restrict to one Telegram user ID
    )
    await adapter.start_polling()   # blocks until cancelled

Thread IDs
----------
Each Telegram chat maps to a single LangGraph thread so conversation history
is maintained across messages:

    thread_id = f"tg-{chat_id}"

This means every chat (DM, group, or forum topic) gets its own persistent
session with the agent.

Message splitting
-----------------
Telegram enforces a 4096-character limit per message.  Responses longer than
that are split on newline boundaries to avoid cutting words.

Dependencies
------------
Requires ``python-telegram-bot[asyncio]>=21``.  Install with::

    pip install "titanclaw[telegram]"
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger(__name__)

_TELEGRAM_LIMIT = 4096


def _split_message(text: str, limit: int = _TELEGRAM_LIMIT) -> list[str]:
    """Split *text* into chunks of at most *limit* characters.

    Splits preferably on newline boundaries so code blocks and paragraphs
    stay intact.
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        # Try to break on a newline within the limit
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return chunks


class TelegramAdapter:
    """
    Telegram channel adapter.

    Receives messages via long-polling and invokes the LangGraph agent
    for each one, sending the response back to the same chat.

    Parameters
    ----------
    graph:
        Compiled LangGraph graph (from ``build_agent_graph``).
    tool_registry:
        Tool registry used to build ``available_tools`` for the agent state.
    bot_token:
        Telegram Bot API token (from @BotFather).
    owner_id:
        If set, messages from any other Telegram user ID are ignored with a
        polite reply.  Leave ``None`` to allow all users.
    """

    def __init__(
        self,
        graph: Any,
        tool_registry: Any,
        bot_token: str,
        owner_id: int | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._graph = graph
        self._tool_registry = tool_registry
        self._bot_token = bot_token
        self._owner_id = owner_id
        self._system_prompt = system_prompt

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

    # ------------------------------------------------------------------
    # Message handler
    # ------------------------------------------------------------------

    async def _handle_message(self, update: Any, context: Any) -> None:
        """PTB MessageHandler callback for incoming text messages."""
        from telegram.constants import ChatAction

        message = update.message
        if message is None or not message.text:
            return

        user = update.effective_user
        chat = update.effective_chat

        # Owner restriction
        if self._owner_id is not None and user.id != self._owner_id:
            await message.reply_text("Sorry, I'm a private assistant.")
            return

        thread_id = f"tg-{chat.id}"
        logger.info(
            "Telegram message from user=%d chat=%d thread=%s: %.80s",
            user.id,
            chat.id,
            thread_id,
            message.text,
        )

        # Show typing indicator while the agent thinks
        await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)

        try:
            response = await self._invoke(message.text, thread_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Agent error for thread %s", thread_id)
            await message.reply_text(f"Sorry, something went wrong: {exc}")
            return

        for chunk in _split_message(response):
            await message.reply_text(chunk)

    async def _handle_command_start(self, update: Any, context: Any) -> None:
        """/start command — welcome message."""
        await update.message.reply_text(
            "Hello! I'm TitanClaw, your AI assistant. Send me a message to get started."
        )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def start_polling(self) -> None:
        """Start the Telegram bot in long-polling mode.  Blocks until cancelled.

        Raises
        ------
        ImportError
            If ``python-telegram-bot`` is not installed.
        """
        try:
            from telegram.ext import (
                Application,
                CommandHandler,
                MessageHandler,
                filters,
            )
        except ImportError as exc:
            raise ImportError(
                "python-telegram-bot is required for the Telegram channel.\n"
                "Install it with: pip install \"titanclaw[telegram]\""
            ) from exc

        app = Application.builder().token(self._bot_token).build()
        app.add_handler(CommandHandler("start", self._handle_command_start))
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

        logger.info("Starting Telegram bot (polling mode)")
        async with app:
            await app.initialize()
            await app.start()
            await app.updater.start_polling(drop_pending_updates=True)
            logger.info("Telegram bot is running — press Ctrl-C to stop")
            # Keep running until cancelled
            import asyncio
            await asyncio.Event().wait()
