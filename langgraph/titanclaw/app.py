"""
Application startup — mirrors src/app.rs.

Wires together: config → LLM → tools → safety → graph → channels → run loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from titanclaw.config import Config
from titanclaw.graph import AgentDeps, build_agent_graph
from titanclaw.memory.workspace import Workspace
from titanclaw.safety.layer import SafetyLayer
from titanclaw.scheduler.scheduler import JobScheduler
from titanclaw.state import AgentState, ToolDefinition
from titanclaw.tools.registry import ToolRegistry
from titanclaw.tools.builtin import (
    echo_tool,
    read_file_tool,
    write_file_tool,
    list_dir_tool,
    http_get_tool,
    http_post_tool,
    memory_search_tool,
    memory_write_tool,
    memory_read_tool,
    shell_tool,
    time_tool,
)
from titanclaw.channels.channel import OutgoingResponse
from titanclaw.channels.repl import ReplChannel
from titanclaw.channels.web import WebGateway
from titanclaw.channels.telegram import TelegramAdapter
from titanclaw.channels.slack import SlackAdapter
from titanclaw.channels.feishu import FeishuAdapter
from titanclaw.channels.wecom import WeComAdapter
from titanclaw.channels.qq import QQAdapter

logger = logging.getLogger(__name__)


def _build_llm(config: Config) -> Any:
    """Instantiate the LLM from config.  Mirrors src/llm/ provider selection."""
    backend = config.llm.backend
    model = config.llm.model
    temperature = config.llm.temperature
    max_tokens = config.llm.max_tokens

    if backend == "anthropic":
        from langchain_anthropic import ChatAnthropic
        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if config.llm.api_key:
            kwargs["anthropic_api_key"] = config.llm.api_key
        return ChatAnthropic(**kwargs)

    if backend == "openai":
        from langchain_openai import ChatOpenAI
        kwargs = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if config.llm.api_key:
            kwargs["openai_api_key"] = config.llm.api_key
        return ChatOpenAI(**kwargs)

    if backend in ("openai_compatible", "ollama"):
        from langchain_openai import ChatOpenAI
        kwargs = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "base_url": config.llm.base_url or "http://localhost:11434/v1",
            "api_key": config.llm.api_key or "ollama",
        }
        return ChatOpenAI(**kwargs)

    raise ValueError(f"Unknown LLM backend: {backend!r}")


def _build_tool_registry(config: Config) -> ToolRegistry:
    """Register built-in tools.  Mirrors src/tools/builtin/mod.rs."""
    registry = ToolRegistry()
    registry.register(echo_tool)
    registry.register(time_tool)
    registry.register(memory_search_tool)
    registry.register(memory_write_tool)
    registry.register(memory_read_tool)

    if config.agent.allow_local_tools:
        registry.register(read_file_tool)
        registry.register(write_file_tool)
        registry.register(list_dir_tool)
        registry.register(http_get_tool)
        registry.register(http_post_tool)
        # Register the bare shell tool first; sandboxes may override it below.
        registry.register(shell_tool)

    # --- Docker sandbox ---
    docker_cfg = config.docker_sandbox
    if docker_cfg.enabled and config.agent.allow_local_tools:
        from titanclaw.tools.sandbox.docker_sandbox import DockerSandbox, SandboxPolicy
        try:
            policy = SandboxPolicy(docker_cfg.policy)
        except ValueError:
            logger.warning(
                "Unknown DOCKER_SANDBOX_POLICY=%r — defaulting to read_only",
                docker_cfg.policy,
            )
            policy = SandboxPolicy.READ_ONLY

        docker_sandbox = DockerSandbox(
            image=docker_cfg.image,
            memory_mb=docker_cfg.memory_mb,
            cpu_quota=docker_cfg.cpu_quota,
            policy=policy,
            network_mode=docker_cfg.network_mode,
            workspace_dir=config.agent.workspace_dir,
            allowed_domains=docker_cfg.allowed_domains,
            credential_mappings=docker_cfg.credential_mappings,
            timeout=docker_cfg.timeout,
        )
        # Overrides the bare shell tool registered above.
        registry.register(docker_sandbox.create_shell_tool())
        logger.info(
            "Docker sandbox enabled — shell commands run in %s "
            "(policy=%s, memory=%dMiB, cpus=%.1f, network=%s, allowed_domains=%d)",
            docker_cfg.image,
            policy.value,
            docker_cfg.memory_mb,
            docker_cfg.cpu_quota,
            docker_cfg.network_mode if not docker_cfg.allowed_domains else "proxy",
            len(docker_cfg.allowed_domains),
        )

    # --- WASM sandbox ---
    wasm_cfg = config.wasm_sandbox
    if wasm_cfg.enabled:
        from titanclaw.tools.sandbox.wasm_sandbox import WasmSandbox
        wasm_sandbox = WasmSandbox(
            tools_dir=wasm_cfg.tools_dir,
            fuel=wasm_cfg.fuel,
            max_memory_mb=wasm_cfg.max_memory_mb,
            workspace_dir=config.agent.workspace_dir,
            timeout=wasm_cfg.timeout,
        )
        wasm_tools = wasm_sandbox.load_tools()
        for wt in wasm_tools:
            registry.register(wt)
        logger.info(
            "WASM sandbox enabled — loaded %d tool(s) from %s (fuel=%d, max_mem=%dMiB)",
            len(wasm_tools),
            wasm_cfg.tools_dir,
            wasm_cfg.fuel,
            wasm_cfg.max_memory_mb,
        )

    return registry


def _tool_defs_from_registry(registry: ToolRegistry) -> list[ToolDefinition]:
    """Convert registry tools to state ToolDefinition objects."""
    return [
        ToolDefinition(
            name=t.name,
            description=t.description,
            parameters=t.parameters_schema,
        )
        for t in registry.list_tools()
    ]


async def _run_repl(
    graph: Any,
    registry: ToolRegistry,
    config: Config,
    system_prompt: str | None = None,
) -> None:
    """Interactive REPL loop — mirrors src/channels/repl.rs."""
    channel = ReplChannel()
    tool_defs = _tool_defs_from_registry(registry)

    print(f"\nTitanClaw (LangGraph) — type your message, /quit to exit\n")

    async for msg in channel.receive():
        if msg.content in ("/quit", "/exit"):
            print("Goodbye.")
            break

        state_input: dict[str, Any] = {
            "messages": [HumanMessage(content=msg.content)],
            "user_id": msg.user_id,
            "available_tools": tool_defs,
        }
        if system_prompt:
            state_input["system_prompt"] = system_prompt
        graph_config = {"configurable": {"thread_id": msg.thread_id}}

        try:
            result = await graph.ainvoke(state_input, config=graph_config)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Graph invocation error")
            await channel.send(OutgoingResponse(
                content=f"Error: {exc}",
                thread_id=msg.thread_id,
            ))
            continue

        # Extract last AI message
        messages = result.get("messages", [])
        last_ai = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage)),
            None,
        )
        response_text = last_ai.content if last_ai else "(no response)"
        if isinstance(response_text, list):
            # Handle content blocks (Anthropic format)
            response_text = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in response_text
            )

        await channel.send(OutgoingResponse(
            content=response_text,
            thread_id=msg.thread_id,
        ))


class TitanclawApp:
    """
    Top-level application.  Mirrors the Rust ``App`` struct.

    Usage::

        app = TitanclawApp(config)
        await app.run()           # interactive REPL
        await app.run_web()       # FastAPI web gateway on WEB_PORT
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config.load()
        self.llm = _build_llm(self.config)
        self.tool_registry = _build_tool_registry(self.config)
        self.safety = SafetyLayer(
            injection_check_enabled=self.config.safety.injection_check_enabled,
            max_output_length=self.config.safety.max_output_length,
        )
        self.deps = AgentDeps(
            llm=self.llm,
            tool_registry=self.tool_registry,
            safety=self.safety,
            config=self.config.agent,
        )
        self.graph = build_agent_graph(self.deps)
        self.scheduler = JobScheduler(
            max_parallel_jobs=self.config.agent.max_parallel_jobs
        )
        self.workspace = Workspace(base_dir=self.config.agent.workspace_dir)

    async def _load_system_prompt(self) -> str | None:
        """
        Load identity context from workspace and combine with agent name.

        Mirrors the Rust ``Agent::build_system_prompt`` which calls
        ``Workspace::identity_context()`` and prepends the agent name.
        Returns None when no identity files are found.
        """
        identity = await self.workspace.load_identity_context()
        if not identity:
            return None
        header = f"You are {self.config.agent.name}.\n\n"
        return header + identity

    async def run(self) -> None:
        """Start the application.  Default mode: interactive REPL."""
        system_prompt = await self._load_system_prompt()
        if system_prompt:
            logger.info("Loaded identity context (%d chars) into system prompt", len(system_prompt))
        await _run_repl(self.graph, self.tool_registry, self.config, system_prompt=system_prompt)

    async def run_web(
        self,
        host: str | None = None,
        port: int | None = None,
        cors_origins: list[str] | None = None,
    ) -> None:
        """Start the FastAPI web gateway (browser UI + SSE).

        Falls back to ``config.channels`` values when arguments are omitted.
        """
        import uvicorn

        system_prompt = await self._load_system_prompt()
        web_cfg = self.config.channels
        resolved_host = host or web_cfg.web_host
        resolved_port = port or web_cfg.web_port
        resolved_cors = cors_origins or web_cfg.web_cors_origins

        gateway = WebGateway(
            graph=self.graph,
            tool_registry=self.tool_registry,
            cors_origins=resolved_cors,
            system_prompt=system_prompt,
        )

        logger.info(
            "Starting web gateway on http://%s:%d", resolved_host, resolved_port
        )
        cfg = uvicorn.Config(
            gateway.app,
            host=resolved_host,
            port=resolved_port,
            log_level="info",
        )
        await uvicorn.Server(cfg).serve()

    async def run_telegram(self) -> None:
        """Start the Telegram bot in long-polling mode.

        Requires ``TELEGRAM_BOT_TOKEN`` to be set.  Optionally restrict to a
        single user with ``TELEGRAM_OWNER_ID``.
        """
        cfg = self.config.telegram
        if not cfg.bot_token:
            raise ValueError(
                "TELEGRAM_BOT_TOKEN is not set.\n"
                "Get a token from @BotFather and set it in your .env file."
            )
        system_prompt = await self._load_system_prompt()
        adapter = TelegramAdapter(
            graph=self.graph,
            tool_registry=self.tool_registry,
            bot_token=cfg.bot_token,
            owner_id=cfg.owner_id,
            system_prompt=system_prompt,
        )
        await adapter.start_polling()

    async def run_slack(self) -> None:
        """Start the Slack Events API webhook server.

        Requires ``SLACK_BOT_TOKEN`` and ``SLACK_SIGNING_SECRET``.
        Binds to ``SLACK_WEBHOOK_HOST:SLACK_WEBHOOK_PORT`` (default 0.0.0.0:3000).

        Configure your Slack app's Event Subscriptions Request URL to point to
        ``https://<public-host>:<port>/slack/events``.
        """
        import uvicorn

        cfg = self.config.slack
        if not cfg.bot_token or not cfg.signing_secret:
            raise ValueError(
                "SLACK_BOT_TOKEN and SLACK_SIGNING_SECRET must both be set.\n"
                "Find them in your Slack app settings at https://api.slack.com/apps"
            )
        system_prompt = await self._load_system_prompt()
        adapter = SlackAdapter(
            graph=self.graph,
            tool_registry=self.tool_registry,
            bot_token=cfg.bot_token,
            signing_secret=cfg.signing_secret,
            system_prompt=system_prompt,
        )
        logger.info(
            "Starting Slack gateway on http://%s:%d/slack/events",
            cfg.webhook_host,
            cfg.webhook_port,
        )
        server_cfg = uvicorn.Config(
            adapter.asgi_app,
            host=cfg.webhook_host,
            port=cfg.webhook_port,
            log_level="info",
        )
        await uvicorn.Server(server_cfg).serve()

    async def run_feishu(self) -> None:
        """Start the Feishu (飞书) Events API webhook server.

        Requires ``FEISHU_APP_ID``, ``FEISHU_APP_SECRET``, and
        ``FEISHU_VERIFICATION_TOKEN``.  Binds to
        ``FEISHU_WEBHOOK_HOST:FEISHU_WEBHOOK_PORT`` (default 0.0.0.0:8010).

        Set ``FEISHU_ENCRYPT_KEY`` if message encryption is enabled on the
        app console.  Requires ``pip install "titanclaw[feishu]"`` for
        encrypted mode.
        """
        import uvicorn

        cfg = self.config.feishu
        if not cfg.app_id or not cfg.app_secret or not cfg.verification_token:
            raise ValueError(
                "FEISHU_APP_ID, FEISHU_APP_SECRET, and FEISHU_VERIFICATION_TOKEN "
                "must all be set.\nGet them from https://open.feishu.cn/app"
            )
        system_prompt = await self._load_system_prompt()
        adapter = FeishuAdapter(
            graph=self.graph,
            tool_registry=self.tool_registry,
            app_id=cfg.app_id,
            app_secret=cfg.app_secret,
            verification_token=cfg.verification_token,
            encrypt_key=cfg.encrypt_key,
            system_prompt=system_prompt,
        )
        logger.info(
            "Starting Feishu gateway on http://%s:%d/feishu/events",
            cfg.webhook_host,
            cfg.webhook_port,
        )
        server_cfg = uvicorn.Config(
            adapter.asgi_app,
            host=cfg.webhook_host,
            port=cfg.webhook_port,
            log_level="info",
        )
        await uvicorn.Server(server_cfg).serve()

    async def run_wecom(self) -> None:
        """Start the WeCom (企业微信) Callback API webhook server.

        Requires ``WECOM_CORP_ID``, ``WECOM_CORP_SECRET``, ``WECOM_AGENT_ID``,
        ``WECOM_TOKEN``, and ``WECOM_ENCODING_AES_KEY``.  Binds to
        ``WECOM_WEBHOOK_HOST:WECOM_WEBHOOK_PORT`` (default 0.0.0.0:8020).

        Requires ``pip install "titanclaw[wecom]"`` (for AES decryption).
        """
        import uvicorn

        cfg = self.config.wecom
        if not all([
            cfg.corp_id, cfg.corp_secret, cfg.agent_id,
            cfg.token, cfg.encoding_aes_key,
        ]):
            raise ValueError(
                "WECOM_CORP_ID, WECOM_CORP_SECRET, WECOM_AGENT_ID, WECOM_TOKEN, "
                "and WECOM_ENCODING_AES_KEY must all be set.\n"
                "Configure them in your WeCom app at https://work.weixin.qq.com"
            )
        system_prompt = await self._load_system_prompt()
        adapter = WeComAdapter(
            graph=self.graph,
            tool_registry=self.tool_registry,
            corp_id=cfg.corp_id,
            corp_secret=cfg.corp_secret,
            agent_id=cfg.agent_id,
            token=cfg.token,
            encoding_aes_key=cfg.encoding_aes_key,
            system_prompt=system_prompt,
        )
        logger.info(
            "Starting WeCom gateway on http://%s:%d/wecom/events",
            cfg.webhook_host,
            cfg.webhook_port,
        )
        server_cfg = uvicorn.Config(
            adapter.asgi_app,
            host=cfg.webhook_host,
            port=cfg.webhook_port,
            log_level="info",
        )
        await uvicorn.Server(server_cfg).serve()

    async def run_qq(self) -> None:
        """Start the QQ OneBot v11 HTTP callback server.

        Requires a running OneBot v11 implementation (NapCat, LLOneBot, …)
        configured to push events to::

            http://<this-host>:QQWEBHOOK_PORT/qq/events

        Relevant env vars::

            ONEBOT_API_URL=http://localhost:3000
            ONEBOT_ACCESS_TOKEN=...   (optional)
            ONEBOT_SELF_ID=...        (recommended — prevents self-reply loops)
            QQ_WEBHOOK_HOST=0.0.0.0
            QQ_WEBHOOK_PORT=8030
        """
        import uvicorn

        cfg = self.config.qq
        system_prompt = await self._load_system_prompt()
        adapter = QQAdapter(
            graph=self.graph,
            tool_registry=self.tool_registry,
            onebot_api_url=cfg.onebot_api_url,
            access_token=cfg.access_token,
            self_id=cfg.self_id,
            system_prompt=system_prompt,
        )
        logger.info(
            "Starting QQ gateway on http://%s:%d/qq/events (OneBot API: %s)",
            cfg.webhook_host,
            cfg.webhook_port,
            cfg.onebot_api_url,
        )
        server_cfg = uvicorn.Config(
            adapter.asgi_app,
            host=cfg.webhook_host,
            port=cfg.webhook_port,
            log_level="info",
        )
        await uvicorn.Server(server_cfg).serve()
