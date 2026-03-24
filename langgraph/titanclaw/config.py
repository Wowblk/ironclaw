"""Configuration — mirrors src/config/ from the Rust implementation.

All settings are read from environment variables (or .env file).
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LlmConfig(BaseSettings):
    """LLM provider configuration."""

    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")

    backend: Literal[
        "openai", "anthropic", "openai_compatible", "ollama"
    ] = "anthropic"
    model: str = "claude-sonnet-4-6"
    base_url: str | None = None
    api_key: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.0


class AgentConfig(BaseSettings):
    """Core agent settings."""

    model_config = SettingsConfigDict(env_prefix="AGENT_", extra="ignore")

    name: str = "TitanClaw"
    max_parallel_jobs: int = Field(default=5, ge=1)
    job_timeout_seconds: int = 300
    max_iterations: int = 50
    max_tokens_per_job: int = 0  # 0 = unlimited
    use_planning: bool = False
    session_idle_timeout_seconds: int = 3600
    allow_local_tools: bool = True
    auto_approve_tools: bool = False
    # Cost guard
    max_cost_per_day_cents: int | None = None
    max_actions_per_hour: int | None = None
    # Tool nudge
    enable_tool_intent_nudge: bool = True
    max_tool_intent_nudges: int = 2
    # Default timezone
    default_timezone: str = "UTC"
    # Workspace directory — filesystem backend for persistent memory & identity files.
    # Defaults to ~/.titanclaw/workspace (mirrors the Rust bootstrap base dir).
    workspace_dir: str = Field(
        default_factory=lambda: os.path.expanduser("~/.titanclaw/workspace"),
    )


class SafetyConfig(BaseSettings):
    """Safety layer configuration.

    Environment variables (prefix SAFETY_)
    ----------------------------------------
    SAFETY_INJECTION_CHECK_ENABLED   : bool   — prompt injection detection (default true)
    SAFETY_MAX_OUTPUT_LENGTH         : int    — max chars in tool output (default 100000)
    SAFETY_LEAK_DETECTION_ENABLED    : bool   — scan tool output for secrets (default true)
    SAFETY_LEAK_ACTION               : str    — "redact" | "block" (default "redact")
    """

    model_config = SettingsConfigDict(env_prefix="SAFETY_", extra="ignore")

    injection_check_enabled: bool = True
    max_output_length: int = 100_000
    leak_detection_enabled: bool = True
    leak_action: str = "redact"   # "redact" | "block"


class DatabaseConfig(BaseSettings):
    """Database connection settings."""

    model_config = SettingsConfigDict(env_prefix="DATABASE_", extra="ignore")

    url: str | None = None


class TelegramConfig(BaseSettings):
    """Telegram bot configuration."""

    model_config = SettingsConfigDict(extra="ignore")

    bot_token: str | None = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    owner_id: int | None = Field(default=None, alias="TELEGRAM_OWNER_ID")


class SlackConfig(BaseSettings):
    """Slack Events API configuration."""

    model_config = SettingsConfigDict(extra="ignore")

    bot_token: str | None = Field(default=None, alias="SLACK_BOT_TOKEN")
    signing_secret: str | None = Field(default=None, alias="SLACK_SIGNING_SECRET")
    webhook_host: str = Field(default="0.0.0.0", alias="SLACK_WEBHOOK_HOST")
    webhook_port: int = Field(default=3000, alias="SLACK_WEBHOOK_PORT")


class FeishuConfig(BaseSettings):
    """Feishu (Lark / 飞书) Events API configuration."""

    model_config = SettingsConfigDict(extra="ignore")

    app_id: str | None = Field(default=None, alias="FEISHU_APP_ID")
    app_secret: str | None = Field(default=None, alias="FEISHU_APP_SECRET")
    verification_token: str | None = Field(default=None, alias="FEISHU_VERIFICATION_TOKEN")
    encrypt_key: str | None = Field(default=None, alias="FEISHU_ENCRYPT_KEY")
    webhook_host: str = Field(default="0.0.0.0", alias="FEISHU_WEBHOOK_HOST")
    webhook_port: int = Field(default=8010, alias="FEISHU_WEBHOOK_PORT")


class WeComConfig(BaseSettings):
    """WeCom (企业微信) Callback API configuration."""

    model_config = SettingsConfigDict(extra="ignore")

    corp_id: str | None = Field(default=None, alias="WECOM_CORP_ID")
    corp_secret: str | None = Field(default=None, alias="WECOM_CORP_SECRET")
    agent_id: int | None = Field(default=None, alias="WECOM_AGENT_ID")
    token: str | None = Field(default=None, alias="WECOM_TOKEN")
    encoding_aes_key: str | None = Field(default=None, alias="WECOM_ENCODING_AES_KEY")
    webhook_host: str = Field(default="0.0.0.0", alias="WECOM_WEBHOOK_HOST")
    webhook_port: int = Field(default=8020, alias="WECOM_WEBHOOK_PORT")


class QQConfig(BaseSettings):
    """QQ channel configuration (OneBot v11 HTTP callback)."""

    model_config = SettingsConfigDict(extra="ignore")

    onebot_api_url: str = Field(
        default="http://localhost:3000", alias="ONEBOT_API_URL"
    )
    access_token: str | None = Field(default=None, alias="ONEBOT_ACCESS_TOKEN")
    self_id: int | None = Field(default=None, alias="ONEBOT_SELF_ID")
    webhook_host: str = Field(default="0.0.0.0", alias="QQ_WEBHOOK_HOST")
    webhook_port: int = Field(default=8030, alias="QQ_WEBHOOK_PORT")


class ChannelConfig(BaseSettings):
    """Channel enablement flags."""

    model_config = SettingsConfigDict(extra="ignore")

    repl_enabled: bool = Field(default=True, alias="REPL_ENABLED")
    http_enabled: bool = Field(default=False, alias="HTTP_ENABLED")
    http_port: int = Field(default=8080, alias="HTTP_PORT")
    http_secret: str | None = Field(default=None, alias="HTTP_SECRET")

    # Web gateway (browser UI + SSE streaming)
    web_enabled: bool = Field(default=False, alias="WEB_ENABLED")
    web_host: str = Field(default="0.0.0.0", alias="WEB_HOST")
    web_port: int = Field(default=8000, alias="WEB_PORT")
    web_cors_origins: list[str] = Field(default=["*"], alias="WEB_CORS_ORIGINS")


class SupermemoryConfig(BaseSettings):
    """Supermemory cloud memory configuration."""

    model_config = SettingsConfigDict(extra="ignore")

    api_key: str | None = Field(default=None, alias="SUPERMEMORY_API_KEY")
    container_tag: str = Field(default="titanclaw", alias="SUPERMEMORY_CONTAINER_TAG")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


class OrchestratorConfig(BaseSettings):
    """Internal orchestrator HTTP API configuration.

    The orchestrator allows sandbox containers to make LLM calls back to the
    host without holding an API key themselves.  A per-job bearer token is
    injected as ORCHESTRATOR_TOKEN; the URL as ORCHESTRATOR_URL.

    Environment variables
    ---------------------
    ORCHESTRATOR_ENABLED   : bool — start the orchestrator (default false)
    ORCHESTRATOR_HOST      : str  — bind host (default 127.0.0.1)
    """

    model_config = SettingsConfigDict(extra="ignore")

    enabled: bool = Field(default=False, alias="ORCHESTRATOR_ENABLED")
    host: str = Field(default="127.0.0.1", alias="ORCHESTRATOR_HOST")


class OrphanReaperConfig(BaseSettings):
    """Container orphan reaper configuration.

    Environment variables
    ---------------------
    REAPER_ENABLED        : bool — enable background reaper (default true when docker sandbox on)
    REAPER_INTERVAL_SECS  : int  — scan interval in seconds (default 300)
    REAPER_THRESHOLD_SECS : int  — containers older than this are orphans (default 600)
    """

    model_config = SettingsConfigDict(extra="ignore")

    enabled: bool = Field(default=True, alias="REAPER_ENABLED")
    interval_secs: int = Field(default=300, alias="REAPER_INTERVAL_SECS")
    threshold_secs: int = Field(default=600, alias="REAPER_THRESHOLD_SECS")


class DockerSandboxConfig(BaseSettings):
    """Docker-based execution sandbox configuration.

    Environment variables
    ---------------------
    DOCKER_SANDBOX_ENABLED          : bool   — enable the sandbox (default false)
    DOCKER_SANDBOX_IMAGE            : str    — base image (default python:3.12-slim)
    DOCKER_SANDBOX_MEMORY_MB        : int    — memory cap in MiB (default 512)
    DOCKER_SANDBOX_CPU_QUOTA        : float  — fractional CPUs (default 1.0)
    DOCKER_SANDBOX_POLICY           : str    — read_only | workspace_write | full_access
    DOCKER_SANDBOX_NETWORK          : str    — docker network mode (default none)
    DOCKER_SANDBOX_ALLOWED_DOMAINS  : str    — comma-separated domain allowlist
    DOCKER_SANDBOX_CREDENTIALS      : str    — JSON array of credential mapping dicts
    DOCKER_SANDBOX_TIMEOUT          : float  — default command timeout seconds (default 60)

    FullAccess double opt-in
    ------------------------
    Setting policy=full_access is not sufficient alone.  The container host
    must also export SANDBOX_ALLOW_FULL_ACCESS=true.  Without it the sandbox
    downgrades to workspace_write and logs an error.  This mirrors the safety
    guard in src/sandbox/manager.rs.
    """

    model_config = SettingsConfigDict(extra="ignore")

    enabled: bool = Field(default=False, alias="DOCKER_SANDBOX_ENABLED")
    image: str = Field(default="python:3.12-slim", alias="DOCKER_SANDBOX_IMAGE")
    memory_mb: int = Field(default=512, alias="DOCKER_SANDBOX_MEMORY_MB")
    cpu_quota: float = Field(default=1.0, alias="DOCKER_SANDBOX_CPU_QUOTA")
    # "read_only" | "workspace_write" | "full_access"
    policy: str = Field(default="read_only", alias="DOCKER_SANDBOX_POLICY")
    network_mode: str = Field(default="none", alias="DOCKER_SANDBOX_NETWORK")
    # Comma-separated list: "pypi.org,*.github.com"
    allowed_domains_csv: str = Field(default="", alias="DOCKER_SANDBOX_ALLOWED_DOMAINS")
    # JSON: '[{"host_pattern":"api.openai.com","secret_env_var":"OPENAI_API_KEY","location":"bearer"}]'
    credentials_json: str = Field(default="[]", alias="DOCKER_SANDBOX_CREDENTIALS")
    timeout: float = Field(default=60.0, alias="DOCKER_SANDBOX_TIMEOUT")

    @property
    def allowed_domains(self) -> list[str]:
        return [d.strip() for d in self.allowed_domains_csv.split(",") if d.strip()]

    @property
    def credential_mappings(self) -> list[dict]:
        import json as _json
        try:
            return _json.loads(self.credentials_json) or []
        except Exception:  # noqa: BLE001
            return []


class WasmSandboxConfig(BaseSettings):
    """WASM/WASI tool sandbox configuration.

    Environment variables
    ---------------------
    WASM_SANDBOX_ENABLED  : bool  — enable the sandbox (default false)
    WASM_TOOLS_DIR        : str   — directory of .wasm tool modules
    WASM_FUEL             : int   — CPU fuel limit per call (default 1B)
    WASM_MAX_MEMORY_MB    : int   — linear memory cap in MiB (default 64)
    WASM_TIMEOUT          : float — wall-clock timeout per call (default 30)
    """

    model_config = SettingsConfigDict(extra="ignore")

    enabled: bool = Field(default=False, alias="WASM_SANDBOX_ENABLED")
    tools_dir: str = Field(
        default_factory=lambda: os.path.expanduser("~/.titanclaw/wasm-tools"),
        alias="WASM_TOOLS_DIR",
    )
    fuel: int = Field(default=1_000_000_000, alias="WASM_FUEL")
    max_memory_mb: int = Field(default=64, alias="WASM_MAX_MEMORY_MB")
    timeout: float = Field(default=30.0, alias="WASM_TIMEOUT")


class Config(BaseSettings):
    """Root configuration — composes all subsystem configs."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm: LlmConfig = Field(default_factory=LlmConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    docker_sandbox: DockerSandboxConfig = Field(default_factory=DockerSandboxConfig)
    wasm_sandbox: WasmSandboxConfig = Field(default_factory=WasmSandboxConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    reaper: OrphanReaperConfig = Field(default_factory=OrphanReaperConfig)
    channels: ChannelConfig = Field(default_factory=ChannelConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    slack: SlackConfig = Field(default_factory=SlackConfig)
    feishu: FeishuConfig = Field(default_factory=FeishuConfig)
    wecom: WeComConfig = Field(default_factory=WeComConfig)
    qq: QQConfig = Field(default_factory=QQConfig)
    supermemory: SupermemoryConfig = Field(default_factory=SupermemoryConfig)

    @classmethod
    def load(cls) -> "Config":
        """Load config from environment, with .env fallback."""
        return cls()
