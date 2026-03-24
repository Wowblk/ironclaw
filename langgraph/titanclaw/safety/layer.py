"""Safety layer — mirrors crates/titanclaw_safety/.

Provides prompt-injection detection, content sanitization, output length
enforcement, and secret/leak detection.  The policy engine maps detection
results to actions: Block, Warn, Sanitize, Redact.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Injection patterns
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore (all )?previous instructions?",
        r"disregard (your|all|the) (previous |prior )?instructions?",
        r"you are now",
        r"new persona",
        r"act as (a|an|the)\b",
        r"forget (your|all|everything|previous)",
        r"system prompt",
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"\[INST\]",
        r"\[/INST\]",
        r"### (Human|Assistant|System):",
        r"jailbreak",
        r"DAN mode",
        r"developer mode",
    ]
]


# ---------------------------------------------------------------------------
# Secret / credential leak patterns
#
# Each entry: (kind_label, compiled_pattern)
# Patterns are ordered by specificity (most specific first) to produce the
# most useful redaction labels.
#
# Mirrors crates/titanclaw_safety/src/leak_detector.rs
# ---------------------------------------------------------------------------

_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Anthropic API key
    ("anthropic_api_key",   re.compile(r'\bsk-ant-[A-Za-z0-9_-]{20,}\b')),
    # OpenAI API key (including project keys "sk-proj-")
    ("openai_api_key",      re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b')),
    # AWS access key ID
    ("aws_access_key_id",   re.compile(r'\bAKIA[0-9A-Z]{16}\b')),
    # AWS session token / secret (heuristic: long base64 strings starting with common prefixes)
    ("aws_secret_key",      re.compile(r'\b(?:aws_secret|AWS_SECRET)[^=\s]*\s*[=:]\s*[A-Za-z0-9/+=]{40}\b', re.IGNORECASE)),
    # GitHub personal access tokens (classic and fine-grained)
    ("github_token",        re.compile(r'\bgh[posaur]_[A-Za-z0-9]{36,}\b')),
    # Slack bot/user/app tokens
    ("slack_token",         re.compile(r'\bxox[bpas]-[0-9A-Za-z]{10,}-[0-9A-Za-z-]{10,}\b')),
    # Stripe live/test keys
    ("stripe_key",          re.compile(r'\b(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{16,}\b')),
    # Telegram bot token
    ("telegram_bot_token",  re.compile(r'\b\d{8,10}:[A-Za-z0-9_-]{35}\b')),
    # Google API key
    ("google_api_key",      re.compile(r'\bAIza[0-9A-Za-z_-]{35}\b')),
    # HuggingFace token
    ("huggingface_token",   re.compile(r'\bhf_[A-Za-z0-9]{30,}\b')),
    # JWT (three base64url segments separated by dots)
    ("jwt",                 re.compile(r'\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b')),
    # Generic bearer token in HTTP header
    ("bearer_token",        re.compile(r'\bBearer\s+[A-Za-z0-9_\-\.]{20,}\b')),
    # URL with embedded credentials  user:pass@host
    ("url_with_credentials", re.compile(r'[a-zA-Z][a-zA-Z0-9+\-.]*://[^:@\s]+:[^@\s]+@[^\s/]+')),
]

# Verbatim text that replaces each match  (kind is inserted so context is preserved)
def _redact_label(kind: str) -> str:
    return f"[REDACTED:{kind}]"


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class PolicyAction(Enum):
    """Action to take when a safety rule fires."""

    Allow = "allow"
    Warn = "warn"
    Sanitize = "sanitize"
    Block = "block"
    Redact = "redact"


@dataclass
class SafetyViolation:
    """A detected safety issue."""

    pattern: str
    action: PolicyAction
    sanitized: str | None = None


class SafetyResult(NamedTuple):
    """Result of a safety check."""

    safe: bool
    violations: list[SafetyViolation]
    sanitized_content: str


# ---------------------------------------------------------------------------
# Leak scan result
# ---------------------------------------------------------------------------

@dataclass
class LeakScanResult:
    """Result of scanning text for credential/secret leaks."""

    found: list[str]            # human-readable descriptions of what was found
    redacted_text: str          # text with secrets replaced by [REDACTED:kind]
    has_leaks: bool             # True if any pattern matched


def _scan_for_leaks(text: str) -> LeakScanResult:
    """
    Scan *text* for credential leaks.  Returns a ``LeakScanResult`` with:
    * ``found``         — list of kind labels for each unique match
    * ``redacted_text`` — text with each match replaced by ``[REDACTED:<kind>]``
    * ``has_leaks``     — True if any pattern matched

    Mirrors ``LeakDetector::scan_and_clean`` in
    ``crates/titanclaw_safety/src/leak_detector.rs``.
    """
    found: list[str] = []
    current = text
    for kind, pattern in _SECRET_PATTERNS:
        matches = pattern.findall(current)
        if matches:
            found.append(kind)
            current = pattern.sub(_redact_label(kind), current)
    return LeakScanResult(found=found, redacted_text=current, has_leaks=bool(found))


# ---------------------------------------------------------------------------
# SafetyLayer
# ---------------------------------------------------------------------------

class SafetyLayer:
    """
    Multi-layer safety enforcement.

    Pipeline (tool output):
        raw output
          → leak detection (redact/block secrets in output)
          → injection check
          → truncation
          → XML boundary wrapping
          → sanitized output

    Pipeline (inbound message):
        raw message
          → credential scan (warn/reject if user sends API keys)
          → injection patterns
          → safe to forward to LLM

    Mirrors the Rust ``SafetyLayer`` in ``crates/titanclaw_safety/``.
    """

    def __init__(
        self,
        injection_check_enabled: bool = True,
        max_output_length: int = 100_000,
        leak_detection_enabled: bool = True,
        leak_action: str = "redact",   # "redact" | "block"
    ) -> None:
        self.injection_check_enabled = injection_check_enabled
        self.max_output_length = max_output_length
        self.leak_detection_enabled = leak_detection_enabled
        self.leak_action = leak_action  # what to do when a leak is found in tool output

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_input(self, content: str) -> SafetyResult:
        """Check and sanitize user or tool input before passing to the LLM."""
        violations: list[SafetyViolation] = []
        current = content

        if self.injection_check_enabled:
            for pattern in _INJECTION_PATTERNS:
                if pattern.search(current):
                    violation = SafetyViolation(
                        pattern=pattern.pattern,
                        action=PolicyAction.Block,
                    )
                    violations.append(violation)

        blocked = any(v.action == PolicyAction.Block for v in violations)
        return SafetyResult(
            safe=not blocked,
            violations=violations,
            sanitized_content=current,
        )

    def scan_inbound_for_secrets(self, content: str) -> SafetyResult:
        """
        Scan an **inbound user message** for embedded credentials before it is
        forwarded to the LLM.

        If the user pastes an API key, returning it verbatim in the LLM context
        risks leaking it through logs, embeddings, or tool calls.  We warn and
        scrub the content.

        Mirrors ``SafetyLayer::scan_inbound_for_secrets`` in
        ``crates/titanclaw_safety/src/lib.rs:142-156``.
        """
        result = _scan_for_leaks(content)
        if not result.has_leaks:
            return SafetyResult(safe=True, violations=[], sanitized_content=content)

        violations = [
            SafetyViolation(pattern=kind, action=PolicyAction.Redact)
            for kind in result.found
        ]
        return SafetyResult(
            safe=True,       # warn+redact, do not hard-block inbound messages
            violations=violations,
            sanitized_content=result.redacted_text,
        )

    def sanitize_tool_output(self, output: str) -> str:
        """
        Sanitize tool output before injecting into the LLM context.

        Steps
        -----
        1. **Leak detection** — scan for and redact (or block) any credentials
           accidentally written to stdout by the tool (e.g. env dumps, debug
           output).
        2. **Truncation** — cap at ``max_output_length`` characters.
        3. **HTML escaping** — prevent tag injection.
        4. **XML boundary** — wrap in ``<tool_output>`` so the model cannot
           interpret the content as instructions.
        """
        current = output

        # 1. Leak detection
        if self.leak_detection_enabled:
            leak = _scan_for_leaks(current)
            if leak.has_leaks:
                if self.leak_action == "block":
                    kinds = ", ".join(leak.found)
                    return (
                        "<tool_output>\n"
                        f"[OUTPUT BLOCKED — potential credential leak detected: {kinds}]\n"
                        "</tool_output>"
                    )
                # default: redact
                current = leak.redacted_text

        # 2. Truncation (UTF-8 safe via Python string indexing on decoded str)
        if len(current) > self.max_output_length:
            current = current[: self.max_output_length]
            current += f"\n[… output truncated at {self.max_output_length} characters]"

        # 3 + 4. Escape and wrap
        safe = html.escape(current, quote=False)
        return f"<tool_output>\n{safe}\n</tool_output>"

    def check_output(self, content: str) -> SafetyResult:
        """Check agent output before sending to the user."""
        violations: list[SafetyViolation] = []

        if len(content) > self.max_output_length:
            violations.append(
                SafetyViolation(
                    pattern="max_output_length",
                    action=PolicyAction.Warn,
                )
            )

        return SafetyResult(
            safe=True,
            violations=violations,
            sanitized_content=content,
        )

    def scan_for_secrets(self, content: str, secrets: list[str]) -> bool:
        """Return True if any known secret value appears verbatim in content."""
        return any(secret in content for secret in secrets if secret)

    # ------------------------------------------------------------------
    # Direct access to the leak scanner (for callers that need raw results)
    # ------------------------------------------------------------------

    @staticmethod
    def scan_leaks(text: str) -> LeakScanResult:
        """Scan *text* for credential patterns; return a ``LeakScanResult``."""
        return _scan_for_leaks(text)
