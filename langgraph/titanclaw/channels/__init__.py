"""Multi-channel input — mirrors src/channels/."""

from titanclaw.channels.channel import Channel, IncomingMessage, OutgoingResponse
from titanclaw.channels.repl import ReplChannel

__all__ = ["Channel", "IncomingMessage", "OutgoingResponse", "ReplChannel"]
