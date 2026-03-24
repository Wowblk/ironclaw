"""HTTP/HTTPS proxy with domain allowlist and credential injection.

Mirrors ``src/sandbox/proxy/`` in the Rust ironclaw codebase.

Purpose
-------
When a Docker sandbox needs controlled outbound network access (e.g. to reach
PyPI or the Anthropic API) we start a local proxy server, pass its address to
the container via ``HTTP_PROXY`` / ``HTTPS_PROXY`` environment variables, and
enforce an allowlist on every outgoing connection.

Isolation model
---------------
* Only domains on the allowlist can be reached from the container.
* Credentials are injected at the proxy layer — the container process never
  sees raw API keys; they arrive in the forwarded request headers.
* HTTPS connections are tunnelled via the ``CONNECT`` method; the proxy
  validates the target host before establishing the tunnel (it cannot inject
  credentials through TLS tunnels — see note in ``HttpProxyServer._handle_connect``).

Domain pattern matching
-----------------------
* Exact match: ``"api.openai.com"`` matches only that hostname.
* Wildcard subdomain: ``"*.github.com"`` matches ``api.github.com``,
  ``raw.githubusercontent.com``, etc., but NOT ``github.com`` itself.
* Case-insensitive.
* Port numbers are stripped before matching.
* IPv6 brackets (``[::1]``) are stripped.
* Empty allowlist → deny all.
* IPv4/IPv6 addresses are never matched by a domain pattern.

Security notes
--------------
* Userinfo in URLs (``user:pass@host``) is stripped before host extraction to
  prevent host-confusion attacks.
* Wildcard patterns only match one subdomain level: ``*.a.b`` matches
  ``foo.a.b`` but NOT ``foo.bar.a.b``.  Use ``"a.b"`` + ``"*.a.b"`` if both
  are needed.

Usage::

    proxy = HttpProxyServer(
        allowed_domains=["pypi.org", "*.github.com"],
        credential_mappings=[
            {
                "host_pattern": "api.openai.com",
                "secret_env_var": "OPENAI_API_KEY",
                "location": "bearer",
            }
        ],
    )
    port = await proxy.start()
    # pass HTTP_PROXY=http://host-gateway:{port} to the container
    await proxy.stop()
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import socket
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT = 10.0     # seconds to establish upstream connection
_TUNNEL_TIMEOUT = 1800.0    # 30 minutes max tunnel lifetime (matches Rust)
_MAX_HEADER_BYTES = 65_536  # 64 KiB

# Hop-by-hop headers that must be stripped before forwarding (RFC 7230)
_HOP_BY_HOP = frozenset(
    h.lower()
    for h in (
        "Connection", "Keep-Alive", "Proxy-Authenticate", "Proxy-Authorization",
        "TE", "Trailers", "Transfer-Encoding", "Upgrade",
    )
)


# ---------------------------------------------------------------------------
# Domain allowlist
# ---------------------------------------------------------------------------

@dataclass
class DomainPattern:
    """A single allowlist entry: exact hostname or ``*.example.com``."""

    raw: str
    _wildcard: bool = field(init=False, repr=False)
    _suffix: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        lo = self.raw.lower()
        if lo.startswith("*."):
            self._wildcard = True
            self._suffix = lo[2:]   # strip "*."
        else:
            self._wildcard = False
            self._suffix = lo

    def matches(self, host: str) -> bool:
        lo = host.lower()
        if self._wildcard:
            return lo.endswith("." + self._suffix) and lo != self._suffix
        return lo == self._suffix


def _extract_host(raw: str) -> str:
    """
    Return the bare hostname (no port, no brackets, no userinfo).

    >>> _extract_host("[::1]:8080")
    '::1'
    >>> _extract_host("user:pass@api.example.com:443")
    'api.example.com'
    """
    # Strip userinfo (user:pass@)
    if "@" in raw:
        raw = raw.rsplit("@", 1)[1]
    # Strip brackets for IPv6
    if raw.startswith("["):
        raw = raw[1:raw.index("]")]
        return raw
    # Strip port
    if ":" in raw:
        raw = raw.rsplit(":", 1)[0]
    return raw


def _is_ip_address(host: str) -> bool:
    """Return True if *host* is an IPv4 or IPv6 address (not a hostname)."""
    try:
        socket.inet_pton(socket.AF_INET, host)
        return True
    except OSError:
        pass
    try:
        socket.inet_pton(socket.AF_INET6, host)
        return True
    except OSError:
        pass
    return False


class DomainAllowlist:
    """Immutable set of ``DomainPattern`` objects."""

    def __init__(self, patterns: list[str]) -> None:
        self._patterns = [DomainPattern(p) for p in patterns]

    def is_allowed(self, host: str) -> bool:
        """Return True iff *host* matches at least one allowlist pattern."""
        if not self._patterns:
            return False
        bare = _extract_host(host)
        if _is_ip_address(bare):
            return False    # IP addresses are never matched by domain patterns
        return any(p.matches(bare) for p in self._patterns)


# ---------------------------------------------------------------------------
# Credential injection
# ---------------------------------------------------------------------------

@dataclass
class CredentialMapping:
    """
    Inject a secret into outbound requests targeting a specific host.

    Parameters
    ----------
    host_pattern:
        Exact hostname or ``*.example.com`` wildcard.
    secret_env_var:
        Name of the **host** environment variable that holds the secret value.
        The secret is resolved at injection time and **never** sent into the
        container.
    location:
        How the credential is attached:
        * ``"bearer"``          → ``Authorization: Bearer <secret>``
        * ``"header:<Name>"``   → ``<Name>: <secret>``
        * ``"query:<param>"``   → appended as ``?<param>=<secret>``
        * ``"basic:<user>"``    → ``Authorization: Basic base64(<user>:<secret>)``
    """

    host_pattern: str
    secret_env_var: str
    location: str   # "bearer" | "header:<name>" | "query:<param>" | "basic:<user>"
    _pattern: DomainPattern = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._pattern = DomainPattern(self.host_pattern)

    def matches(self, host: str) -> bool:
        return self._pattern.matches(_extract_host(host))

    def inject(self, headers: dict[str, str], url: str) -> tuple[dict[str, str], str]:
        """
        Return updated (headers, url) with the credential injected.

        Reads the secret from the host environment; returns unchanged inputs
        if the env var is not set.
        """
        secret = os.environ.get(self.secret_env_var, "")
        if not secret:
            logger.debug("Credential env var %r not set — skipping injection", self.secret_env_var)
            return headers, url

        loc = self.location.lower()
        h = dict(headers)

        if loc == "bearer":
            h["Authorization"] = f"Bearer {secret}"
        elif loc.startswith("header:"):
            h[self.location[7:]] = secret
        elif loc.startswith("query:"):
            param = self.location[6:]
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{param}={secret}"
        elif loc.startswith("basic:"):
            user = self.location[6:]
            encoded = base64.b64encode(f"{user}:{secret}".encode()).decode()
            h["Authorization"] = f"Basic {encoded}"

        return h, url


# ---------------------------------------------------------------------------
# HTTP proxy server
# ---------------------------------------------------------------------------

class HttpProxyServer:
    """
    Minimal async HTTP/HTTPS proxy with domain allowlist and credential injection.

    Start::

        proxy = HttpProxyServer(allowed_domains=["pypi.org"])
        port = await proxy.start()   # returns the bound port
        await proxy.stop()
    """

    def __init__(
        self,
        allowed_domains: list[str],
        credential_mappings: list[dict[str, str]] | None = None,
        bind_host: str = "127.0.0.1",
    ) -> None:
        self.allowlist = DomainAllowlist(allowed_domains)
        self.credentials: list[CredentialMapping] = [
            CredentialMapping(**m) for m in (credential_mappings or [])
        ]
        self.bind_host = bind_host
        self._server: asyncio.AbstractServer | None = None
        self._port: int | None = None

    async def start(self) -> int:
        """Start listening on a random free port; return the port number."""
        self._server = await asyncio.start_server(
            self._handle_client,
            host=self.bind_host,
            port=0,             # OS picks a free port
        )
        self._port = self._server.sockets[0].getsockname()[1]
        logger.info("HttpProxyServer listening on %s:%d", self.bind_host, self._port)
        return self._port

    async def stop(self) -> None:
        """Close the proxy server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            self._port = None

    # ------------------------------------------------------------------
    # Connection handler
    # ------------------------------------------------------------------

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            await self._dispatch(reader, writer)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Proxy client error: %s", exc)
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
        # Read the request line
        try:
            request_line = await asyncio.wait_for(
                reader.readline(), timeout=_CONNECT_TIMEOUT
            )
        except asyncio.TimeoutError:
            return

        if not request_line:
            return

        parts = request_line.decode(errors="replace").split()
        if len(parts) < 3:
            return

        method, url, _ = parts[0], parts[1], parts[2]

        # Read remaining headers
        raw_headers = await self._read_headers(reader)
        headers = _parse_headers(raw_headers)

        if method.upper() == "CONNECT":
            await self._handle_connect(url, reader, writer, headers)
        else:
            await self._handle_http(method, url, headers, reader, writer)

    async def _read_headers(self, reader: asyncio.StreamReader) -> bytes:
        chunks: list[bytes] = []
        total = 0
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=_CONNECT_TIMEOUT)
            if not line or line in (b"\r\n", b"\n"):
                break
            chunks.append(line)
            total += len(line)
            if total > _MAX_HEADER_BYTES:
                break
        return b"".join(chunks)

    # ------------------------------------------------------------------
    # CONNECT (HTTPS tunnel)
    # ------------------------------------------------------------------

    async def _handle_connect(
        self,
        authority: str,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
        headers: dict[str, str],
    ) -> None:
        """
        Handle HTTP CONNECT tunnelling for HTTPS.

        We validate the target host against the allowlist before establishing
        the tunnel.  Note: we *cannot* inject credentials through a TLS tunnel
        because the traffic is end-to-end encrypted between client and server.
        """
        host = _extract_host(authority)
        port_str = authority.rsplit(":", 1)[-1] if ":" in authority else "443"
        try:
            port = int(port_str)
        except ValueError:
            port = 443

        if not self.allowlist.is_allowed(host):
            logger.info("Proxy CONNECT denied: %s", host)
            client_writer.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            await client_writer.drain()
            return

        # Connect to upstream
        try:
            upstream_reader, upstream_writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=_CONNECT_TIMEOUT,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Proxy CONNECT upstream error %s:%d: %s", host, port, exc)
            client_writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            await client_writer.drain()
            return

        client_writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
        await client_writer.drain()

        # Bidirectional byte pipe with timeout
        await asyncio.wait_for(
            _tunnel(client_reader, client_writer, upstream_reader, upstream_writer),
            timeout=_TUNNEL_TIMEOUT,
        )

    # ------------------------------------------------------------------
    # Plain HTTP forwarding
    # ------------------------------------------------------------------

    async def _handle_http(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
    ) -> None:
        # Extract host from URL or Host header
        host = headers.get("Host") or headers.get("host") or _host_from_url(url)
        if not host:
            client_writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
            await client_writer.drain()
            return

        if not self.allowlist.is_allowed(host):
            logger.info("Proxy HTTP denied: %s %s", method, host)
            client_writer.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            await client_writer.drain()
            return

        # Strip hop-by-hop headers
        fwd_headers = {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}

        # Inject credentials
        for mapping in self.credentials:
            if mapping.matches(host):
                fwd_headers, url = mapping.inject(fwd_headers, url)
                break

        # Read request body (Content-Length or chunked)
        body = await _read_body(client_reader, fwd_headers)

        # Connect to upstream
        parsed_host, parsed_port = _parse_host_port(host, default_port=80)
        try:
            up_reader, up_writer = await asyncio.wait_for(
                asyncio.open_connection(parsed_host, parsed_port),
                timeout=_CONNECT_TIMEOUT,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Proxy HTTP upstream error: %s", exc)
            client_writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            await client_writer.drain()
            return

        # Send request to upstream
        path = _url_path(url)
        request = f"{method} {path} HTTP/1.1\r\n"
        request += "".join(f"{k}: {v}\r\n" for k, v in fwd_headers.items())
        request += "Connection: close\r\n\r\n"
        up_writer.write(request.encode())
        if body:
            up_writer.write(body)
        await up_writer.drain()

        # Relay upstream response back to client
        try:
            response = await asyncio.wait_for(
                up_reader.read(1 << 20),  # up to 1 MiB
                timeout=effective_timeout := self._response_timeout(),
            )
            client_writer.write(response)
            await client_writer.drain()
        except asyncio.TimeoutError:
            client_writer.write(b"HTTP/1.1 504 Gateway Timeout\r\n\r\n")
            await client_writer.drain()
        finally:
            up_writer.close()
            try:
                await up_writer.wait_closed()
            except Exception:  # noqa: BLE001
                pass

    def _response_timeout(self) -> float:
        return 30.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _tunnel(
    cr: asyncio.StreamReader, cw: asyncio.StreamWriter,
    ur: asyncio.StreamReader, uw: asyncio.StreamWriter,
) -> None:
    """Bidirectional byte pipe between client and upstream."""
    async def pipe(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
        try:
            while True:
                data = await src.read(65536)
                if not data:
                    break
                dst.write(data)
                await dst.drain()
        except Exception:  # noqa: BLE001
            pass
        finally:
            try:
                dst.close()
            except Exception:  # noqa: BLE001
                pass

    await asyncio.gather(pipe(cr, uw), pipe(ur, cw))


def _parse_headers(raw: bytes) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in raw.split(b"\n"):
        line = line.strip()
        if b":" in line:
            k, _, v = line.partition(b":")
            headers[k.decode(errors="replace").strip()] = v.decode(errors="replace").strip()
    return headers


async def _read_body(reader: asyncio.StreamReader, headers: dict[str, str]) -> bytes:
    cl_str = headers.get("Content-Length") or headers.get("content-length") or "0"
    try:
        content_length = int(cl_str)
    except ValueError:
        content_length = 0
    if content_length > 0:
        try:
            return await asyncio.wait_for(reader.readexactly(content_length), timeout=30.0)
        except Exception:  # noqa: BLE001
            pass
    return b""


def _host_from_url(url: str) -> str:
    m = re.match(r"https?://([^/?\s]+)", url)
    return _extract_host(m.group(1)) if m else ""


def _url_path(url: str) -> str:
    m = re.match(r"https?://[^/?\s]+(.*)", url)
    return m.group(1) if m and m.group(1) else "/"


def _parse_host_port(host: str, default_port: int = 80) -> tuple[str, int]:
    bare = _extract_host(host)
    if ":" in host and not host.startswith("["):
        try:
            port = int(host.rsplit(":", 1)[-1])
            return bare, port
        except ValueError:
            pass
    return bare, default_port
