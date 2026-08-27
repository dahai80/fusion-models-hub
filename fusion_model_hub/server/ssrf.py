import ipaddress
import logging
import re
import socket
from urllib.parse import urlparse

from fastapi import HTTPException

logger = logging.getLogger(__name__)

_IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

# E-S14: prior blocklist missed 0.0.0.0, ip6-localhost / IPv6-mapped IPv4,
# trailing-dot variants, and integer/hex/octal IP encodings. The string
# blocklist is a fast first pass; the network/IP checks below are the real gate.
_BLOCKED_HOSTS = {
    "localhost",
    "localhost.",
    "ip6-localhost",
    "ip6-localhost.",
    "ip6-loopback",
    "metadata.google.internal",
    "metadata.google.internal.",
}
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
    # E-S14: IPv6-mapped IPv4 (e.g. ::ffff:169.254.169.254) bypassed the IPv4
    # checks when the host was an IPv6 literal. ip_address handles the mapping
    # but the explicit network catches edge cases cleanly.
    ipaddress.ip_network("::ffff:0:0/96"),
]

_BLOCKED_DETAIL = "URL cannot point to internal network"

# E-S14: integer/decimal IP encodings like http://2130706433/ (127.0.0.1) and
# hex http://0x7f000001/ slip past string-prefix checks. Detect a pure-numeric
# (non-dotted) host and force it through ip_address, which python's ipaddress
# does NOT parse as a packed-integer — so decode it explicitly.
_DECIMAL_INT_RE = re.compile(r"^\d+$")
_HEX_OCTET_RE = re.compile(r"^0x[0-9a-fA-F]+$")


def _reject(msg: str = _BLOCKED_DETAIL) -> None:
    raise HTTPException(status_code=400, detail=msg)


def _check_ip(ip: _IPAddress) -> None:
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
        _reject()
    for net in _BLOCKED_NETWORKS:
        if ip in net:
            _reject()


def _decode_numeric_host(hostname: str) -> _IPAddress | None:
    # E-S14: http://2130706433/ == 127.0.0.1; http://0x7f000001/ likewise.
    # Decimal: a single 32-bit unsigned int packed big-endian.
    if _DECIMAL_INT_RE.match(hostname):
        try:
            val = int(hostname)
            if 0 <= val <= 0xFFFFFFFF:
                return ipaddress.IPv4Address(val)
        except (ValueError, ipaddress.AddressValueError):
            return None
    if _HEX_OCTET_RE.match(hostname):
        try:
            val = int(hostname, 16)
            if 0 <= val <= 0xFFFFFFFF:
                return ipaddress.IPv4Address(val)
        except (ValueError, ipaddress.AddressValueError):
            return None
    return None


def _resolve_and_check(hostname: str) -> None:
    # E-S14: DNS rebinding — validation resolves the hostname to a public IP,
    # but the later HTTP fetch re-resolves and the attacker has flipped the A
    # record to 127.0.0.1. We cannot pin the resolved IP across the fetch
    # without reworking every caller's httpx client, so the defense here is to
    # resolve now and reject if ANY resolved address is internal. This closes
    # the "validation-time public, fetch-time private" gap for the common case
    # (single A record). Callers MUST also fetch with follow_redirects=False
    # (see validate_external_url docstring) so a 302 to an internal URL is not
    # silently followed.
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        # Unresolvable at validation time — let the fetch fail naturally rather
        # than guessing; a typo'd external host is a client error, not SSRF.
        return
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        _check_ip(ip)


def validate_external_url(url_str: str, *, allow_https_only: bool = False) -> None:
    # E-S14: callers that then fetch the URL MUST use follow_redirects=False,
    # so a public URL that 302-redirects to an internal address cannot be
    # followed. This only validates the provided URL plus its resolved IPs at
    # validation time; it cannot control redirect behavior of the fetch itself.
    parsed = urlparse(url_str)
    if allow_https_only:
        if parsed.scheme != "https":
            raise HTTPException(status_code=400, detail="URL must use https scheme")
    else:
        if parsed.scheme not in ("http", "https"):
            raise HTTPException(status_code=400, detail="URL must use http or https scheme")
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise HTTPException(status_code=400, detail="URL must include a hostname")
    # Strip a trailing dot (FQDN form) before blocklist/numeric checks.
    bare = hostname.rstrip(".")
    if hostname in _BLOCKED_HOSTS or bare in _BLOCKED_HOSTS:
        _reject()
    # Direct IP literal?
    try:
        ip = ipaddress.ip_address(bare)
    except ValueError:
        ip = _decode_numeric_host(bare)
    if ip is not None:
        _check_ip(ip)
        # Literal public IP — no DNS rebinding risk, no further resolution.
        logger.info("URL passed SSRF check (IP literal): %s", bare)
        return
    # Hostname — also guard the legacy dotted-prefix fast path for the
    # 172.16-31 / 10 / 192.168 / 169.254 ranges in case DNS resolves externally
    # but the textual form is already internal (belt + braces).
    octets = bare.split(".")
    if len(octets) == 4 and octets[0] == "172" and octets[1].isdigit() and 16 <= int(octets[1]) <= 31:
        _reject()
    if bare.startswith(("10.", "192.168.", "169.254.")):
        _reject()
    # Resolve and reject if any resolved address is internal (DNS-rebinding).
    _resolve_and_check(bare)
    logger.info("URL passed SSRF check: %s", url_str)
