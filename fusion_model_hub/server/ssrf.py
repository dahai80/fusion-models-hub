import ipaddress
import logging
from urllib.parse import urlparse

from fastapi import HTTPException

logger = logging.getLogger(__name__)

_BLOCKED_HOSTS = {"localhost", "metadata.google.internal"}
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
]


def validate_external_url(url_str: str, *, allow_https_only: bool = False) -> None:
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
    if hostname in _BLOCKED_HOSTS:
        raise HTTPException(status_code=400, detail="URL cannot point to internal network")
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        ip = None
    if ip is not None:
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise HTTPException(status_code=400, detail="URL cannot point to internal network")
        for net in _BLOCKED_NETWORKS:
            if ip in net:
                raise HTTPException(status_code=400, detail="URL cannot point to internal network")
    else:
        octets = hostname.split(".")
        if len(octets) == 4 and octets[0] == "172" and octets[1].isdigit() and 16 <= int(octets[1]) <= 31:
            raise HTTPException(status_code=400, detail="URL cannot point to internal network")
        if hostname.startswith(("10.", "192.168.", "169.254.")):
            raise HTTPException(status_code=400, detail="URL cannot point to internal network")
    logger.info("URL passed SSRF check: %s", url_str)
