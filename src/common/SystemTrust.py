"""Makes Python verify TLS against the system trust store (the macOS keychain) instead of only a bundled CA list.

requests, httpx and aiohttp each ship certifi's list of public CAs and trust nothing else. On
a network that inspects HTTPS (a corporate proxy, an antivirus' "HTTPS scanning", a filtering
VPN) every certificate is re-signed by a root the machine's keychain trusts and certifi has
never heard of: the browser works and every call from here fails with CERTIFICATE_VERIFY_FAILED.
truststore swaps `ssl.SSLContext` for one that asks the system, so what the Mac trusts the
server trusts; a CA bundle a library loads on top (certifi) stays trusted as well, and a
context that switches verification off stays off.

`install()` must run before any library builds an SSL context — aiohttp builds one when it is
imported — which is why floxy_ai_server.py calls it ahead of its other imports. Not covered,
because they do not go through Python's ssl module: curl_cffi (WebAccess' last-resort fetch)
and gRPC (google-cloud-speech)."""


def install() -> str:
    """Switch this process to the system trust store. Returns what happened, for the caller
    to log once logging is up; a missing package is left to RequirementsChecker to report."""
    try:
        import truststore
    except ImportError:
        return "TLS trust: truststore is not installed, verifying against the bundled CA list only"
    truststore.inject_into_ssl()
    return "TLS trust: verifying against the system trust store (the keychain), bundled CA list on top"
