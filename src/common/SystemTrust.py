"""Makes Python verify TLS against the system trust store (the macOS keychain) instead of only a bundled CA list.

requests, httpx and aiohttp each ship certifi's list of public CAs and trust nothing else. On
a network that inspects HTTPS (a corporate proxy, an antivirus' "HTTPS scanning", a filtering
VPN) every certificate is re-signed by a root the machine's keychain trusts and certifi has
never heard of: the browser works and every call from here fails with CERTIFICATE_VERIFY_FAILED.
truststore swaps `ssl.SSLContext` for one that asks the system, so what the Mac trusts the
server trusts; a CA bundle a library loads on top (certifi) stays trusted as well.

Switching verification off is not enough once that swap is in place: truststore evaluates the
chain with the system's policy whatever `verify_mode` says and forgives only an unknown root
and an expired certificate. A device on the LAN whose certificate the policy rejects outright
(a Hue bridge: "does not meet current standards" on macOS) is unreachable with `verify=False`
alone — `unverified_context()` is the way to talk to one.

`install()` must run before any library builds an SSL context — aiohttp builds one when it is
imported — which is why floxy_ai_server.py calls it ahead of its other imports. Not covered,
because they do not go through Python's ssl module: curl_cffi (WebAccess' last-resort fetch)
and gRPC (google-cloud-speech)."""
import ssl

# The interpreter's own class, the one `install()` swaps out. Looked up by module so the
# result is the same whether or not the swap has happened yet when this module is imported.
_OPENSSL_CONTEXT = next(cls for cls in ssl.SSLContext.__mro__ if cls.__module__ == "ssl")
_C_CONTEXT = _OPENSSL_CONTEXT.__mro__[1]


class _PlainOpenSSLContext(_OPENSSL_CONTEXT):
    """The stock context, usable after install(): ssl.py's setters reach their C base through
    the module name `SSLContext`, which the swap points at truststore's class — the stock
    class's own `verify_mode = …` then recurses forever. These setters go to the C base directly."""


for _name in ("verify_mode", "verify_flags", "options", "minimum_version", "maximum_version"):
    setattr(_PlainOpenSSLContext, _name, property(getattr(_OPENSSL_CONTEXT, _name).fget, getattr(_C_CONTEXT, _name).__set__))


def install() -> str:
    """Switch this process to the system trust store. Returns what happened, for the caller
    to log once logging is up; a missing package is left to RequirementsChecker to report."""
    try:
        import truststore
    except ImportError:
        return "TLS trust: truststore is not installed, verifying against the bundled CA list only"
    truststore.inject_into_ssl()
    return "TLS trust: verifying against the system trust store (the keychain), bundled CA list on top"


def unverified_context() -> ssl.SSLContext:
    """A TLS context that verifies nothing and never asks the system trust store — for a device
    on the LAN reached by its address that serves its own certificate (a Hue bridge). Hand it to
    the HTTP library as its `ssl_context`; `verify=False` on its own does not get past
    truststore's policy check."""
    context = _PlainOpenSSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context
