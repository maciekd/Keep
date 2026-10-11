"""Blanks what looks like a credential out of a Claude Code transcript before it is stored.

A session's prompts and commands are the owner's own text, and now and then that text holds
an API key, a bearer token or a password typed in to get something done. The history is read
back into every prompt, so none of that may reach it: each pattern below is swapped for
`[secret]` before a row is written. The patterns are the well-known key shapes (OpenAI,
Anthropic, GitHub, Slack, AWS, Google, JWTs, PEM blocks), a `password=` / `token:` style
assignment, and a conservative catch-all for a long opaque token — mixed case with digits
and no slash, so a path, a git sha or a session id stays. Pure functions, no state."""
import re

REDACTED = "[secret]"

_ASSIGNMENT_KEYS = r"(?:password|passwd|pwd|passphrase|secret|token|api[_-]?key|apikey|auth|authorization|bearer)"

PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL),
    re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_\-]{16,}"),                 # OpenAI / Anthropic keys
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}"),         # GitHub tokens
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),                    # Slack
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                               # AWS access key id
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),                         # Google API key
    re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b"),   # JWT
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9_\-.=+/]{16,}"),
    # key=value / key: value / "key": "value" — the value is replaced, the key stays so the row still says what was set
    re.compile(rf"(?i)(\b{_ASSIGNMENT_KEYS}\b[\"']?\s*[:=]\s*[\"']?)([^\s\"',;]{{4,}})"),
    # a long opaque token: letters of both cases and digits, no slash (paths) — a 40-hex sha
    # or a session id has one case only and stays
    re.compile(r"(?<![A-Za-z0-9/])(?=[A-Za-z0-9_\-+=]{40,}(?![A-Za-z0-9/]))"
               r"(?=[A-Za-z0-9_\-+=]*[a-z])(?=[A-Za-z0-9_\-+=]*[A-Z])(?=[A-Za-z0-9_\-+=]*\d)[A-Za-z0-9_\-+=]{40,}"),
]

# a value that is plainly not a secret: the example config's YOUR_…, a <placeholder>, xxx, none
_PLACEHOLDER = re.compile(r"(?i)^(?:<|\*+$|x+$|\.\.\.|(?:your|my|the|an?|none|null|redacted|change_?me)(?![a-z0-9]))")


def scrub(text: str) -> str:
    """`text` with every credential-looking run replaced by `[secret]`."""
    if not text:
        return text
    for pattern in PATTERNS:
        if pattern.groups:
            text = pattern.sub(lambda m: m.group(1) + (m.group(2) if _PLACEHOLDER.match(m.group(2)) else REDACTED), text)
        else:
            text = pattern.sub(REDACTED, text)
    return text


def looks_secret(text: str) -> bool:
    return scrub(text) != text
