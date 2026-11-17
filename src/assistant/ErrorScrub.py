"""What an error message loses before the telemetry reports it: the things an exception text
quotes that say who the household is or would let somebody in.

The vault redaction (`redact_for_storage`) knows only the values the owner stored; an error
quotes whatever it failed on — the address a mail could not be sent to, a stranger's phone
number, a URL with a bot token in its path, a file under the home directory. Error events go
out for as long as the install runs, so these go by pattern. Short numbers (a port, a status
code, a line) stay: they are what makes an error readable."""
import re
from pathlib import Path

_HOME = str(Path.home())
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# 24+ characters of an id's alphabet with a digit among them: an API key, a token, a uuid
_TOKEN = re.compile(r"(?<![\w-])(?=[\w-]*\d)[\w-]{24,}(?![\w-])")
# 7+ digits, single spaces or dashes between them: a phone number, a personal id
_NUMBER = re.compile(r"\+?\d(?:[ -]?\d){6,}")


def scrub(text: str) -> str:
    text = text.replace(_HOME, "~")
    text = _EMAIL.sub("<email>", text)
    text = _TOKEN.sub("<token>", text)
    return _NUMBER.sub("<number>", text)
