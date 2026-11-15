"""Turns media the user sent into text the chat LLM can work with — the same for every
chat platform: voice notes are transcribed, videos are described by the video tier, and a
document (a file sent as an attachment) is read the way a mail attachment is — a PDF through
`PdfExtractor`, a text file decoded — or, when it cannot be, named with the reason, so the
assistant can say so instead of leaving the message unanswered. A picture sent as a file is
the exception: `picture_as_jpeg` makes it a photo for the vision tier."""
import codecs
from datetime import datetime

from common.audio_log import save_audio_file
from common.logging import error, info, trace
from common.Paths import LOG_DIR
from interfaces.chat.ChatTypes import Attachment

# A document's text is capped where a mail attachment's PDF text is.
DOCUMENT_TEXT_CHARS = 100_000
TEXT_EXTENSIONS = {"txt", "md", "markdown", "csv", "tsv", "json", "yaml", "yml", "xml", "html", "htm", "log", "ini",
                   "toml", "cfg", "conf", "tex", "srt", "vtt", "ics", "eml"}
PICTURE_EXTENSIONS = {"jpg", "jpeg", "png", "heic", "heif", "webp", "gif", "tif", "tiff", "bmp"}
# A picture is sent as a file for its detail (a drawing, a scanned page), so it keeps more of
# it than a platform's own compressed photo does.
PICTURE_MAX_DIMENSION = 2048
# MIME types a sender's client puts on a file it did not recognise; the content decides then.
GENERIC_MIME_TYPES = {"", "application/octet-stream", "binary/octet-stream"}
# A subfolder: the clips directly in log/audio are the mics', and the voice discovery reads those.
VOICE_NOTE_DIR = LOG_DIR / "audio" / "voice_notes"
VOICE_NOTE_EXTENSIONS = {"audio/ogg": "ogg", "audio/opus": "ogg", "audio/mpeg": "mp3", "audio/mp4": "m4a",
                         "audio/x-m4a": "m4a", "audio/aac": "aac", "audio/wav": "wav", "audio/x-wav": "wav",
                         "audio/x-caf": "caf", "audio/amr": "amr", "audio/webm": "webm"}
NO_TRANSCRIPT_NAME = "no_transcript"


class UnreadableDocument(Exception):
    """The file has no text to give: why, in a few words."""


def transcribe_voice(data: bytes, mime_type: str = "audio/ogg") -> str:
    """Speech-to-text for a voice note; "" when nothing came out. The note is kept in the audio
    log as it arrived, named after what was heard — the failed ones are why it is kept."""
    from interfaces.telegram.GoogleCloudSTT import GoogleCloudSTT
    transcript = GoogleCloudSTT.get_instance().transcribe(data) or ""
    saved = _log_voice_note(data, mime_type, transcript)
    if transcript:
        trace(f"Voice note transcribed ({len(data)} bytes, {saved}): {transcript!r}")
    else:
        info(f"Voice note ({len(data)} bytes) produced no transcript, the audio is in {saved}.")
    return transcript


def _log_voice_note(data: bytes, mime_type: str, transcript: str) -> str:
    """Where the note was written, or the reason it was not — a full disk costs no transcript."""
    extension = VOICE_NOTE_EXTENSIONS.get((mime_type or "").split(";")[0].strip().lower(), "ogg")
    name = f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{transcript or NO_TRANSCRIPT_NAME}"
    try:
        return save_audio_file(str(VOICE_NOTE_DIR), name, data, extension)
    except OSError as e:
        error(f"Voice note could not be saved to {VOICE_NOTE_DIR}: {e}")
        return "not saved"


def describe_video(data: bytes, mime_type: str, context: str) -> str:
    """A description of the clip from the video tier; "" when it could not be analysed."""
    from assistant.tools.VideoAnalyzer import video_analyzer
    description = video_analyzer().describe_bytes(data, mime_type=mime_type, context=context, fallback="") or ""
    if not description:
        info(f"Video ({len(data)} bytes, {mime_type}) could not be analysed.")
    return description


def is_picture(name: str, mime_type: str) -> bool:
    """Whether a file sent as a document is a picture the vision tier can be shown (an SVG is not)."""
    mime_type = (mime_type or "").lower()
    if mime_type.startswith("image/"):
        return "svg" not in mime_type
    return mime_type in GENERIC_MIME_TYPES and _extension(name) in PICTURE_EXTENSIONS


def picture_as_jpeg(data: bytes):
    """A picture file re-encoded as a JPEG the vision tier reads (HEIC included), its long side
    at most PICTURE_MAX_DIMENSION; None when it cannot be decoded."""
    from assistant.tools.ImageAnalyzer import resize_to_jpeg
    try:
        jpeg = resize_to_jpeg(data, PICTURE_MAX_DIMENSION)
    except Exception as e:
        error(f"Picture file ({len(data)} bytes) could not be converted: {type(e).__name__}: {e}")
        return None
    if jpeg is None:
        info(f"Picture file ({len(data)} bytes) could not be decoded.")
    else:
        trace(f"Picture file converted for vision ({len(data)} → {len(jpeg)} bytes)")
    return jpeg


def _extension(name: str) -> str:
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def document_text(att: Attachment, caption: str = "") -> tuple[str, str]:
    """(body, headline) for a document: the body is what the assistant reads — a header naming
    the file, its type and size, the caption, then the file's text, or instead the reason there
    is none (not downloaded, no reader for its type, a scanned PDF) — and the headline is the
    body without the text, for the log and the transcript."""
    name = att.name or "unnamed"
    size = len(att.data) or att.size
    kind = att.mime_type or "unknown type"
    headline = f"[User sent a document: {name} ({kind}, {describe_size(size)})]"
    if caption:
        headline += f"\nCaption: {caption}"
    if att.undelivered:
        info(f"Document {name} ({describe_size(size)}) was not downloaded: {att.undelivered}; the assistant is told")
        return f"{headline}\nThe file was not downloaded: {att.undelivered}. Its contents are unknown.", headline
    try:
        text = extract_document_text(name, att.mime_type, att.data)
    except UnreadableDocument as e:
        info(f"Document {name} ({kind}, {describe_size(size)}) could not be read: {e}; the assistant is told")
        return f"{headline}\nThe file could not be read ({e}), so its contents are unknown.", headline
    info(f"Document {name} ({kind}, {describe_size(size)}): {len(text)} chars of text read")
    return f"{headline}\n{text}", headline


def extract_document_text(name: str, mime_type: str, data: bytes) -> str:
    """The text of a file: a PDF through the mail attachments' extractor, a text file decoded
    (UTF-8, else Latin-1) and cut at DOCUMENT_TEXT_CHARS; raises UnreadableDocument for the
    rest, and for a PDF without a text layer."""
    from assistant.tools import PdfExtractor
    mime_type = (mime_type or "").lower()
    ext = _extension(name)
    if is_picture(name, mime_type):
        # A transport shows a picture to the vision tier; one that ends up here could not be decoded.
        raise UnreadableDocument("the picture could not be decoded")
    if mime_type == "application/pdf" or ext == "pdf":
        text = PdfExtractor.extract_text(data)
        if text.startswith("ERROR:"):
            raise UnreadableDocument(text[len("ERROR:"):].strip().rstrip("."))
        return text
    # A file without an extension under a generic type (README, a pasted note) is judged by
    # its content; one with an extension no reader claims is not — a .xlsx is a zip, and a
    # zip's first bytes pass for UTF-8.
    if (mime_type.startswith("text/") or ext in TEXT_EXTENSIONS
            or (mime_type in GENERIC_MIME_TYPES and not ext and _looks_like_text(data))):
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = data.decode("latin-1")
        if not text.strip():
            raise UnreadableDocument("the file is empty")
        if len(text) > DOCUMENT_TEXT_CHARS:
            info(f"Document {name}: {len(text)} chars of text, cut to {DOCUMENT_TEXT_CHARS}")
            text = text[:DOCUMENT_TEXT_CHARS] + f"\n\n[NOTE: Text truncated to {DOCUMENT_TEXT_CHARS} characters.]"
        return text
    raise UnreadableDocument(f"no reader for {mime_type or ('.' + ext if ext else 'files of that kind')}")


def _looks_like_text(data: bytes) -> bool:
    """Judged on the first 8 KB: no NUL byte, and UTF-8 throughout — a multi-byte character
    the head cuts through counts as text when the file goes on past the head."""
    head = data[:8192]
    if not head or b"\x00" in head:
        return False
    try:
        codecs.getincrementaldecoder("utf-8")().decode(head, final=len(data) <= len(head))
        return True
    except UnicodeDecodeError:
        return False


def describe_size(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / 1024 / 1024:.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size} bytes"
