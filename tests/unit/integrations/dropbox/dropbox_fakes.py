"""Stand-ins for `requests` answering the shapes Dropbox's token endpoint and API v2 send,
written from the API reference's examples with a fictional household's files, and builders for
the smallest PDF, .docx and .pptx files that carry given text (no document library is pinned)."""
import io
import json as jsonlib
import time
import zipfile
from dataclasses import dataclass, field
from typing import Optional

from integrations.dropbox.DropboxAuth import TOKEN_URL
from integrations.dropbox.DropboxClient import API_URL, CONTENT_URL

MODIFIED = "2024-03-02T10:00:00Z"
HASH = "a" * 64


class FakeResponse:
    def __init__(self, status: int, body=None, raw: Optional[bytes] = None, headers: Optional[dict] = None):
        self.status_code = status
        self.content = raw if raw is not None else (jsonlib.dumps(body).encode() if body is not None else b"")
        self.text = self.content.decode("utf-8", errors="replace")
        self.headers = headers or {}

    def json(self):
        return jsonlib.loads(self.content)


@dataclass
class Call:
    method: str
    path: str
    headers: dict
    body: Optional[dict]        # the JSON body of an RPC call, or the Dropbox-API-Arg of a content call
    data: Optional[dict] = None  # the form of a token request


@dataclass
class FakeDropbox:
    """`route(path, body)` answers an API request (path relative to the api or content host);
    `route_answers` scripts a path's answers in turn; `token_answers` is what the token endpoint
    answers next, in order. Every request is kept in `calls`."""
    routes: dict = field(default_factory=dict)
    token_answers: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    transport_errors: list = field(default_factory=list)

    def route(self, path: str, body=None, status: int = 200, raw: Optional[bytes] = None, headers: Optional[dict] = None):
        self.routes[path] = FakeResponse(status, body, raw, headers)

    def route_answers(self, path: str, *answers: tuple):
        """(status, body) answers in turn; the last one stands for every later call."""
        self.routes[path] = [FakeResponse(status, body) for status, body in answers]

    def answer_token(self, body: dict, status: int = 200):
        self.token_answers.append(FakeResponse(status, body))

    # requests.Session.request, as the client calls it
    def request(self, method, url, headers=None, data=None, timeout=None):
        if self.transport_errors:
            raise self.transport_errors.pop(0)
        path = url.removeprefix(API_URL).removeprefix(CONTENT_URL)
        headers = dict(headers or {})
        if "Dropbox-API-Arg" in headers:
            body = jsonlib.loads(headers["Dropbox-API-Arg"])
        else:
            body = jsonlib.loads(data) if data else None
        self.calls.append(Call(method, path, headers, body))
        answer = self.routes.get(path)
        if isinstance(answer, list):
            answer = answer.pop(0) if len(answer) > 1 else answer[0]
        return answer if answer is not None else FakeResponse(409, {"error_summary": f"path/not_found/..", "error": {".tag": "path"}})

    # requests.post, as the auth calls it for the token endpoint
    def post(self, url, data=None, headers=None, timeout=None):
        self.calls.append(Call("POST", url, dict(headers or {}), None, dict(data or {})))
        assert url == TOKEN_URL, f"posted to {url}"
        if not self.token_answers:
            return FakeResponse(400, {"error": "invalid_request", "error_description": "no answer scripted"})
        return self.token_answers.pop(0)

    def sent(self, path: str) -> list:
        return [c for c in self.calls if c.path == path]


def token_grant(access_token="at.new", refresh_token="rt.new", expires_in=14400) -> dict:
    body = {"access_token": access_token, "token_type": "bearer", "expires_in": expires_in,
            "scope": "account_info.read files.content.read files.metadata.read", "uid": "12345",
            "account_id": "dbid:AAH4f99T0taONIb-OurWxbNQ6ywGRopQngc"}
    if refresh_token:
        body["refresh_token"] = refresh_token
    return body


def token_file(expires_in=10000, refresh_token="rt.old", access_token="at.old") -> dict:
    return {"refresh_token": refresh_token, "access_token": access_token, "expiry": time.time() + expires_in}


def file_metadata(path: str, size: int = 1200, modified: str = MODIFIED, content_hash: str = HASH) -> dict:
    name = path.rsplit("/", 1)[-1]
    return {".tag": "file", "name": name, "path_lower": path.lower(), "path_display": path, "id": "id:" + name,
            "client_modified": modified, "server_modified": modified, "rev": "0123456789abcdef", "size": size,
            "is_downloadable": True, "content_hash": content_hash}


def folder_metadata(path: str) -> dict:
    name = path.rsplit("/", 1)[-1]
    return {".tag": "folder", "name": name, "path_lower": path.lower(), "path_display": path, "id": "id:" + name}


def search_match(metadata: dict, match_type: str = "filename", highlight: str = "") -> dict:
    match = {"metadata": {".tag": "metadata", "metadata": metadata}, "match_type": {".tag": match_type}}
    if highlight:
        match["highlight_spans"] = [{"highlight_str": "the ", "is_highlighted": False},
                                    {"highlight_str": highlight, "is_highlighted": True},
                                    {"highlight_str": " of two months", "is_highlighted": False}]
    return match


def account(account_type="basic", email="marcus@example.com", display_name="Marcus") -> dict:
    return {"account_id": "dbid:AAH4f99T0taONIb-OurWxbNQ6ywGRopQngc", "name": {"display_name": display_name},
            "email": email, "account_type": {".tag": account_type}}


class StubAuth:
    def __init__(self, token="at.stub", signed_in=True, again=False, missing=()):
        self._token = token
        self._signed_in = signed_in
        self._again = again
        self._missing = list(missing)
        self.refresh_error = "the Dropbox sign-in is no longer valid" if again else ""

    def signed_in(self) -> bool:
        return self._signed_in

    def needs_sign_in_again(self) -> bool:
        return self._again

    def missing_scopes(self) -> list:
        return self._missing

    def token(self) -> str:
        from integrations.dropbox.DropboxAuth import NotSignedIn, not_signed_in_hint
        if not self._signed_in:
            raise NotSignedIn(not_signed_in_hint())
        return self._token


# --- document builders -------------------------------------------------------------------------

def make_rtf(lines: list[str], code_page: int = 1252, cocoa: bool = True) -> bytes:
    """An RTF as TextEdit writes it (`cocoa`: the font and colour tables, a `{\\*\\…}` group, a
    backslash-newline per paragraph) or as Word does (`\\par`), the non-ASCII letters as
    `\\'xx` escapes of the code page — which is what the reader has to undo."""
    def escaped(line: str) -> bytes:
        out = bytearray()
        for char in line:
            byte = char.encode(f"cp{code_page}", errors="strict")
            out.extend(byte if byte.isascii() else b"".join(b"\\'%02x" % b for b in byte))
        return bytes(out)
    head = (b"{\\rtf1\\ansi\\ansicpg%d\\cocoartf2822\n\\cocoatextscaling0{\\fonttbl\\f0\\fswiss\\fcharset0 Helvetica;}\n"
            b"{\\colortbl;\\red255\\green255\\blue255;}\n{\\*\\expandedcolortbl;;}\n"
            b"\\paperw11900\\margl1440\\pard\\tx566\\pardirnatural\\partightenfactor0\n\n\\f0\\fs24 \\cf0 " % code_page
            if cocoa else b"{\\rtf1\\ansi\\ansicpg%d\\deff0{\\fonttbl{\\f0 Calibri;}}\\pard " % code_page)
    separator = b"\\\n" if cocoa else b"\\par "
    return head + separator.join(escaped(line) for line in lines) + separator + b"}"


def make_pdf(pages: list[str]) -> bytes:
    """A PDF with one page per text (an empty text is a page without a text layer), built by
    hand with a correct xref table so pdfplumber reads it as a normal file."""
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids = []
    pages_id = len(objects) + 2 * len(pages) + 1
    for text in pages:
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").encode("latin-1", "replace")
        stream = b"BT /F1 12 Tf 72 720 Td (" + escaped + b") Tj ET" if text else b""
        content = add(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
        page = add(b"<< /Type /Page /Parent " + str(pages_id).encode() + b" 0 R /MediaBox [0 0 612 792] "
                   b"/Resources << /Font << /F1 " + str(font).encode() + b" 0 R >> >> /Contents " + str(content).encode() + b" 0 R >>")
        page_ids.append(page)
    kids = b" ".join(str(p).encode() + b" 0 R" for p in page_ids)
    assert add(b"<< /Type /Pages /Kids [" + kids + b"] /Count " + str(len(page_ids)).encode() + b" >>") == pages_id
    catalog = add(b"<< /Type /Catalog /Pages " + str(pages_id).encode() + b" 0 R >>")

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(out.tell())
        out.write(str(number).encode() + b" 0 obj\n" + body + b"\nendobj\n")
    xref = out.tell()
    out.write(b"xref\n0 " + str(len(objects) + 1).encode() + b"\n0000000000 65535 f \n")
    for offset in offsets:
        out.write(f"{offset:010d} 00000 n \n".encode())
    out.write(b"trailer\n<< /Size " + str(len(objects) + 1).encode() + b" /Root " + str(catalog).encode() + b" 0 R >>\n"
              b"startxref\n" + str(xref).encode() + b"\n%%EOF\n")
    return out.getvalue()


WORD_XMLNS = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
DRAWING_XMLNS = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"'


def make_docx(paragraphs: list[str], table: Optional[list[list[str]]] = None) -> bytes:
    """A .docx with the paragraphs, then a table of the rows, each cell one paragraph."""
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    if table:
        rows = "".join("<w:tr>" + "".join(f"<w:tc><w:p><w:r><w:t>{cell}</w:t></w:r></w:p></w:tc>" for cell in row) + "</w:tr>"
                       for row in table)
        body += f"<w:tbl>{rows}</w:tbl>"
    document = f'<?xml version="1.0" encoding="UTF-8"?><w:document {WORD_XMLNS}><w:body>{body}</w:body></w:document>'
    return _zip({"[Content_Types].xml": "<Types/>", "word/document.xml": document})


def make_pptx(slides: list[list[str]]) -> bytes:
    """A .pptx with one slide per list of paragraphs, numbered as given."""
    files = {"[Content_Types].xml": "<Types/>"}
    for number, paragraphs in enumerate(slides, 1):
        runs = "".join(f"<a:p><a:r><a:t>{p}</a:t></a:r></a:p>" for p in paragraphs)
        files[f"ppt/slides/slide{number}.xml"] = (f'<?xml version="1.0" encoding="UTF-8"?><p:sld {DRAWING_XMLNS}>'
                                                  f"<p:cSld><p:spTree><p:sp><p:txBody>{runs}</p:txBody></p:sp></p:spTree></p:cSld></p:sld>")
    return _zip(files)


def _zip(files: dict) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return out.getvalue()
