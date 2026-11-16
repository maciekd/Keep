"""MediaIntake.transcribe_voice: a voice note is kept in the audio log whatever the STT made of it.
MediaIntake.document_text: what the assistant reads for a file a chat user sent — the PDF
extractor the mail attachments use, a text file decoded, and the reasons a file gives no text."""
from unittest.mock import patch

import pytest

from interfaces.chat import MediaIntake
from interfaces.chat.ChatTypes import KIND_DOCUMENT, Attachment
from interfaces.chat.MediaIntake import UnreadableDocument, describe_size, document_text, extract_document_text

PDF_EXTRACTOR = "assistant.tools.PdfExtractor.extract_text"


def document(data: bytes, mime_type: str, name: str, **kw) -> Attachment:
    return Attachment(data, mime_type, KIND_DOCUMENT, name=name, **kw)


@pytest.mark.unit
class TestExtract:
    def test_a_pdf_goes_through_the_mail_attachments_extractor(self):
        with patch(PDF_EXTRACTOR, return_value="--- Page 1 ---\nTotal: 120 SEK") as extract:
            text = extract_document_text("invoice.pdf", "application/pdf", b"%PDF-1.4")
        extract.assert_called_once_with(b"%PDF-1.4")
        assert "Total: 120 SEK" in text

    def test_a_pdf_is_known_by_its_extension_when_the_sender_gave_no_type(self):
        with patch(PDF_EXTRACTOR, return_value="text") as extract:
            extract_document_text("scan.PDF", "application/octet-stream", b"%PDF-1.4")
        extract.assert_called_once()

    def test_a_pdf_without_a_text_layer_is_unreadable_with_the_extractors_reason(self):
        with patch(PDF_EXTRACTOR, return_value="ERROR: PDF contains no extractable text (might be scanned/image-based)."):
            with pytest.raises(UnreadableDocument, match="no extractable text"):
                extract_document_text("scan.pdf", "application/pdf", b"%PDF-1.4")

    @pytest.mark.parametrize("name,mime_type", [
        ("notes.txt", "text/plain"),
        ("notes.md", ""),                        # the extension alone decides
        ("data.csv", "text/csv"),
        ("README", "application/octet-stream"),  # neither: the content is sniffed as text
    ])
    def test_text_files_are_decoded(self, name, mime_type):
        assert extract_document_text(name, mime_type, "Hallå Marcus\n".encode("utf-8")) == "Hallå Marcus\n"

    def test_a_byte_order_mark_is_dropped_and_latin1_is_the_fallback(self):
        assert extract_document_text("a.txt", "text/plain", b"\xef\xbb\xbfhello") == "hello"
        assert extract_document_text("b.txt", "text/plain", "smörgås".encode("latin-1")) == "smörgås"

    def test_binary_content_under_a_generic_type_is_unreadable(self):
        with pytest.raises(UnreadableDocument):
            extract_document_text("blob", "application/octet-stream", b"\x00\x01\x02\xff")

    def test_a_type_without_a_reader_is_unreadable_by_name(self):
        with pytest.raises(UnreadableDocument, match="application/zip"):
            extract_document_text("photos.zip", "application/zip", b"PK\x03\x04")
        with pytest.raises(UnreadableDocument, match=r"\.xlsx"):
            extract_document_text("sheet.xlsx", "", b"PK\x03\x04")

    def test_an_empty_text_file_is_unreadable(self):
        with pytest.raises(UnreadableDocument, match="empty"):
            extract_document_text("empty.txt", "text/plain", b"  \n")

    def test_a_long_text_is_cut_at_the_cap_with_a_note(self):
        text = extract_document_text("big.log", "text/plain", b"x" * (MediaIntake.DOCUMENT_TEXT_CHARS + 500))
        kept, note = text.split("\n\n", 1)
        assert kept == "x" * MediaIntake.DOCUMENT_TEXT_CHARS and "truncated" in note


@pytest.mark.unit
class TestPictureFiles:
    @pytest.mark.parametrize("name, mime_type", [
        ("scan.jpg", "image/jpeg"), ("IMG_0001.HEIC", "image/heic"), ("unnamed", "image/webp"),
        ("drawing.PNG", ""), ("photo.heic", "application/octet-stream")])
    def test_a_picture_is_known_by_its_type_or_by_its_extension_under_a_generic_type(self, name, mime_type):
        assert MediaIntake.is_picture(name, mime_type)

    @pytest.mark.parametrize("name, mime_type", [
        ("logo.svg", "image/svg+xml"), ("invoice.pdf", "application/pdf"), ("notes.txt", "text/plain"),
        ("README", ""), ("holiday.jpg", "application/pdf")])
    def test_what_vision_cannot_be_shown_is_no_picture(self, name, mime_type):
        assert not MediaIntake.is_picture(name, mime_type)

    def test_a_large_picture_is_scaled_to_the_cap_and_keeps_its_shape(self):
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (4096, 1024), "blue").save(buf, format="PNG")
        jpeg = MediaIntake.picture_as_jpeg(buf.getvalue())
        assert Image.open(io.BytesIO(jpeg)).size == (MediaIntake.PICTURE_MAX_DIMENSION, MediaIntake.PICTURE_MAX_DIMENSION // 4)

    def test_a_picture_that_raises_in_the_converter_is_none(self):
        with patch("assistant.tools.ImageAnalyzer.resize_to_jpeg", side_effect=OSError("image file is truncated")):
            assert MediaIntake.picture_as_jpeg(b"\xff\xd8 cut short") is None

    def test_a_picture_that_reaches_the_text_readers_is_unreadable(self):
        with pytest.raises(UnreadableDocument):
            extract_document_text("broken.png", "image/png", b"not a picture")


@pytest.mark.unit
class TestDocumentText:
    def test_the_body_names_the_file_its_type_and_size_then_the_caption_then_the_text(self):
        body, headline = document_text(document(b"a,b\n1,2\n", "text/csv", "sales.csv"), "Which sold more?")
        assert body == ("[User sent a document: sales.csv (text/csv, 8 bytes)]\n"
                        "Caption: Which sold more?\n"
                        "a,b\n1,2\n")
        assert headline == "[User sent a document: sales.csv (text/csv, 8 bytes)]\nCaption: Which sold more?"

    def test_no_caption_means_no_caption_line(self):
        body, headline = document_text(document(b"hi", "text/plain", "a.txt"))
        assert body.split("\n") == ["[User sent a document: a.txt (text/plain, 2 bytes)]", "hi"]
        assert headline == body.split("\n")[0]

    def test_an_undelivered_file_is_described_by_the_size_the_platform_reported(self):
        att = document(b"", "video/quicktime", "film.mov", size=25 * 1024 * 1024, undelivered="it is over the cap")
        body, headline = document_text(att)
        assert "25.0 MB" in headline
        assert body.endswith("The file was not downloaded: it is over the cap. Its contents are unknown.")

    def test_an_unreadable_file_tells_the_assistant_why(self):
        body, headline = document_text(document(b"PK\x03\x04", "application/zip", "photos.zip"), "Look")
        assert body.startswith(headline)
        assert "could not be read (no reader for application/zip)" in body and "unknown" in body

    def test_a_nameless_file_with_no_type_still_gets_a_header(self):
        body, _ = document_text(document(b"text", "", ""))
        assert body.startswith("[User sent a document: unnamed (unknown type, 4 bytes)]")


@pytest.mark.unit
def test_describe_size_picks_the_unit():
    assert describe_size(900) == "900 bytes"
    assert describe_size(3 * 1024) == "3 KB"
    assert describe_size(int(1.5 * 1024 * 1024)) == "1.5 MB"


@pytest.mark.unit
class TestVoiceNoteLog:
    STT = "interfaces.telegram.GoogleCloudSTT.GoogleCloudSTT.get_instance"

    @pytest.fixture
    def folder(self, tmp_path, monkeypatch):
        monkeypatch.setattr(MediaIntake, "VOICE_NOTE_DIR", tmp_path / "audio" / "voice_notes")
        return MediaIntake.VOICE_NOTE_DIR

    def transcribe(self, heard: str, data: bytes, mime_type: str = "audio/ogg") -> str:
        with patch(self.STT) as stt:
            stt.return_value.transcribe.return_value = heard
            return MediaIntake.transcribe_voice(data, mime_type)

    def test_a_note_the_stt_heard_nothing_in_is_kept_for_the_diagnosis(self, folder):
        assert self.transcribe("", b"OggS-silence") == ""
        [saved] = folder.iterdir()
        assert saved.read_bytes() == b"OggS-silence" and saved.suffix == ".ogg"

    def test_a_transcribed_note_is_named_after_what_was_heard(self, folder):
        assert self.transcribe("Dim the office lamp", b"OggS-speech") == "Dim the office lamp"
        [saved] = folder.iterdir()
        assert "Dim_the_office_lamp" in saved.name and saved.read_bytes() == b"OggS-speech"

    def test_the_file_type_follows_what_the_platform_said_it_sent(self, folder):
        self.transcribe("hello", b"ftypM4A", "audio/mp4")
        [saved] = folder.iterdir()
        assert saved.suffix == ".m4a"

    def test_a_note_that_cannot_be_saved_is_still_transcribed(self, folder):
        with patch.object(MediaIntake, "save_audio_file", side_effect=OSError("disk full")):
            assert self.transcribe("hello", b"OggS") == "hello"
