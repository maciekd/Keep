"""DocumentText: what each kind of file extracts to, the paging that keeps a page under the
history's cap and cuts at a boundary, and the find that says which page a phrase is on."""
import pytest

from integrations.dropbox.DocumentText import PAGE_CHARS, Unreadable, extract, find, pages, rtf_text, snippet

from .dropbox_fakes import make_docx, make_pdf, make_pptx, make_rtf


@pytest.mark.unit
class TestExtract:
    def test_utf8_text_is_decoded_and_a_bom_dropped(self):
        assert extract("notes.txt", "﻿Hej världen\n".encode("utf-8")) == "Hej världen\n"

    def test_latin1_text_is_a_fallback(self):
        assert extract("old.txt", "Hej världen".encode("latin-1")) == "Hej världen"

    def test_an_unknown_extension_that_looks_like_text_is_read_as_text(self):
        assert extract("Makefile.custom", b"all:\n\techo hi\n") == "all:\n\techo hi\n"

    def test_a_binary_of_an_unknown_kind_is_refused_as_binary(self):
        with pytest.raises(Unreadable, match="binary"):
            extract("blob.dat", b"\x00\x01\x02\xff" * 10)

    def test_images_and_spreadsheets_are_refused_by_kind(self):
        with pytest.raises(Unreadable, match=r"\.jpg"):
            extract("photo.jpg", b"\xff\xd8\xff")
        with pytest.raises(Unreadable, match=r"\.xlsx"):
            extract("budget.xlsx", b"PK\x03\x04")

    def test_a_pdf_gives_its_pages_in_order_each_under_a_marker(self):
        text = extract("lease.pdf", make_pdf(["The deposit is two months rent", "Signed in Stockholm"]))
        assert text.split("\n\n") == ["[p. 1]\nThe deposit is two months rent", "[p. 2]\nSigned in Stockholm"]

    def test_a_pdf_without_a_text_layer_says_it_is_a_scan(self):
        with pytest.raises(Unreadable, match="no text layer"):
            extract("scan.pdf", make_pdf(["", "", ""]))

    def test_a_broken_pdf_is_unreadable_not_a_crash(self):
        with pytest.raises(Unreadable, match="could not be read"):
            extract("broken.pdf", b"%PDF-1.4 garbage")

    def test_a_docx_gives_its_paragraphs_and_table_rows_with_tabs(self):
        text = extract("minutes.docx", make_docx(["Minutes", "Present: Marcus, Charlotte"], [["Item", "Owner"], ["Roof", "Frances"]]))
        assert text == "Minutes\nPresent: Marcus, Charlotte\nItem\tOwner\nRoof\tFrances"

    def test_a_zip_that_is_not_a_document_is_unreadable(self):
        with pytest.raises(Unreadable):
            extract("odd.docx", b"not a zip at all")

    def test_a_pptx_gives_its_slides_in_number_order_under_markers(self):
        text = extract("deck.pptx", make_pptx([["Plan", "Step one"], ["Step two"]]))
        assert text == "[slide 1]\nPlan\nStep one\n\n[slide 2]\nStep two"

    def test_a_textedit_rtf_is_plain_text_with_the_swedish_letters_decoded(self):
        text = extract("recept.rtf", make_rtf(["Friterad kyckling å la Åsa", "", "1 kg kycklinglår, smör och ägg"]))
        assert text == "Friterad kyckling å la Åsa\n\n1 kg kycklinglår, smör och ägg"
        assert "\\" not in text and "fonttbl" not in text and "Helvetica" not in text

    def test_a_word_rtf_in_another_code_page_decodes_in_that_one(self):
        text = extract("dopis.rtf", make_rtf(["Příliš žluťoučký kůň", "Řádek dva"], code_page=1250, cocoa=False))
        assert text == "Příliš žluťoučký kůň\nŘádek dva"

    def test_an_rtf_is_known_by_its_header_whatever_the_name(self):
        assert extract("TXT.rtf", make_rtf(["Inuti bundeln"])) == "Inuti bundeln"
        assert extract("notes.txt", make_rtf(["Fel ändelse"])) == "Fel ändelse"
        assert extract("plain.rtf", "Ingen RTF alls\n".encode()) == "Ingen RTF alls\n", "a .rtf without the header is text"

    def test_unicode_escapes_fields_and_tabs_in_an_rtf(self):
        data = (b"{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0 Calibri;}}\\pard Pris\\tab \\u8364?10\\par "
                b"{\\field{\\*\\fldinst HYPERLINK \"http://x.example\"}{\\fldrslt x.example}}\\par "
                b"\\uc2 \\u229??ngest {\\b fet} \\'e4r\\par}")
        assert rtf_text(data) == "Pris\t€10\nx.example\nångest fet är"


def paragraphs(count: int, words: int = 60) -> str:
    return "\n\n".join(f"Paragraph {i}: " + " ".join(f"w{i}-{j}" for j in range(words)) for i in range(1, count + 1))


@pytest.mark.unit
class TestPages:
    def test_a_short_text_is_one_page_and_an_empty_one_still_a_page(self):
        assert pages("hello") == ["hello"] and pages("  \n") == [""]

    def test_every_page_is_under_the_limit_and_cut_at_a_paragraph_boundary(self):
        text = paragraphs(60)
        out = pages(text)

        assert len(out) > 1 and all(len(p) <= PAGE_CHARS for p in out)
        for page in out:
            assert page.startswith("Paragraph ") and not page.endswith("-")
        assert "\n\n".join(out).replace("\n\n", " ").split() == text.replace("\n\n", " ").split(), "nothing is lost"

    def test_a_text_without_blank_lines_is_cut_at_a_line_end(self):
        text = "\n".join(f"line {i} " + "x" * 80 for i in range(200))
        out = pages(text, limit=1000)
        assert all(len(p) <= 1000 for p in out) and all(p.endswith("x") for p in out[:-1])

    def test_one_endless_word_is_cut_hard_rather_than_never(self):
        assert [len(p) for p in pages("a" * 2500, limit=1000)] == [1000, 1000, 500]


@pytest.mark.unit
class TestFind:
    TEXT = "[p. 1]\nThe lease begins in March.\nA deposit of two months is payable.\n\n[p. 2]\nThe deposit is returned within 30 days.\nSigned."

    def test_hits_carry_the_page_the_pdf_page_and_the_line(self):
        hits = find(self.TEXT, ["deposit"])
        assert [(h.page, h.marker, h.line) for h in hits] == [
            (1, "[p. 1]", "A deposit of two months is payable."),
            (1, "[p. 2]", "The deposit is returned within 30 days."),
        ]

    def test_every_word_must_be_on_the_line_case_aside(self):
        assert [h.line for h in find(self.TEXT, ["DEPOSIT", "returned"])] == ["The deposit is returned within 30 days."]
        assert find(self.TEXT, ["deposit", "march"]) == []

    def test_the_page_number_is_the_page_of_pages(self):
        text = paragraphs(60)
        hits = find(text, ["Paragraph 60:"])
        assert len(hits) == 1 and hits[0].page == len(pages(text)) and hits[0].marker == ""

    def test_the_hits_are_capped(self):
        text = "\n".join(f"needle {i}" for i in range(50))
        assert len(find(text, ["needle"], limit=20)) == 20

    def test_no_words_means_no_hits(self):
        assert find(self.TEXT, ["", "  "]) == []

    def test_a_long_line_is_shown_around_the_word(self):
        line = "x" * 300 + " deposit " + "y" * 300
        shown = snippet(line, "deposit", limit=100)
        assert "deposit" in shown and len(shown) <= 102 and shown.startswith("…") and shown.endswith("…")
        assert snippet("short", "short") == "short"
