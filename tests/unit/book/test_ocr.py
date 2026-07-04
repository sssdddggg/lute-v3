"""
Tests for the OCR fallback helpers and the PDF import path.
"""

from unittest.mock import patch
from lute.book import ocr
from lute.book.service import FileTextExtraction


def test_tesseract_language_maps_known_names():
    "Known Lute language names map to Tesseract codes, case-insensitively."
    assert ocr.tesseract_language("Italian") == "ita"
    assert ocr.tesseract_language("italian") == "ita"
    assert ocr.tesseract_language("  German ") == "deu"
    assert ocr.tesseract_language("Classical Chinese") == "chi_tra"


def test_tesseract_language_unknown_returns_none():
    "Unknown or empty language names return None."
    assert ocr.tesseract_language("Klingon") is None
    assert ocr.tesseract_language("") is None
    assert ocr.tesseract_language(None) is None


def test_ocr_available_false_without_tesseract():
    "ocr_available() is False when the tesseract binary is absent."
    with patch("lute.book.ocr.shutil.which", return_value=None):
        assert ocr.ocr_available() is False


class _FakeReader:
    "Minimal stand-in for pypdf.PdfReader."

    def __init__(self, page_texts):
        self.pages = [_FakePage(t) for t in page_texts]


class _FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


def test_pdf_with_text_layer_skips_ocr():
    "A PDF with a real text layer returns that text and never calls OCR."
    fte = FileTextExtraction()
    rich = "Questo è un testo lungo e leggibile. " * 5
    with patch("lute.book.service.PdfReader", return_value=_FakeReader([rich])), patch(
        "lute.book.ocr.ocr_pdf_stream"
    ) as mock_ocr:
        content = fte._get_pdf_content("book.pdf", b"", "Italian")
    assert content.strip().startswith("Questo")
    mock_ocr.assert_not_called()


def test_scanned_pdf_triggers_ocr_when_available():
    "A thin/empty text layer falls back to OCR with the mapped language."
    fte = FileTextExtraction()
    with patch(
        "lute.book.service.PdfReader", return_value=_FakeReader(["", ""])
    ), patch(
        "lute.book.service.FileTextExtraction._ocr_enabled", return_value=True
    ), patch(
        "lute.book.ocr.ocr_available", return_value=True
    ), patch(
        "lute.book.ocr.ocr_pdf_stream", return_value="ocr text"
    ) as mock_ocr, patch(
        "lute.book.service.flash"
    ):
        content = fte._get_pdf_content("scan.pdf", b"", "Italian")
    assert content == "ocr text"
    # Called with the Tesseract code for Italian.
    args, _ = mock_ocr.call_args
    assert args[1] == "ita"


def test_scanned_pdf_returns_thin_text_when_ocr_unavailable():
    "Empty text layer is returned when OCR unavailable, so caller flags it."
    fte = FileTextExtraction()
    with patch("lute.book.service.PdfReader", return_value=_FakeReader([""])), patch(
        "lute.book.service.FileTextExtraction._ocr_enabled", return_value=True
    ), patch("lute.book.ocr.ocr_available", return_value=False), patch(
        "lute.book.service.flash"
    ) as mock_flash:
        content = fte._get_pdf_content("scan.pdf", b"", "Italian")
    assert content == ""
    mock_flash.assert_called()  # user told how to enable OCR


def test_scanned_pdf_disabled_by_config_skips_ocr():
    "When OCR is disabled in config, no OCR is attempted."
    fte = FileTextExtraction()
    with patch("lute.book.service.PdfReader", return_value=_FakeReader([""])), patch(
        "lute.book.service.FileTextExtraction._ocr_enabled", return_value=False
    ), patch("lute.book.ocr.ocr_available", return_value=True) as mock_avail:
        content = fte._get_pdf_content("scan.pdf", b"", "Italian")
    assert content == ""
    mock_avail.assert_not_called()
