"""
Tests for the OCR fallback helpers and the PDF import path.
"""

import os
import tempfile
from io import BytesIO
from unittest.mock import patch
from pypdf import PdfReader, PdfWriter
from lute.book import ocr
from lute.book.service import BookImportException, FileTextExtraction


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


def test_scanned_pdf_unknown_language_falls_back_to_english():
    "An unrecognized language name flashes a notice, and OCR still runs (English)."
    fte = FileTextExtraction()
    with patch(
        "lute.book.service.PdfReader", return_value=_FakeReader([""])
    ), patch(
        "lute.book.service.FileTextExtraction._ocr_enabled", return_value=True
    ), patch(
        "lute.book.ocr.ocr_available", return_value=True
    ), patch(
        "lute.book.ocr.ocr_pdf_stream", return_value="ocr text"
    ) as mock_ocr, patch(
        "lute.book.service.flash"
    ) as mock_flash:
        content = fte._get_pdf_content("scan.pdf", b"", "Klingon")
    assert content == "ocr text"
    args, _ = mock_ocr.call_args
    assert args[1] is None  # ocr_pdf_stream itself defaults None -> "eng"
    # Two notices expected: unknown-language warning, plus the "running OCR" notice.
    assert mock_flash.call_count == 2


def test_ocr_pdf_fallback_wraps_ocr_failure_in_book_import_exception():
    "If ocrmypdf raises, the caller sees a BookImportException, not a raw exception."
    fte = FileTextExtraction()
    with patch(
        "lute.book.service.PdfReader", return_value=_FakeReader([""])
    ), patch(
        "lute.book.service.FileTextExtraction._ocr_enabled", return_value=True
    ), patch(
        "lute.book.ocr.ocr_available", return_value=True
    ), patch(
        "lute.book.ocr.ocr_pdf_stream", side_effect=RuntimeError("missing langpack")
    ), patch(
        "lute.book.service.flash"
    ):
        try:
            fte._get_pdf_content("scan.pdf", b"", "Italian")
            assert False, "expected BookImportException"
        except BookImportException as e:
            assert "missing langpack" in e.message


def test_limit_pages_caps_page_count():
    "Only the first max_pages pages are copied to the output file."
    writer = PdfWriter()
    for _ in range(5):
        writer.add_blank_page(width=72, height=72)
    with tempfile.TemporaryDirectory() as workdir:
        src = os.path.join(workdir, "src.pdf")
        dest = os.path.join(workdir, "dest.pdf")
        with open(src, "wb") as f:
            writer.write(f)

        n = ocr._limit_pages(src, dest, 3)  # pylint: disable=protected-access

        assert n == 3
        assert len(PdfReader(dest).pages) == 3


def test_limit_pages_below_max_copies_all():
    "If the PDF has fewer pages than the cap, all pages are copied."
    writer = PdfWriter()
    for _ in range(2):
        writer.add_blank_page(width=72, height=72)
    with tempfile.TemporaryDirectory() as workdir:
        src = os.path.join(workdir, "src.pdf")
        dest = os.path.join(workdir, "dest.pdf")
        with open(src, "wb") as f:
            writer.write(f)

        n = ocr._limit_pages(src, dest, 50)  # pylint: disable=protected-access

        assert n == 2
        assert len(PdfReader(dest).pages) == 2


def test_ocr_pdf_stream_reads_sidecar_text_and_cleans_up_workdir():
    "ocr_pdf_stream() writes the capped PDF, invokes ocrmypdf, and returns sidecar text."
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buf = BytesIO()
    writer.write(buf)
    buf.seek(0)

    captured = {}

    def fake_ocr(capped, out_pdf, language, sidecar, force_ocr, progress_bar):
        # ocrmypdf normally writes out_pdf and sidecar; emulate just enough
        # for ocr_pdf_stream to read the sidecar back.
        captured["capped_exists"] = os.path.exists(capped)
        captured["language"] = language
        captured["force_ocr"] = force_ocr
        captured["progress_bar"] = progress_bar
        with open(out_pdf, "wb") as f:
            f.write(b"%PDF-1.4 fake")
        with open(sidecar, "w", encoding="utf-8") as f:
            f.write("recognized text")

    workdirs_used = []
    real_mkdtemp = tempfile.mkdtemp

    def tracking_mkdtemp(*args, **kwargs):
        d = real_mkdtemp(*args, **kwargs)
        workdirs_used.append(d)
        return d

    with patch("lute.book.ocr.tempfile.mkdtemp", side_effect=tracking_mkdtemp), patch.dict(
        "sys.modules", {"ocrmypdf": type(os)("ocrmypdf")}
    ):
        import sys  # pylint: disable=import-outside-toplevel

        sys.modules["ocrmypdf"].ocr = fake_ocr
        result = ocr.ocr_pdf_stream(buf, "ita")

    assert result == "recognized text"
    assert captured["capped_exists"] is True
    assert captured["language"] == "ita"
    assert captured["force_ocr"] is True
    assert captured["progress_bar"] is False
    # Temp workdir is cleaned up afterwards.
    assert workdirs_used and not os.path.exists(workdirs_used[0])
