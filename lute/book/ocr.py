"""
OCR fallback for scanned PDFs (PDFs with no embedded text layer).

This is an OPTIONAL feature.  It requires:

  * the 'ocr' extra:  pip install 'lute3[ocr]'   (installs ocrmypdf)
  * a system Tesseract install with the relevant language data,
    e.g.  apt install tesseract-ocr tesseract-ocr-ita

If either is missing, callers fall back to whatever (possibly empty)
text the PDF text layer yielded -- OCR is never required to import a
normal, born-digital PDF.

OCR runs synchronously inside the import request and is CPU-heavy, so
the number of pages processed is capped (MAX_OCR_PAGES).
"""

import os
import shutil
import tempfile

from pypdf import PdfReader, PdfWriter

# Map Lute language names (Language.name, lower-cased) to Tesseract
# language codes (ISO 639-3).  The matching Tesseract traineddata must
# be installed, e.g. `apt install tesseract-ocr-deu` for German.
_TESSERACT_LANGS = {
    "arabic": "ara",
    "czech": "ces",
    "danish": "dan",
    "dutch": "nld",
    "english": "eng",
    "finnish": "fin",
    "french": "fra",
    "german": "deu",
    "greek": "ell",
    "hindi": "hin",
    "hungarian": "hun",
    "italian": "ita",
    "japanese": "jpn",
    "korean": "kor",
    "norwegian": "nor",
    "polish": "pol",
    "portuguese": "por",
    "romanian": "ron",
    "russian": "rus",
    "spanish": "spa",
    "swedish": "swe",
    "turkish": "tur",
    "ukrainian": "ukr",
    "vietnamese": "vie",
    "classical chinese": "chi_tra",
    "mandarin chinese": "chi_sim",
}

# OCR is slow and runs in-request; cap the work.
MAX_OCR_PAGES = 50


def ocr_available():
    "True if both the tesseract binary and ocrmypdf are installed."
    if shutil.which("tesseract") is None:
        return False
    try:
        import ocrmypdf  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import,import-error
    except ImportError:
        return False
    return True


def tesseract_language(language_name):
    """
    Map a Lute language name to a Tesseract language code.

    Returns None if the language is unknown, in which case the caller
    should fall back to a default (typically English).
    """
    if not language_name:
        return None
    return _TESSERACT_LANGS.get(language_name.strip().lower())


def _limit_pages(src_path, dest_path, max_pages):
    "Write at most max_pages of src_path to dest_path; return page count."
    reader = PdfReader(src_path)
    writer = PdfWriter()
    n = min(len(reader.pages), max_pages)
    for i in range(n):
        writer.add_page(reader.pages[i])
    with open(dest_path, "wb") as f:
        writer.write(f)
    return n


def ocr_pdf_stream(filestream, tess_lang, max_pages=MAX_OCR_PAGES):
    """
    OCR a scanned PDF stream and return the extracted plain text.

    Requires ocr_available() to be True.  tess_lang is a Tesseract
    language code (see tesseract_language()); if None, English is used.

    Raises whatever ocrmypdf raises on failure (e.g. a missing language
    pack); the caller is expected to translate that into a
    BookImportException.
    """
    import ocrmypdf  # pylint: disable=import-outside-toplevel,import-error

    workdir = tempfile.mkdtemp(prefix="lute_ocr_")
    try:
        raw = os.path.join(workdir, "raw.pdf")
        capped = os.path.join(workdir, "in.pdf")
        out_pdf = os.path.join(workdir, "out.pdf")
        sidecar = os.path.join(workdir, "out.txt")

        filestream.seek(0)
        with open(raw, "wb") as f:
            f.write(filestream.read())

        _limit_pages(raw, capped, max_pages)

        # force_ocr: scanned pages have no usable text layer, so
        # rasterize and OCR every page.  The sidecar file receives the
        # recognized plain text.
        ocrmypdf.ocr(
            capped,
            out_pdf,
            language=tess_lang or "eng",
            sidecar=sidecar,
            force_ocr=True,
            progress_bar=False,
        )
        with open(sidecar, "r", encoding="utf-8") as f:
            return f.read()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
