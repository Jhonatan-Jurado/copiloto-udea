"""In-memory OCR for a single PDF, reusing the preprocessing pipeline.

`pdf_to_text(pdf_path)` replicates `preprocessing/src/ocr/pipeline.procesar_pdf`'s
orchestration (detect native vs scanned → extract per page → join with page
markers → clean) WITHOUT writing a .txt. The result is header-less, page-markered
text that `app.chunking.chunk_document` consumes directly.

Native PDFs use PyMuPDF only (fast, no Tesseract). Scanned PDFs need Tesseract
(with the `spa` language pack) + Poppler, whose paths come from preprocessing/.env.
"""
from pathlib import Path

# repo_root/preprocessing/src/ocr  (this file is repo_root/backend/app/ocr.py)
_OCR_DIR = Path(__file__).resolve().parent.parent.parent / "preprocessing" / "src" / "ocr"
_PREPROCESSING_ENV = _OCR_DIR.parent.parent / ".env"


class OCRDependencyError(RuntimeError):
    """OCR libs or external binaries (PyMuPDF/pdf2image/pytesseract/Tesseract/Poppler) missing."""


def _load_ocr_module():
    import sys

    # Make preprocessing/.env deterministic regardless of cwd/debugger: load it
    # before importing extractor_ocr (whose own load_dotenv() could otherwise pick
    # backend/.env under a REPL/debugger). load_dotenv does not override existing.
    try:
        from dotenv import load_dotenv
        if _PREPROCESSING_ENV.exists():
            load_dotenv(_PREPROCESSING_ENV)
    except Exception:
        pass

    ocr_dir = str(_OCR_DIR)
    if ocr_dir not in sys.path:
        sys.path.insert(0, ocr_dir)
    try:
        from pdf_detector import diagnosticar_pdf
        from extractor_nativo import extraer_texto_nativo
        from extractor_ocr import extraer_texto_ocr      # runs load_dotenv() at import
        from limpiador import limpiar_texto
    except Exception as e:
        raise OCRDependencyError(
            f"No se pudo cargar el pipeline OCR ({e}). Instala pymupdf, pdf2image, "
            f"pytesseract y verifica TESSERACT_PATH/POPPLER_PATH en preprocessing/.env."
        ) from e
    return diagnosticar_pdf, extraer_texto_nativo, extraer_texto_ocr, limpiar_texto


def pdf_to_text(pdf_path: str) -> str:
    """Detect → extract per page → join with `--- PÁGINA N ---` markers → clean.

    Returns header-less text shaped like the processed `.txt`, ready for
    `app.chunking.chunk_document`. Raises OCRDependencyError if the OCR stack is
    missing; other extraction errors (bad PDF, missing `spa`, etc.) propagate.
    """
    diagnosticar_pdf, extraer_nativo, extraer_ocr, limpiar = _load_ocr_module()
    diagnostico = diagnosticar_pdf(pdf_path)
    if diagnostico["necesita_ocr"]:
        paginas = extraer_ocr(pdf_path)                  # Tesseract+Poppler; needs `spa`
        fuente = "ocr"
    else:
        paginas = extraer_nativo(pdf_path)               # PyMuPDF only (fast)
        fuente = "nativo"
    partes = [f"--- PÁGINA {p['pagina']} ---\n{p['texto']}" for p in paginas]
    return limpiar("\n\n".join(partes), fuente=fuente)
