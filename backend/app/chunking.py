"""Shared text-chunking for the corpus loaders and the upload endpoint.

Pure functions over the OCR-processed text (text in → chunks out). No DB / Azure
imports, so this module is import-safe and unit-testable on its own.

`chunk_document(raw_text)` returns a list of (articulo, pagina, content) chunks:
text is split at each ARTÍCULO heading (an article's body is carried across page
breaks), and long segments are sub-split to ~TARGET_CHARS. Text before the first
heading — or text with no detectable headings — is chunked per page with
articulo=None. Works on both the processed `.txt` (with a FUENTE/==== header) and
header-less in-memory OCR text (see app/ocr.py): `_strip_header` is a no-op when
there is no `====` separator line.
"""
import re

TARGET_CHARS = 600          # sub-split long segments to ~this size
MIN_CHUNK_CHARS = 60        # drop tiny noise fragments

PAGE_RE = re.compile(r"---\s*P[ÁA]GINA\s+(\d+)\s*---", re.IGNORECASE)
FOOTER_RE = re.compile(
    r"(Ciudad Universitaria|Comutador|Apartado:|Nit:|Medell[ií]n,|http)", re.IGNORECASE
)

# An article heading: ARTÍCULO (accent optional, tolerates OCR), then a number
# OR a spelled ordinal word, then ':' / '.' / '°'. Matched at line start; an
# uppercase guard (below) rejects lowercase inline references like
# "el artículo 130 del...". IGNORECASE here is intentional — the guard, not the
# regex, decides case.
HEADING_RE = re.compile(
    r"^\s*ART[IÍ]CULO\s+(?P<id>\d{1,3}|[A-Za-zÁÉÍÓÚáéíóú]+)\s*[:.°]",
    re.IGNORECASE,
)

# Spelled ordinals UdeA uses for article numbering (accent-stripped, lowercase).
_ORDINALS = {
    "primero": 1, "segundo": 2, "tercero": 3, "cuarto": 4, "quinto": 5,
    "sexto": 6, "septimo": 7, "octavo": 8, "noveno": 9, "decimo": 10,
    "undecimo": 11, "decimoprimero": 11, "duodecimo": 12, "decimosegundo": 12,
    "decimotercero": 13, "decimocuarto": 14, "decimoquinto": 15,
    "decimosexto": 16, "decimoseptimo": 17, "decimoctavo": 18,
    "decimonoveno": 19, "vigesimo": 20,
}


def _strip_accents(s: str) -> str:
    for a, b in (("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u")):
        s = s.replace(a, b)
    return s


def _is_heading_line(line: str) -> tuple[bool, str | None]:
    """(is_heading, articulo_label) for one stripped line.

    Rejects lowercase/Title-case inline references via an uppercase guard; maps
    spelled ordinals to a number; falls back to the word form for unknown words
    so a true heading boundary is never lost.
    """
    m = HEADING_RE.match(line)
    if not m:
        return False, None
    head = line[: m.end()]
    letters = [c for c in head if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) < 0.8:
        return False, None
    ident = m.group("id")
    if ident.isdigit():
        return True, f"Artículo {int(ident)}"
    n = _ORDINALS.get(_strip_accents(ident.lower()))
    if n is not None:
        return True, f"Artículo {n}"
    return True, f"Artículo {ident.title()}"


def _strip_header(text: str) -> str:
    """Drop the FUENTE/PAGINAS/... header above the '====' separator line."""
    parts = re.split(r"^={5,}\s*$", text, maxsplit=1, flags=re.MULTILINE)
    return parts[1] if len(parts) == 2 else text


def _pages(body: str) -> list[tuple[int, str]]:
    """Split a document body into (page_number, page_text) pairs."""
    segments = PAGE_RE.split(body)
    # segments = [pre, num, text, num, text, ...]
    out: list[tuple[int, str]] = []
    for i in range(1, len(segments), 2):
        out.append((int(segments[i]), segments[i + 1]))
    if not out:  # no page markers — treat whole body as page 1
        out = [(1, body)]
    return out


def _segments(pages: list[tuple[int, str]]) -> list[tuple[str | None, int, str]]:
    """Split into (articulo, pagina, text) segments.

    One segment per ARTÍCULO heading; an article's body is carried across page
    breaks (pagina = the heading's page). Outside any article (preamble, or a
    document with no headings at all) each PAGE becomes its own articulo=None
    segment, so page numbers stay accurate. Footer/boilerplate lines are dropped.
    """
    current_art: str | None = None
    start_page = pages[0][0] if pages else 1
    buf: list[str] = []
    segments: list[tuple[str | None, int, str]] = []

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            segments.append((current_art, start_page, text))

    for pagina, page_text in pages:
        # Preamble / heading-less mode: flush at each page boundary so a chunk
        # gets the page it actually lives on. Inside an article, let it span.
        if current_art is None:
            if buf:
                flush()
                buf = []
            start_page = pagina
        for raw in page_text.splitlines():
            line = raw.strip()
            if not line:
                buf.append("")          # keep paragraph breaks for _split_text
                continue
            if FOOTER_RE.search(line):
                continue                # drop address/phone/URL boilerplate
            is_head, label = _is_heading_line(line)
            if is_head:
                flush()
                buf = []
                current_art = label
                start_page = pagina
            buf.append(line)
    flush()
    return segments


def _split_text(text: str) -> list[str]:
    """Group paragraphs into ~TARGET_CHARS chunks (a short article stays whole)."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        buf = f"{buf}\n\n{para}".strip() if buf else para
        if len(buf) >= TARGET_CHARS:
            chunks.append(buf)
            buf = ""
    if buf:
        chunks.append(buf)
    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


def chunk_document(raw_text: str) -> list[tuple[str | None, int, str]]:
    """(articulo, pagina, content) chunks from processed or in-memory OCR text."""
    body = _strip_header(raw_text)
    out: list[tuple[str | None, int, str]] = []
    for articulo, pagina, seg_text in _segments(_pages(body)):
        for chunk in _split_text(seg_text):
            out.append((articulo, pagina, chunk))
    return out
