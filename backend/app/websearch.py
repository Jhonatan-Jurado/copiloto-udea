"""Web-search tool for the agent, scoped to udea.edu.co and its subdomains.

A single LangChain tool (`buscar_web_udea`) for CURRENT / topical information about
the university (events, news, academic-calendar dates, calls) — NOT for regulations
(that's `buscar_reglamento`). Backed by DuckDuckGo (no API key).

The domain restriction is enforced two ways: a `site:udea.edu.co` query operator
(a hint to the engine) AND a strict post-filter on each result's hostname, which is
the real guarantee that only udea.edu.co / *.udea.edu.co results are returned.
"""
import json
import logging
from urllib.parse import urlparse

from langchain_core.tools import tool

from app.config import settings

logger = logging.getLogger("udea-faq")

ALLOWED_DOMAIN = "udea.edu.co"

_searcher = None


def _get_searcher():
    """Build the DuckDuckGo searcher lazily so importing this module doesn't require
    langchain-community/ddgs (and a missing dep degrades gracefully at call time).
    Asks for more than we keep, since the hostname post-filter drops off-domain hits.
    """
    global _searcher
    if _searcher is None:
        from langchain_community.tools import DuckDuckGoSearchResults
        from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
        _searcher = DuckDuckGoSearchResults(
            output_format="list",
            api_wrapper=DuckDuckGoSearchAPIWrapper(
                region="co-es",
                max_results=max(settings.web_search_top_k * 3, 10),
            ),
        )
    return _searcher


def _host_allowed(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host == ALLOWED_DOMAIN or host.endswith("." + ALLOWED_DOMAIN)


def _normalize(item) -> dict | None:
    """Map a DuckDuckGo result to {titulo, url, snippet}, tolerant of key drift
    (duckduckgo-search -> ddgs use link/href, snippet/body)."""
    if not isinstance(item, dict):
        return None
    url = item.get("link") or item.get("href") or ""
    if not url or not _host_allowed(url):
        return None
    return {
        "titulo": item.get("title") or "",
        "url": url,
        "snippet": item.get("snippet") or item.get("body") or item.get("description") or "",
    }


@tool
def buscar_web_udea(consulta: str) -> str:
    """Búsqueda web acotada a sitios de la Universidad de Antioquia (udea.edu.co y subdominios).

    Úsala EXCLUSIVAMENTE para información ACTUAL o de coyuntura de la Universidad:
    eventos, noticias, fechas del calendario académico vigente, convocatorias,
    inscripciones abiertas y novedades. NUNCA la uses para preguntas sobre el
    reglamento estudiantil, acuerdos, resoluciones, normativa, requisitos o trámites
    académicos: para eso usa siempre `buscar_reglamento`. Menciona la URL de la fuente.

    Args:
        consulta: la búsqueda en español (términos o pregunta sobre el tema actual).
    """
    empty = json.dumps(
        {"resultados": [], "nota": "Sin resultados recientes en udea.edu.co."},
        ensure_ascii=False,
    )
    try:
        raw = _get_searcher().invoke(f"site:{ALLOWED_DOMAIN} {consulta}")
        if isinstance(raw, str):        # some versions return a JSON string
            raw = json.loads(raw)
    except Exception as e:
        logger.warning("buscar_web_udea failed: %s", e)
        return empty

    if not isinstance(raw, list):
        return empty
    resultados = [r for r in (_normalize(it) for it in raw) if r][: settings.web_search_top_k]
    if not resultados:
        return empty
    return json.dumps({"resultados": resultados}, ensure_ascii=False)
