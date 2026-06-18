"""Tiny in-process metrics (SPEC_02 §7).

Process-level counters (reset on restart) + one structured log line per
request. Surfaced via GET /stats.
"""
import logging
import threading
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("udea-faq")


@dataclass
class Counters:
    total: int = 0
    cache_hits: int = 0
    agent_calls: int = 0


counters = Counters()
_lock = threading.Lock()   # sync handlers run concurrently in FastAPI's threadpool


def record(source: str, score: float, latency_ms: int) -> None:
    with _lock:
        counters.total += 1
        if source == "cache":
            counters.cache_hits += 1
        else:
            counters.agent_calls += 1
    logger.info("source=%s score=%.4f latency_ms=%d", source, score, latency_ms)


def snapshot() -> dict:
    with _lock:
        total = counters.total
        cache_hits = counters.cache_hits
        agent_calls = counters.agent_calls
    hit_rate = cache_hits / total if total else 0.0
    return {
        "total": total,
        "cache_hits": cache_hits,
        "agent_calls": agent_calls,
        "hit_rate": round(hit_rate, 4),
    }
