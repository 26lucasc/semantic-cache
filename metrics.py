"""In-memory counters. Reset on restart -- enough to make the hit rate visible,
not a monitoring system."""
import threading

_lock = threading.Lock()
_c = {"exact_hits": 0, "semantic_hits": 0, "misses": 0, "bypasses": 0, "poisoned_skips": 0,
      "personal_skips": 0}


def bump(name: str) -> None:
    with _lock:
        _c[name] += 1


def snapshot() -> dict:
    with _lock:
        hits = _c["exact_hits"] + _c["semantic_hits"]
        total = hits + _c["misses"]
        return {
            **_c,
            "hit_rate": round(hits / total, 4) if total else 0.0,
            "_note": "in-memory, reset on restart",
        }


def reset() -> None:
    with _lock:
        for k in _c:
            _c[k] = 0
