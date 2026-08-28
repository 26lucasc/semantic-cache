"""Cache ON vs OFF over identical traffic. Produces the comparison table.

Run:  .venv/bin/python benchmark.py

WHAT IS MEASURED vs MODELED
  measured  every latency on the cache path (embed, hash lookup, vector
            search, verifier), hit rates, memory, disk
  modeled   the generation itself -- llm.py's `fake` backend sleeps 800ms and
            derives token counts from text length, priced at Claude Sonnet 5
            list ($2/$10 per Mtok). Cost and generation latency are therefore a
            PARAMETERIZED MODEL, not a measurement. Set LLM_BACKEND=anthropic
            with a real key to measure them.
The hit rate is real. What each hit is worth depends on your actual model.
"""
import gc
import random
import resource
import time
from pathlib import Path

import metrics
from cache import FLOOR, THRESHOLD, TOP_K, VERIFY, SemanticCache
from eval_pairs import NEGATIVES, POSITIVES
from store import Store

random.seed(0)

# Traffic model: a few popular questions asked many ways, plus a long tail of
# one-off questions that can never hit. Zipf over topics is the shape real
# support traffic takes; the tail is what keeps the benchmark honest.
TAIL_FRACTION = 0.30


def build_traffic(n=200):
    topics = {}
    for base, variant, _ in POSITIVES:
        topics.setdefault(base, [base]).append(variant)
    bases = sorted(topics)
    weights = [1 / (i + 1) for i in range(len(bases))]          # Zipf-ish
    tail = [b for _, b, _ in NEGATIVES]                          # unique, never repeat

    out, t = [], 0
    for _ in range(n):
        if random.random() < TAIL_FRACTION and t < len(tail):
            out.append(tail[t]); t += 1
        else:
            base = random.choices(bases, weights)[0]
            out.append(random.choice(topics[base]))
    return out


def run(traffic, bypass: bool, label: str):
    Path("cache.db").unlink(missing_ok=True)
    metrics.reset()
    gc.collect()
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    cpu0 = resource.getrusage(resource.RUSAGE_SELF).ru_utime

    cache = SemanticCache(Store("cache.db"))
    lat, cost, tin, tout = [], 0.0, 0, 0
    t0 = time.perf_counter()
    for i, q in enumerate(traffic, 1):
        r = cache.ask(q, bypass=bypass)
        lat.append(r["latency_ms"]); cost += r["cost"]
        if not r["cached"]:
            tin += len(q) // 4
            tout += len(r["answer"]) // 4
        print(f"\r  {label}: {i}/{len(traffic)}", end="", flush=True)
    wall = time.perf_counter() - t0

    m = metrics.snapshot()
    hits = m["exact_hits"] + m["semantic_hits"]
    pct = lambda p: sorted(lat)[min(int(len(lat) * p), len(lat) - 1)]
    print()
    return {
        "label": label, "wall_s": wall, "requests": len(traffic),
        "llm_calls": len(traffic) - hits, "hits": hits,
        "exact": m["exact_hits"], "semantic": m["semantic_hits"],
        "cost": cost, "tokens_in": tin, "tokens_out": tout,
        "p50": pct(0.50), "p95": pct(0.95), "mean": sum(lat) / len(lat),
        "rss_mb": (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss0) / 1e6,
        "cpu_s": resource.getrusage(resource.RUSAGE_SELF).ru_utime - cpu0,
        "entries": cache.store.stats()["entries"],
        "db_kb": Path("cache.db").stat().st_size / 1024,
        "vec_kb": cache.store.matrix.nbytes / 1024,
    }


def model_rss():
    """Resident memory attributable to the two models, measured in a fresh
    process. A per-run delta of ru_maxrss cannot show this: it is a high-water
    mark for the whole process and never comes back down, so the second run's
    delta is meaningless (it reads negative)."""
    import subprocess
    import sys
    import textwrap
    code = textwrap.dedent("""
        import resource
        b = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        from embedder import embed; embed("warm")
        from verifier import verify; verify("a", ["b"])
        a = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print((a - b) / 1e6)
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=".")
    try:
        return float(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return float("nan")


def report(off, on):
    d = lambda a, b: (f"{(1 - b / a) * 100:.0f}% less" if a else "n/a")
    x = lambda a, b: (f"{a / b:.0f}x faster" if b else "n/a")
    print(f"\n{'=' * 68}")
    print(f"  SEMANTIC CACHE: {on['requests']} requests, identical traffic")
    print(f"  verify={VERIFY} k={TOP_K} floor={FLOOR}  |  generation = modeled")
    print("=" * 68)

    print("\n  THROUGHPUT & SPEED")
    print(f"    {'':22s}{'no cache':>14s}{'with cache':>14s}{'':>16s}")
    print(f"    {'wall clock':22s}{off['wall_s']:13.1f}s{on['wall_s']:13.1f}s"
          f"{'  ' + d(off['wall_s'], on['wall_s']):>16s}")
    print(f"    {'throughput (req/s)':22s}{off['requests']/off['wall_s']:14.2f}"
          f"{on['requests']/on['wall_s']:14.2f}"
          f"{'  ' + x(on['requests']/on['wall_s'], off['requests']/off['wall_s']):>16s}")
    print(f"    {'mean latency':22s}{off['mean']:12.0f}ms{on['mean']:12.0f}ms"
          f"{'  ' + d(off['mean'], on['mean']):>16s}")
    print(f"    {'p50 latency':22s}{off['p50']:12.0f}ms{on['p50']:12.0f}ms"
          f"{'  ' + d(off['p50'], on['p50']):>16s}")
    print(f"    {'p95 latency':22s}{off['p95']:12.0f}ms{on['p95']:12.0f}ms"
          f"{'  ' + d(off['p95'], on['p95']):>16s}")

    print("\n  MODEL RESOURCE UTILIZATION")
    print(f"    {'inference calls':22s}{off['llm_calls']:14d}{on['llm_calls']:14d}"
          f"{'  ' + d(off['llm_calls'], on['llm_calls']):>16s}")
    print(f"    {'input tokens':22s}{off['tokens_in']:14d}{on['tokens_in']:14d}"
          f"{'  ' + d(off['tokens_in'], on['tokens_in']):>16s}")
    print(f"    {'output tokens':22s}{off['tokens_out']:14d}{on['tokens_out']:14d}"
          f"{'  ' + d(off['tokens_out'], on['tokens_out']):>16s}")
    print(f"    {'cost (Sonnet 5)':22s}{'$' + format(off['cost'], '.4f'):>14s}"
          f"{'$' + format(on['cost'], '.4f'):>14s}"
          f"{'  ' + d(off['cost'], on['cost']):>16s}")
    per_1k = lambda r: r["cost"] / r["requests"] * 1000
    print(f"    {'cost / 1k requests':22s}{'$' + format(per_1k(off), '.3f'):>14s}"
          f"{'$' + format(per_1k(on), '.3f'):>14s}"
          f"{'  ' + d(per_1k(off), per_1k(on)):>16s}")

    print("\n  CACHE OVERHEAD (what you pay for the saving)")
    print(f"    hit rate            {on['hits'] / on['requests']:.0%}  "
          f"(exact {on['exact'] / on['requests']:.0%} + semantic "
          f"{on['semantic'] / on['requests']:.0%})")
    print(f"    entries stored      {on['entries']}")
    print(f"    vectors in RAM      {on['vec_kb']:.1f} KB  "
          f"({on['vec_kb'] * 1024 / max(on['entries'], 1):.0f} bytes/entry)")
    print(f"    sqlite on disk      {on['db_kb']:.1f} KB")
    print(f"    model RAM           {model_rss():.0f} MB (embedder + cross-encoder)")
    print(f"    CPU time            {off['cpu_s']:.1f}s -> {on['cpu_s']:.1f}s")
    print()


if __name__ == "__main__":
    traffic = build_traffic(200)
    print(f"traffic: {len(traffic)} requests, {len(set(traffic))} distinct "
          f"({TAIL_FRACTION:.0%} unique tail)")
    off = run(traffic, bypass=True, label="no cache ")
    on = run(traffic, bypass=False, label="with cache")
    report(off, on)
