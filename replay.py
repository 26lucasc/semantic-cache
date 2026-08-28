"""Traffic replay: proves the running system works, after sweep.py picked the
threshold. Produces the headline table for the README.

Run:  .venv/bin/python replay.py
"""
import random

import metrics
from cache import THRESHOLD, SemanticCache
from eval_pairs import POSITIVES

random.seed(0)   # reproducible; do not use time-based seeds in a benchmark


def build_traffic(n=100):
    """Shuffled stream of paraphrases drawn from the eval positives.

    TODO(you): this reuses the eval set, which flatters the result. Generate a
    separate held-out set of paraphrases (an LLM will write 20 per base
    question) and report that number instead. Say which one you used.
    """
    queries = [b for _, b, _ in POSITIVES] + [a for a, _, _ in POSITIVES]
    return [random.choice(queries) for _ in range(n)]


def main(n=100):
    cache = SemanticCache()
    metrics.reset()
    cost, lat = 0.0, {"exact": [], "semantic": [], "llm": []}

    for i, q in enumerate(build_traffic(n), 1):
        r = cache.ask(q)
        cost += r["cost"]
        lat[r["layer"]].append(r["latency_ms"])
        print(f"\r{i}/{n}", end="", flush=True)

    p50 = lambda xs: sorted(xs)[len(xs) // 2] if xs else 0
    hits = len(lat["exact"]) + len(lat["semantic"])

    # Baseline measured through the same code path, not estimated: every
    # unique prompt would have cost one LLM call.
    baseline = cost / (1 - hits / n) if hits < n else float("inf")

    print(f"\n\nthreshold        {THRESHOLD}")
    print(f"requests         {n}")
    print(f"LLM calls        {n - hits}")
    print(f"hit rate         {hits / n:.0%}"
          f"  (exact {len(lat['exact']) / n:.0%} + semantic {len(lat['semantic']) / n:.0%})")
    print(f"p50 exact hit    {p50(lat['exact']):.1f}ms")
    print(f"p50 semantic hit {p50(lat['semantic']):.1f}ms")
    print(f"p50 miss         {p50(lat['llm']):.0f}ms")
    print(f"cost             ${cost:.4f}  (no cache: ~${baseline:.4f})")
    print(f"\nmetrics          {metrics.snapshot()}")


if __name__ == "__main__":
    main()
