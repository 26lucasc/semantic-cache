"""End-to-end eval of retrieve-loose -> verify, against the threshold design.

Run:  .venv/bin/python eval_two_stage.py
"""
import time

import numpy as np

from embedder import embed_many
from eval_retrieval import CORPUS, NEGATIVE_QUERIES, POSITIVE_QUERIES
from verifier import ACCEPT, BACKEND, verify

K = 5
FLOOR = 0.15   # permissive: only keeps the cold-cache garbage out


def run():
    cvecs = embed_many(CORPUS)
    queries = POSITIVE_QUERIES + NEGATIVE_QUERIES
    qvecs = embed_many([q for q, _, _ in queries])

    tp = fp = fn = tn = 0
    t_verify = []
    misses, false_hits = [], []

    for (query, target, tag), qv in zip(queries, qvecs):
        sims = cvecs @ qv
        order = [i for i in np.argsort(-sims)[:K] if sims[i] >= FLOOR]
        cands = [CORPUS[i] for i in order]

        t0 = time.perf_counter()
        pick = verify(query, cands)
        t_verify.append((time.perf_counter() - t0) * 1000)

        served = cands[pick] if pick is not None else None
        if target is None:
            if served is None:
                tn += 1
            else:
                fp += 1
                false_hits.append((query, served, tag))
        else:
            if served == target:
                tp += 1
            elif served is None:
                fn += 1
                misses.append((query, target, tag))
            else:
                fp += 1
                false_hits.append((query, served, tag))

    n_pos = len(POSITIVE_QUERIES)
    print(f"\nverifier={BACKEND}  k={K}  accept>={ACCEPT}  floor={FLOOR}\n")
    print(f"  correct hits      {tp:2d}/{n_pos}   ({tp / n_pos:.0%} of answerable queries)")
    print(f"  FALSE HITS        {fp:2d}      <- wrong answers served")
    print(f"  misses            {fn:2d}      <- fell through to the model")
    print(f"  correct rejects   {tn:2d}/{len(NEGATIVE_QUERIES)}   "
          f"({tn / len(NEGATIVE_QUERIES):.0%} of traps refused)")
    print(f"\n  p50 verify latency  {sorted(t_verify)[len(t_verify) // 2]:.1f}ms")

    if false_hits:
        print("\n  false hits:")
        for q, s, tag in false_hits:
            print(f"    [{tag}] {q!r}\n          served: {s!r}")
    if misses:
        print("\n  misses (safe -- just cost a generation):")
        for q, t, tag in misses:
            print(f"    [{tag}] {q!r}")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "p50_verify_ms": sorted(t_verify)[len(t_verify) // 2]}


if __name__ == "__main__":
    run()
