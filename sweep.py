"""Threshold sweep: turns 'what should THRESHOLD be' into a measurement.

Run:  .venv/bin/python sweep.py

Read the output with this asymmetry in mind:
  a FALSE MISS costs one wasted API call (~$0.002, nobody notices)
  a FALSE HIT  serves a confidently wrong answer to a user
They are not equally bad, so do NOT pick the row with the best accuracy.
Pick the lowest threshold with zero false hits, then back off one notch.
"""
import numpy as np

from embedder import BACKEND, LOCAL_MODEL, embed_many
from eval_pairs import PAIRS
from normalize import query_hash


def score_pairs():
    """Embed every pair once, return [(similarity, should_match, tag, a, b)]."""
    texts = [t for a, b, _, _ in PAIRS for t in (a, b)]
    vecs = embed_many(texts)                      # one batched call, not 100
    return [
        (float(np.dot(vecs[2 * i], vecs[2 * i + 1])), should_match, tag, a, b)
        for i, (a, b, should_match, tag) in enumerate(PAIRS)
    ]


def layer1(scored):
    """How much does the exact-match layer catch before the threshold matters?

    Layer 1 hits are free and carry NO false-hit risk, so every pair it catches
    is removed from the threshold decision entirely. Returns the pairs Layer 2
    still has to judge.
    """
    caught = [r for r in scored if query_hash(r[3]) == query_hash(r[4])]
    remaining = [r for r in scored if query_hash(r[3]) != query_hash(r[4])]
    pos = sum(1 for r in caught if r[1])
    neg = sum(1 for r in caught if not r[1])
    print(f"\nLayer 1 (normalized hash) catches {len(caught)}/{len(scored)} pairs "
          f"-- {pos} positive, {neg} negative")
    if neg:
        print("  !! a negative matched by hash means normalize() is too aggressive")
    print(f"  these are free and cannot be wrong; Layer 2 judges the other {len(remaining)}")
    return remaining


def combined(scored, lo=0.90, hi=1.00, step=0.02):
    """What the whole system does: Layer 1 hits + Layer 2 hits.

    Layer 2's table looks WORSE after Layer 1 is added, because Layer 1 removed
    the easiest positives from its pool. That is not a regression -- read this
    table, not that one.
    """
    l1 = [r for r in scored if query_hash(r[3]) == query_hash(r[4])]
    l2 = [r for r in scored if query_hash(r[3]) != query_hash(r[4])]
    l1_hits = sum(1 for r in l1 if r[1])
    l1_bad = sum(1 for r in l1 if not r[1])
    n_pos = sum(1 for r in scored if r[1])

    print("\n=== whole system (Layer 1 + Layer 2) ===")
    print(f"Layer 1 contributes {l1_hits} hits, {l1_bad} false hits, at no cost\n")
    print("thresh   exact  semantic  TOTAL  FALSE_HITS   hit_rate")
    print("-" * 56)
    for t in np.arange(lo, hi, step):
        tp = sum(1 for s_, m, *_ in l2 if s_ >= t and m)
        fp = sum(1 for s_, m, *_ in l2 if s_ >= t and not m)
        total = l1_hits + tp
        flag = "  <-- ship this" if fp + l1_bad == 0 else ""
        print(f" {t:.2f}   {l1_hits:5d}  {tp:8d}  {total:5d}  {fp + l1_bad:10d}   "
              f"{total / n_pos:6.0%}{flag}")


def sweep(scored, lo=0.60, hi=1.00, step=0.02):
    print(f"\nbackend={BACKEND} model={LOCAL_MODEL}  n={len(scored)}\n")
    print("thresh   hits  FALSE_HITS  misses   hit_rate")
    print("-" * 48)
    for t in np.arange(lo, hi, step):
        tp = sum(1 for s, m, *_ in scored if s >= t and m)
        fp = sum(1 for s, m, *_ in scored if s >= t and not m)
        fn = sum(1 for s, m, *_ in scored if s < t and m)
        n_pos = tp + fn
        flag = "  <-- first clean row" if fp == 0 else ""
        print(f" {t:.2f}   {tp:4d}  {fp:10d}  {fn:6d}   {tp / n_pos:6.0%}{flag}")


def by_tag(scored):
    """Mean similarity per rewording style / negative family.

    This is the interesting table: if your hardest positives (reframe) overlap
    your hardest negatives (antonym, negation), no single threshold separates
    them and you need the two-stage design described in the README.
    """
    groups = {}
    for s, m, tag, *_ in scored:
        groups.setdefault((m, tag), []).append(s)
    print("\n  match  family        n   mean    min    max")
    print("-" * 48)
    for (m, tag), vals in sorted(groups.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"  {str(m):5s}  {tag:12s} {len(vals):2d}  "
              f"{np.mean(vals):.3f}  {min(vals):.3f}  {max(vals):.3f}")


def worst_offenders(scored, k=8):
    """The negatives that scored highest -- these are what a loose threshold serves."""
    bad = sorted([r for r in scored if not r[1]], key=lambda r: -r[0])[:k]
    print(f"\nhighest-scoring NEGATIVES (the dangerous ones):")
    print("-" * 48)
    for s, _, tag, a, b in bad:
        print(f"  {s:.3f}  [{tag}]\n         {a!r}\n         {b!r}")


def hardest_positives(scored, k=6):
    """The positives that scored lowest -- these are what a tight threshold drops."""
    bad = sorted([r for r in scored if r[1]], key=lambda r: r[0])[:k]
    print(f"\nlowest-scoring POSITIVES (what you lose at a high threshold):")
    print("-" * 48)
    for s, _, tag, a, b in bad:
        print(f"  {s:.3f}  [{tag}]\n         {a!r}\n         {b!r}")


if __name__ == "__main__":
    scored = score_pairs()
    remaining = layer1(scored)
    print("\n=== Layer 2 only (pairs Layer 1 did not already catch) ===")
    sweep(remaining)
    combined(scored)
    by_tag(scored)
    worst_offenders(scored)
    hardest_positives(scored)
