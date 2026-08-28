"""Measure recall@k -- the go/no-go for the two-stage design.

Run:  .venv/bin/python recall.py
"""
import numpy as np

from embedder import BACKEND, LOCAL_MODEL, embed_many
from eval_retrieval import CORPUS, NEGATIVE_QUERIES, POSITIVE_QUERIES
from normalize import query_hash

KS = (1, 3, 5, 10, 20)


def rank_all():
    """For each positive query, where does its correct entry rank?"""
    cvecs = embed_many(CORPUS)
    qvecs = embed_many([q for q, _, _ in POSITIVE_QUERIES])
    idx = {c: i for i, c in enumerate(CORPUS)}

    rows = []
    for (query, target, tag), qv in zip(POSITIVE_QUERIES, qvecs):
        sims = cvecs @ qv
        order = np.argsort(-sims)                       # best first
        rank = int(np.where(order == idx[target])[0][0]) + 1
        rows.append({"query": query, "target": target, "tag": tag,
                     "rank": rank, "sim": float(sims[idx[target]]),
                     "top_sim": float(sims[order[0]]),
                     "top": CORPUS[order[0]]})
    return rows


def recall_table(rows):
    print(f"\nbackend={BACKEND} model={LOCAL_MODEL}")
    print(f"corpus={len(CORPUS)}  positive queries={len(rows)}\n")
    print("     k   recall@k")
    print("-" * 22)
    for k in KS:
        hit = sum(1 for r in rows if r["rank"] <= k)
        print(f"  {k:4d}   {hit / len(rows):7.0%}  ({hit}/{len(rows)})")


def by_family(rows):
    print("\n  family        n   recall@5   median rank")
    print("-" * 46)
    fams = {}
    for r in rows:
        fams.setdefault(r["tag"], []).append(r)
    for tag, rs in sorted(fams.items(), key=lambda kv: -sum(
            1 for r in kv[1] if r["rank"] <= 5) / len(kv[1])):
        hit = sum(1 for r in rs if r["rank"] <= 5)
        med = sorted(r["rank"] for r in rs)[len(rs) // 2]
        print(f"  {tag:12s} {len(rs):2d}   {hit / len(rs):7.0%}   {med:6d}")


def failures(rows, k=5):
    bad = [r for r in rows if r["rank"] > k]
    if not bad:
        print(f"\nno positive query failed to surface its target in top-{k}")
        return
    print(f"\nqueries whose target did NOT reach top-{k} "
          f"(unrecoverable by any verifier):")
    print("-" * 46)
    for r in sorted(bad, key=lambda r: -r["rank"]):
        print(f"  rank {r['rank']:3d}  [{r['tag']}]  {r['query']!r}")
        print(f"            wanted: {r['target']!r} (sim {r['sim']:.3f})")
        print(f"            got #1: {r['top']!r} (sim {r['top_sim']:.3f})")


def verifier_load(k=5):
    """How many candidates would the verifier judge, and how many are traps?

    Every query sends k candidates. For should-serve-nothing queries, ALL k are
    wrong by construction -- the verifier must reject the whole slate. That is
    the workload, and it is why the verifier must default to "no".
    """
    n = len(POSITIVE_QUERIES) + len(NEGATIVE_QUERIES)
    print(f"\nverifier workload at k={k}: {n} calls, {k} candidates each")
    print(f"  {len(POSITIVE_QUERIES)} queries have exactly 1 correct candidate")
    print(f"  {len(NEGATIVE_QUERIES)} queries have 0 -- the whole slate must be rejected")


if __name__ == "__main__":
    rows = rank_all()
    recall_table(rows)
    by_family(rows)
    failures(rows)
    verifier_load()
