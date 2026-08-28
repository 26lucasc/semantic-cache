"""Retrieval-shaped eval: measures recall@k, not pairwise similarity.

WHY THIS EXISTS. sweep.py asks "does this pair score above t" -- a pairwise
question. A two-stage design asks a different one: "for this query, is the
right cache entry among the top k candidates, at ANY score?" Those come apart
completely. An entry can rank #1 for a query and still sit at 0.4 absolute
similarity, which the threshold rejects and a top-k retriever keeps.

recall@k is the go/no-go for the two-stage design. The verifier can only
reject bad candidates -- it can never recover an entry retrieval failed to
surface. If recall@5 is high, verification has something to work with. If it
is low, no verifier quality saves the design.

CORPUS: what is sitting in the cache.
QUERIES: what users type, each with the entry that SHOULD be served, or None
         when nothing in the corpus answers it.
"""
from eval_pairs import NEGATIVES, POSITIVES

# The 5 canonical questions whose answers are cached.
BASES = sorted({a for a, _, _ in POSITIVES})

# Traps: plausible questions that are lexically near a query but mean something
# different. Only the A side is cached -- the B side is what the user types and
# must NOT be in the corpus, or the test is trivially satisfied by an exact
# match rather than by the verifier doing its job.
TRAPS = sorted({a for a, _, _ in NEGATIVES})

CORPUS = BASES + [t for t in TRAPS if t not in BASES]

# Queries that SHOULD retrieve their base question.
POSITIVE_QUERIES = [(variant, base, tag) for base, variant, tag in POSITIVES]

# Queries where the correct answer is "serve nothing". Each is lexically close
# to a trap already in the corpus, so retrieval will surface that trap -- the
# verifier's job is to reject it. Retrieval surfacing it is NOT an error.
NEGATIVE_QUERIES = [(b, None, tag) for _, b, tag in NEGATIVES]

QUERIES = POSITIVE_QUERIES + NEGATIVE_QUERIES

if __name__ == "__main__":
    print(f"corpus  {len(CORPUS)} entries ({len(BASES)} answerable + "
          f"{len(CORPUS) - len(BASES)} traps)")
    print(f"queries {len(QUERIES)} ({len(POSITIVE_QUERIES)} answerable, "
          f"{len(NEGATIVE_QUERIES)} should-serve-nothing)")
