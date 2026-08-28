"""Stage 2: decide whether a retrieved candidate really means the same thing.

WHY A SECOND MODEL AT ALL. The embedder compresses each text into a fixed
vector WITHOUT knowing what it will be compared against, so it keeps the
features generally useful against an arbitrary counterpart -- which is topic.
"not" is a low-magnitude feature with total consequence, so it gets discarded.
A cross-encoder (or an LLM) sees both texts jointly and can attend across
them, so "not" in B can attend to its absence in A.

Measured on our own pairs:

    pair                                       cosine   cross-encoder
    'free plan include' / 'free plan NOT ...'   0.976       0.001
    'cancel subscription' / 'upgrade ...'       0.933       0.003
    'reset my password' / 'i forgot my ...'     0.351       0.962

Cosine ranks the two wrong answers ABOVE the right one. The cross-encoder
inverts all three. That is the entire argument for two stages.

MEASURED FAILURE -- READ BEFORE TRUSTING THE LOCAL BACKEND.

The numbers above come from eval_pairs.py, which was written by the same
person who then tuned ACCEPT against it. On eval_heldout.py -- 25 paraphrase
pairs written afterwards by someone else -- the local cross-encoder recovers
1/25 (4%). It scores 0.000 on pairs like:

    'Does it support SSO?' / 'Can I log in with single sign-on?'
    'Why is latency high?' / 'What is making my requests slow?'

Three local cross-encoders were tried (quora-distilroberta, stsb-roberta,
ms-marco-MiniLM). Best clean cutoff on held-out data recovers 4-8%. This is
not a tuning problem -- these models pattern-match their training
distribution rather than reasoning about meaning, and product-support phrasing
is out of it.

The ARCHITECTURE is unaffected: retrieval still hits recall@5 = 100% on the
same held-out set, so the right candidate reaches the verifier every time. It
is stage 2's model that is wrong, not the split.

Backends:
  crossencoder  local, free, ~15ms      runs with no API key. MEASURED at 4%
                                        recovery on held-out data. Use it to
                                        run the pipeline, not in production.
  anthropic     Claude, ~400ms, $0.0003 RECOMMENDED. An LLM reads the pair
                                        instead of pattern-matching. Untested
                                        here -- no API key available. Run
                                        eval_heldout.py with
                                        VERIFY_BACKEND=anthropic to check.
"""
import config  # noqa: F401  -- loads .env, must come first
import functools
import os

BACKEND = os.getenv("VERIFY_BACKEND", "crossencoder")
CE_MODEL = os.getenv("VERIFY_MODEL", "cross-encoder/quora-distilroberta-base")

# Asymmetric on purpose. A false hit serves a confidently wrong answer; a false
# miss costs one generation. When unsure, say no.
ACCEPT = float(os.getenv("VERIFY_ACCEPT", "0.97"))  # lowest clean row in eval_two_stage.py


@functools.lru_cache(maxsize=1)
def _ce():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(CE_MODEL)


PROMPT = """You decide whether a cached answer can be reused.

A user asked: {query}

Candidate cached questions:
{candidates}

Return the number of the candidate that has THE SAME ANSWER as the user's
question, or 0 if none does.

Be strict. Questions that differ by negation ("is X included" vs "what is not
included"), by direction ("transfer to me" vs "transfer to someone else"), by
opposite action (cancel vs upgrade), or by a different specific (Pro vs
Enterprise) do NOT have the same answer. When uncertain, answer 0."""


def verify(query: str, candidates: list[str]) -> int | None:
    """Return the index of the candidate that truly matches, or None.

    Takes ALL candidates in one call. Judging them one at a time would multiply
    cost and latency by k for no benefit.
    """
    if not candidates:
        return None

    if BACKEND == "crossencoder":
        scores = _ce().predict([(query, c) for c in candidates])
        best = max(range(len(candidates)), key=lambda i: scores[i])
        return best if scores[best] >= ACCEPT else None

    if BACKEND == "anthropic":
        import anthropic
        numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(candidates))
        msg = anthropic.Anthropic().messages.create(
            model=os.getenv("VERIFY_LLM", "claude-haiku-4-5"),
            max_tokens=8,
            messages=[{"role": "user",
                       "content": PROMPT.format(query=query, candidates=numbered)}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text").strip()
        try:
            pick = int("".join(ch for ch in text if ch.isdigit()) or 0)
        except ValueError:
            return None
        return pick - 1 if 1 <= pick <= len(candidates) else None

    raise ValueError(f"unknown VERIFY_BACKEND: {BACKEND}")
