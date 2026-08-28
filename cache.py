"""Layered cache: exact match -> semantic match -> LLM.

The layering is the point. Each layer costs more than the last, so you only
escalate when the cheap one misses:

  Layer 1  normalized hash lookup   ~0.1ms   no embedding, NO false-hit risk
  Layer 2  cosine similarity        ~10ms    one embedding call, threshold gamble
  Layer 3  the model                ~800ms+  costs money

Layer 1 matters more than it looks. In sweep.py the `punctuation` family
scores 0.989 -- the highest of any group -- which means those queries were
being paid for with an embedding call AND a threshold bet when a hash lookup
would have caught them for free and for certain.
"""
import config  # noqa: F401  -- loads .env, must come first
import os
import time

import llm
import metrics
from embedder import embed
from normalize import query_hash
from store import Store
from verifier import verify

# Set this from sweep.py output. Do NOT pick the most "accurate" row -- a false
# hit serves a confidently wrong answer, a false miss costs $0.002. Pick the
# lowest threshold with zero false hits, then back off one notch.
THRESHOLD = float(os.getenv("CACHE_THRESHOLD", "0.98"))

# Two-stage retrieval. ON by default: eval_two_stage.py measures 40% hit rate
# at zero false hits, against 20% for the single-threshold design.
#
# Note what changes when this is on: THRESHOLD stops being the decision and
# becomes a floor. Retrieval should over-suggest -- the verifier throws out
# what does not belong, and anything retrieval drops is unrecoverable. This
# inverts the instinct to tighten the threshold when you see bad hits.
VERIFY = os.getenv("CACHE_VERIFY", "1") == "1"
TOP_K = int(os.getenv("CACHE_TOP_K", "5"))
FLOOR = float(os.getenv("CACHE_FLOOR", "0.15"))

# TTL is a HARD filter, deliberately not blended into the similarity score.
# A reference implementation multiplies them (confidence = similarity x
# freshness) and gates at 0.75 -- which silently makes a valid 0.90 match
# unusable after 17% of its TTL. Worse, it conflates two unrelated things: a
# semantically wrong match does not become right by being fresh.
TTL_SECONDS = float(os.getenv("CACHE_TTL", "3600"))

_ERROR_PREFIXES = ("[LLM Error]", "[Error]", "[ERROR]")


def is_safe_to_cache(answer: str) -> bool:
    """Cache poisoning guard. If the model 500s and you store the error string,
    you then serve that error to every similar query until it expires."""
    if not answer or not answer.strip():
        return False
    return not answer.startswith(_ERROR_PREFIXES)


def is_cacheable(prompt: str) -> bool:
    """TODO(you): some prompts must never be shared across users.
    'what is my account balance' is the obvious one. A keyword denylist is the
    start; per-user cache keys (tenant=user_id) are the real fix, and the store
    already supports that."""
    return True


class SemanticCache:
    def __init__(self, store: Store | None = None):
        self.store = store or Store()

    def ask(self, prompt: str, tenant: str = "default", bypass: bool = False) -> dict:
        t0 = time.perf_counter()
        ms = lambda: (time.perf_counter() - t0) * 1000

        if bypass:
            metrics.bump("bypasses")
            return self._generate(prompt, tenant, vector=None, t0=t0, layer="bypass")

        # ---- Layer 1: exact match ------------------------------------
        qhash = query_hash(prompt)
        entry_id = self.store.get_by_hash(tenant, qhash)
        if entry_id is not None:
            entry = self.store.get(entry_id)
            if self._fresh(entry):
                self.store.record_hit(entry_id)
                metrics.bump("exact_hits")
                return {"answer": entry["answer"], "cached": True, "layer": "exact",
                        "similarity": 1.0, "matched_prompt": entry["prompt"],
                        "cost": 0.0, "latency_ms": ms()}

        # ---- Layer 2: semantic match ---------------------------------
        vector = embed(prompt)
        if VERIFY:
            cands = [(i, s) for i, s in self.store.search_topk(tenant, vector, TOP_K, FLOOR)
                     if self._fresh(self.store.get(i))]
            score = cands[0][1] if cands else 0.0
            if cands:
                pick = verify(prompt, [self.store.get(i)["prompt"] for i, _ in cands])
                if pick is not None:
                    entry_id, score = cands[pick]
                    entry = self.store.get(entry_id)
                    self.store.record_hit(entry_id)
                    metrics.bump("semantic_hits")
                    return {"answer": entry["answer"], "cached": True, "layer": "semantic",
                            "similarity": score, "matched_prompt": entry["prompt"],
                            "cost": 0.0, "latency_ms": ms()}
        else:
            entry_id, score = self.store.search(tenant, vector)
            if entry_id is not None and score >= THRESHOLD:
                entry = self.store.get(entry_id)
                if self._fresh(entry):
                    self.store.record_hit(entry_id)
                    metrics.bump("semantic_hits")
                    return {"answer": entry["answer"], "cached": True, "layer": "semantic",
                            "similarity": score, "matched_prompt": entry["prompt"],
                            "cost": 0.0, "latency_ms": ms()}

        # ---- Layer 3: the model --------------------------------------
        metrics.bump("misses")
        return self._generate(prompt, tenant, vector, t0, layer="llm", similarity=score)

    def _generate(self, prompt, tenant, vector, t0, layer, similarity=0.0):
        result = llm.complete(prompt)
        # A bypassed request must NOT populate the cache. It is the no-cache
        # baseline; storing would make it pay embedding and write costs the
        # baseline does not actually have, and corrupt the comparison.
        if layer != "bypass" and is_safe_to_cache(result["answer"]) and is_cacheable(prompt):
            if vector is None:
                vector = embed(prompt)
            self.store.add(tenant, prompt, query_hash(prompt), result["answer"], vector)
        else:
            metrics.bump("poisoned_skips")
        return {"answer": result["answer"], "cached": False, "layer": layer,
                "similarity": similarity, "matched_prompt": None,
                "cost": result["cost"],
                "latency_ms": (time.perf_counter() - t0) * 1000}

    def _fresh(self, entry: dict) -> bool:
        return (time.time() - entry["created_at"]) < TTL_SECONDS

