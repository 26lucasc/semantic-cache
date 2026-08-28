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


# Nouns that name a per-user fact rather than a topic. The answer to a
# question about one of these is different for every user, so a shared cache
# entry is a data leak, not a stale answer.
_PERSONAL_NOUNS = (
    "balance", "invoice", "invoices", "receipt", "receipts", "order", "orders",
    "appointment", "appointments", "ticket", "tickets", "usage", "quota",
    "subscription", "plan", "payment", "payments", "refund", "card",
    "address", "email", "password", "api key", "api keys", "token",
    "account", "profile", "settings", "history", "data",
)

# Predicates that name a per-user fact without using a noun from the list
# above -- "how much do I owe" has no personal NOUN but is plainly personal.
_PERSONAL_PREDICATES = ("owe", "spent", "paid", "signed up", "am i on",
                        "do i have", "have i used", "am i charged")

# First-person markers. "my balance" is personal; "the balance" is not.
_FIRST_PERSON = (" my ", " mine", " i ", " me ", " our ", " we ")

# Instructional framings. "how do I check my balance" wants the PROCEDURE,
# which is identical for every user and therefore safe to cache. Only a
# question asking for the VALUE is personal. This distinction is the whole
# point -- a naive `"my" in prompt` check refuses to cache both.
_INSTRUCTIONAL = (
    "how do i", "how to", "how can i", "how would i", "where do i",
    "where can i", "what is the way", "what\'s the way", "steps to",
    "the process", "the procedure", "the setup", "can i ", "is it possible",
    "am i able", "do you support", "does it support",
)


def is_cacheable(prompt: str) -> bool:
    """False for prompts whose answer is specific to one user.

    A cache entry is shared by construction, so caching "what is my balance"
    serves one user's number to the next person who asks something similar.
    That is a data leak with a plausible-looking answer attached -- worse than
    a wrong answer, because nothing about it looks wrong.

    This is a keyword heuristic and it WILL have both false positives and false
    negatives; see eval_cacheable.py for the measured boundary. The real fix is
    per-user cache keys (`tenant=user_id`), which the store already supports --
    then personal answers are cached but never shared. Use both: partitioning
    for correctness, this for defence in depth when a caller forgets to pass a
    tenant.
    """
    p = f" {prompt.lower().strip()} "

    if not any(m in p for m in _FIRST_PERSON):
        return True                                  # nothing personal claimed
    if not (any(f" {n} " in p or f" {n}?" in p for n in _PERSONAL_NOUNS)
            or any(v in p for v in _PERSONAL_PREDICATES)):
        return True                                  # personal, but no user data
    # Reached only when the prompt claims something personal AND names user
    # data. Cacheable only if it asks HOW rather than WHAT -- a procedure is
    # the same for everyone, a value is not.
    return any(i in p for i in _INSTRUCTIONAL)


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
        if layer == "bypass":
            pass                                     # baseline: never populate
        elif not is_safe_to_cache(result["answer"]):
            metrics.bump("poisoned_skips")           # an error response
        elif not is_cacheable(prompt):
            metrics.bump("personal_skips")           # per-user data; see eval_cacheable.py
        else:
            if vector is None:
                vector = embed(prompt)
            self.store.add(tenant, prompt, query_hash(prompt), result["answer"], vector)
        return {"answer": result["answer"], "cached": False, "layer": layer,
                "similarity": similarity, "matched_prompt": None,
                "cost": result["cost"],
                "latency_ms": (time.perf_counter() - t0) * 1000}

    def _fresh(self, entry: dict) -> bool:
        return (time.time() - entry["created_at"]) < TTL_SECONDS

