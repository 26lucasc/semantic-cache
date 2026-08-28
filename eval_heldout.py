"""Held-out paraphrase pairs -- written by someone else, AFTER the verifier
   was tuned. This is the set that caught the overfit.

   eval_pairs.py was written by the same person who then tuned VERIFY_ACCEPT
   against it. That is a closed loop, and it produced a number (40% hits, 0
   false hits) that did not survive contact with new data: 1/25 here.

   Run:  .venv/bin/python eval_heldout.py
"""
import numpy as np
from embedder import embed_many
from normalize import query_hash
from sentence_transformers import CrossEncoder

PAIRS = [
 ("How do I enable TLS verification?", "What's the way to turn on TLS verification?"),
 ("How do I reset my password?", "I forgot my password, how do I change it?"),
 ("What's the max file upload size?", "How large can an uploaded file be?"),
 ("How do I cancel my subscription?", "What's the process for ending my plan?"),
 ("Why is my build failing?", "My build keeps erroring out, what's wrong?"),
 ("How do I add a team member?", "What's the way to invite someone to my workspace?"),
 ("Where are logs stored?", "What's the location of the log files?"),
 ("How do I export data as CSV?", "Can I download my data in CSV format?"),
 ("What are the rate limits?", "How many requests am I allowed per minute?"),
 ("How do I connect to Postgres?", "What's the setup for a Postgres connection?"),
 ("Is there a free tier?", "Do you offer a plan that costs nothing?"),
 ("How do I rotate API keys?", "What's the procedure for cycling my API credentials?"),
 ("Why am I getting a 403?", "What causes a forbidden error response?"),
 ("How do I set up webhooks?", "What's involved in configuring webhook delivery?"),
 ("Can I use this offline?", "Does it work without an internet connection?"),
 ("How do I upgrade my plan?", "What's the way to move to a higher tier?"),
 ("What Python versions are supported?", "Which releases of Python does this work with?"),
 ("How do I delete my account?", "What's the process to remove my account permanently?"),
 ("Where do I find my org ID?", "How do I look up my organization identifier?"),
 ("How do I schedule a recurring job?", "What's the setup for a job that runs on a schedule?"),
 ("Does it support SSO?", "Can I log in with single sign-on?"),
 ("How do I roll back a deployment?", "What's the way to revert to a previous deploy?"),
 ("Why is latency high?", "What's making my requests slow?"),
 ("How do I filter results by date?", "Can I narrow the output to a date range?"),
 ("What's the retention period?", "How long is data kept before deletion?"),
]

texts = [t for p in PAIRS for t in p]
v = embed_many(texts)
cos = [float(np.dot(v[2*i], v[2*i+1])) for i in range(len(PAIRS))]
ce = CrossEncoder("cross-encoder/quora-distilroberta-base").predict(PAIRS)
l1 = [query_hash(a) == query_hash(b) for a, b in PAIRS]

print(f"\n{'cosine':>7} {'verif':>7}  {'L1':>3}  {'old':>4} {'new':>4}  pair")
print("-" * 92)
for i, (a, b) in enumerate(PAIRS):
    old = "HIT" if (l1[i] or cos[i] >= 0.98) else "-"
    new = "HIT" if (l1[i] or ce[i] >= 0.97) else "-"
    print(f"{cos[i]:7.3f} {ce[i]:7.3f}  {'Y' if l1[i] else '.':>3}  {old:>4} {new:>4}  {a[:34]:34s} | {b[:36]}")

old_hits = sum(1 for i in range(len(PAIRS)) if l1[i] or cos[i] >= 0.98)
new_hits = sum(1 for i in range(len(PAIRS)) if l1[i] or ce[i] >= 0.97)
print("-" * 92)
print(f"threshold-only design (cos>=0.98):  {old_hits:2d}/25  {old_hits/25:.0%}")
print(f"two-stage design      (verif>=0.97): {new_hits:2d}/25  {new_hits/25:.0%}")
print(f"\ncosine     mean {np.mean(cos):.3f}  min {min(cos):.3f}  max {max(cos):.3f}")
print(f"verifier   mean {np.mean(ce):.3f}  min {min(ce):.3f}  max {max(ce):.3f}")
for t in (0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99):
    print(f"  verifier >= {t:.2f}: {sum(1 for s in ce if s >= t):2d}/25")
