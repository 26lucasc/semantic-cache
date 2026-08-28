"""Labeled prompts for is_cacheable(). The boundary is the interesting part.

The easy calls are "what is my balance" (never cache) and "what are the rate
limits" (always cache). The hard ones sit between: "how do I check my balance"
asks for a PROCEDURE, identical for every user, and refusing to cache it
throws away real hits for no safety gain.

Run:  .venv/bin/python eval_cacheable.py
"""
from cache import is_cacheable

# (prompt, should_be_cacheable)
CASES = [
    # -- personal data: the answer is a value unique to one user -------------
    ("what is my account balance", False),
    ("how much do I owe", False),
    ("when is my appointment", False),
    ("what's my current usage", False),
    ("show me my recent orders", False),
    ("what plan am I on", False),
    ("when does my subscription renew", False),
    ("what is my API key", False),
    ("what's my email address on file", False),
    ("how many tickets do I have open", False),

    # -- procedures: identical for everyone, safe to share ------------------
    ("how do I check my balance", True),
    ("how do I cancel my subscription", True),
    ("how do I rotate my API keys", True),
    ("where do I find my org ID", True),
    ("how can I export my data", True),
    ("what's the way to update my email address", True),
    ("can I change my plan mid-cycle", True),

    # -- no personal claim at all -------------------------------------------
    ("what are the rate limits", True),
    ("does it support SSO", True),
    ("what Python versions are supported", True),
    ("why is the app slow", True),
    ("where are logs stored", True),
]

if __name__ == "__main__":
    wrong = [(p, want) for p, want in CASES if is_cacheable(p) != want]
    for p, want in CASES:
        got = is_cacheable(p)
        mark = "  " if got == want else "XX"
        print(f"{mark} cacheable={str(got):5s} want={str(want):5s}  {p!r}")
    print(f"\n{len(CASES) - len(wrong)}/{len(CASES)} correct")
    if wrong:
        print("\nMISCLASSIFIED:")
        for p, want in wrong:
            kind = "LEAK (cached personal data)" if not want else "lost hit (over-refused)"
            print(f"  {kind}: {p!r}")
