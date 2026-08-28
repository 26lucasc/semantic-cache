"""Query normalization + hashing for the exact-match layer (Layer 1).

Layer 1 is worth having because it is the only lookup with ZERO false-hit
risk: the strings are identical after normalization, so there is no judgment
call. It also costs no embedding call. Anything it catches never reaches the
threshold gamble in Layer 2.

Normalization decides how much it catches. Case and whitespace are obviously
safe. Trailing punctuation is safe too -- "reset my password?" and "reset my
password" are the same question. INTERNAL punctuation is NOT stripped: "can't"
and "cant" are fine to merge, but stripping all punctuation would collapse
things like "3.5" and "35".
"""
import hashlib
import re

_TRAILING = re.compile(r"[?!.,;:\s]+$")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = _TRAILING.sub("", text)      # "How do I reset my password?" -> "...password"
    text = _WS.sub(" ", text)           # collapse runs of whitespace
    return text


def query_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode()).hexdigest()
