"""The expensive thing the cache exists to avoid.

Default backend is `fake`: it sleeps to imitate real latency and returns a
canned answer, so the whole system runs end-to-end with no API key. Set
LLM_BACKEND=anthropic + ANTHROPIC_API_KEY when you want real answers.
"""
import config  # noqa: F401  -- loads .env, must come first
import hashlib
import os
import time

BACKEND = os.getenv("LLM_BACKEND", "fake")
MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5")

# Claude Sonnet 5 list price, $2.00 / $10.00 per million tokens.
PRICE_IN, PRICE_OUT = 2.00 / 1e6, 10.00 / 1e6


def complete(prompt: str) -> dict:
    """Returns {answer, tokens_in, tokens_out, cost, latency_ms}."""
    t0 = time.perf_counter()

    if BACKEND == "fake":
        # Deterministic per prompt, and slow enough to make the cache visible.
        time.sleep(0.8)
        digest = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        answer = f"[fake answer {digest}] {prompt.strip().capitalize()} -- here are the steps..."
        tin, tout = len(prompt) // 4, len(answer) // 4

    elif BACKEND == "anthropic":
        # TODO: uv pip install anthropic
        from anthropic import Anthropic
        msg = Anthropic().messages.create(
            model=MODEL, max_tokens=512, messages=[{"role": "user", "content": prompt}]
        )
        answer = msg.content[0].text
        tin, tout = msg.usage.input_tokens, msg.usage.output_tokens

    else:
        raise ValueError(f"unknown LLM_BACKEND: {BACKEND}")

    return {
        "answer": answer,
        "tokens_in": tin,
        "tokens_out": tout,
        "cost": tin * PRICE_IN + tout * PRICE_OUT,
        "latency_ms": (time.perf_counter() - t0) * 1000,
    }
