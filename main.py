"""HTTP front door.  Run:  .venv/bin/uvicorn main:app --reload"""
import metrics
from fastapi import FastAPI
from pydantic import BaseModel, field_validator

from cache import THRESHOLD, TTL_SECONDS, SemanticCache

app = FastAPI(title="semantic cache")
cache = SemanticCache()


class Ask(BaseModel):
    prompt: str
    tenant: str = "default"
    bypass_cache: bool = False   # skip both layers; used for baseline measurement

    @field_validator("prompt")
    @classmethod
    def not_blank(cls, v: str) -> str:
        """Reject junk before it costs an embedding call or pollutes the cache."""
        if not v.strip():
            raise ValueError("prompt cannot be empty")
        return v.strip()


@app.post("/v1/ask")
def ask(req: Ask):
    return cache.ask(req.prompt, req.tenant, bypass=req.bypass_cache)


@app.get("/v1/stats")
def stats():
    return {"threshold": THRESHOLD, "ttl_seconds": TTL_SECONDS,
            **cache.store.stats(), **metrics.snapshot()}
