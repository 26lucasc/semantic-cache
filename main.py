"""HTTP front door.  Run:  .venv/bin/uvicorn main:app --reload"""
from pathlib import Path

import metrics
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
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


_DEMO = Path(__file__).parent / "demo.html"


@app.get("/", include_in_schema=False)
def root():
    """A page a human can use.

    Without this the bare domain returned FastAPI's {"detail":"Not Found"},
    which reads as a broken deployment. Swagger at /docs works but asks a
    visitor to hand-edit JSON; this shows the layer, the similarity score and
    the matched prompt, which is the part worth seeing.
    """
    if _DEMO.exists():
        return HTMLResponse(_DEMO.read_text(encoding="utf-8"))
    return RedirectResponse("/docs")   # fall back if the file did not ship


@app.post("/v1/ask")
def ask(req: Ask):
    return cache.ask(req.prompt, req.tenant, bypass=req.bypass_cache)


@app.get("/health")
def health():
    """Liveness probe for the host. Deliberately does not touch the models."""
    return {"ok": True}


@app.get("/v1/stats")
def stats():
    return {"threshold": THRESHOLD, "ttl_seconds": TTL_SECONDS,
            **cache.store.stats(), **metrics.snapshot()}
