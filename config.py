"""Loads .env so the modules below can read os.getenv().

Imported for its side effect by embedder.py, verifier.py, llm.py and cache.py.
Without this, a .env file sits on disk doing nothing -- which is what happened
before, while .env.example implied otherwise.
"""
from dotenv import load_dotenv

load_dotenv()   # no-op if .env is absent; never overwrites a real env var
