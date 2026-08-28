"""Labeled query pairs for tuning the semantic cache threshold.

Domain: customer support for a project-management SaaS. Threshold values do
NOT transfer across domains -- if you change the product, rewrite these pairs.

Each entry is (query_a, query_b, should_match, tag).
  should_match=True   -> paraphrases; the cache SHOULD serve a stored answer
  should_match=False  -> different intent; serving a stored answer is a bug
"""

# ---------------------------------------------------------------------------
# POSITIVES: 5 base questions x 5 rewording styles.
# The styles are ordered roughly easiest -> hardest. The "reframe" rows are the
# ones a too-high threshold will kill, so they have to be in here.
# ---------------------------------------------------------------------------
POSITIVES = [
    # 1. password reset
    ("how do I reset my password", "How do I reset my password?", "punctuation"),
    ("how do I reset my password", "password reset how", "shortened"),
    ("how do I reset my password", "how do I change my login credentials", "synonym"),
    ("how do I reset my password", "i can't log in, need a new password", "reframe"),
    ("how do I reset my password", "how do i rest my pasword", "typos"),

    # 2. inviting a teammate
    ("how do I invite a teammate", "How do I invite a teammate?", "punctuation"),
    ("how do I invite a teammate", "invite teammate steps", "shortened"),
    ("how do I invite a teammate", "how do I add a new member to my workspace", "synonym"),
    ("how do I invite a teammate", "my coworker needs access to our projects", "reframe"),
    ("how do I invite a teammate", "how do i invte a teamate", "typos"),

    # 3. plan contents
    ("what does the Pro plan include", "What does the Pro plan include?", "punctuation"),
    ("what does the Pro plan include", "Pro plan features list", "shortened"),
    ("what does the Pro plan include", "what features come with Pro", "synonym"),
    ("what does the Pro plan include", "im thinking about Pro, what do i get", "reframe"),
    ("what does the Pro plan include", "what does the Pro plna inclue", "typos"),

    # 4. data export
    ("how do I export my data", "How do I export my data?", "punctuation"),
    ("how do I export my data", "data export how to", "shortened"),
    ("how do I export my data", "how do I download all my information", "synonym"),
    ("how do I export my data", "i want to take my stuff out of the app", "reframe"),
    ("how do I export my data", "how do i exprot my dat", "typos"),

    # 5. performance complaint
    ("why is the app slow", "Why is the app so slow?", "punctuation"),
    ("why is the app slow", "app slow performance", "shortened"),
    ("why is the app slow", "the application is laggy for me", "synonym"),
    ("why is the app slow", "everything takes forever to load", "reframe"),
    ("why is the app slow", "why is the ap slwo", "typos"),
]

# ---------------------------------------------------------------------------
# HARD NEGATIVES: near-identical wording, different intent. These are the whole
# point of the eval -- random unrelated negatives prove nothing.
# ---------------------------------------------------------------------------
NEGATIVES = [
    # -- opposite action: one verb flips the meaning ------------------------
    ("how do I cancel my subscription", "how do I upgrade my subscription", "antonym"),
    ("how do I archive a project", "how do I unarchive a project", "antonym"),
    ("how do I mute notifications", "how do I enable notifications", "antonym"),
    ("how do I lock a task", "how do I unlock a task", "antonym"),
    ("how do I hide completed tasks", "how do I show completed tasks", "antonym"),

    # -- negation: embeddings are famously weak at "not" --------------------
    ("is the Slack integration included in Pro", "what is not included in Pro", "negation"),
    ("which file types can I upload", "which file types can't I upload", "negation"),
    ("what does the free plan include", "what does the free plan not include", "negation"),
    ("can I recover deleted tasks", "which deleted tasks cannot be recovered", "negation"),
    ("is billing prorated", "when is billing not prorated", "negation"),

    # -- direction / role swap: one preposition flips who does what ---------
    ("how do I remove someone from my team", "how do I leave a team", "direction"),
    ("how do I transfer ownership to me", "how do I transfer ownership to someone else", "direction"),
    ("how do I share a project with a client", "how do I get a project shared with me", "direction"),
    ("how do I import tasks from Jira", "how do I export tasks to Jira", "direction"),
    ("how do I assign a task to a teammate", "how do I get a task assigned to me", "direction"),

    # -- same topic, different specific: the most common real-world case ----
    ("what does the Pro plan cost", "what does the Enterprise plan cost", "specific"),
    ("how do I export to CSV", "how do I import from CSV", "specific"),
    ("how do I reset my password", "how do I reset my API key", "specific"),
    ("how do I delete a task", "how do I delete my account", "specific"),
    ("how do I change my email address", "how do I change my billing address", "specific"),

    # -- easy negatives: sanity check. If these score > 0.5 your embedder is
    #    misconfigured (wrong model, empty strings, etc).
    ("how do I reset my password", "what's the weather in Tokyo", "unrelated"),
    ("how do I invite a teammate", "best recipe for banana bread", "unrelated"),
    ("what does the Pro plan include", "who won the world cup in 1998", "unrelated"),
    ("how do I export my data", "how tall is Mount Everest", "unrelated"),
    ("why is the app slow", "translate hello into French", "unrelated"),
]

# Unified list the sweep consumes: (a, b, should_match, tag)
PAIRS = (
    [(a, b, True, tag) for a, b, tag in POSITIVES]
    + [(a, b, False, tag) for a, b, tag in NEGATIVES]
)

if __name__ == "__main__":
    n_pos = sum(1 for *_, m, _ in [(p[0], p[1], p[2], p[3]) for p in PAIRS] if m)
    print(f"{len(PAIRS)} pairs: {n_pos} positive, {len(PAIRS) - n_pos} negative")
