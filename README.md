# Semantic Cache

A layered LLM cache. Three layers, escalating in cost:

```
Layer 1   normalized hash lookup   ~0.7ms    no embedding, NO false-hit risk
Layer 2   cosine similarity        ~15ms     one embedding call, threshold gamble
Layer 3   the model                ~825ms    costs money
```

The interesting part is not the cache. It is **"close enough"** -- Layer 2's
threshold, which is the only place the system can be confidently wrong.

## Run it

```bash
uv venv --python 3.12
uv pip install -r requirements.txt

python sweep.py     # pick the threshold  (no API key needed)
python replay.py    # measure the running system
uvicorn main:app --reload
```

`sweep.py` and `replay.py` produce every number in this README. Re-run them
after any change; nothing here is hand-written.

Defaults need no API keys: embeddings run locally (`all-MiniLM-L6-v2`) and
generation is a fake backend that sleeps 800ms. Set `LLM_BACKEND=anthropic`
for real answers. See `.env.example`.

## Finding: no single threshold works with MiniLM

`sweep.py` scores 50 labeled pairs -- 25 paraphrases that *should* match, 25
hard negatives that *must not* -- and sweeps the threshold. Result:

```
thresh   hits  FALSE_HITS  misses   hit_rate
 0.80      8           8      17      32%
 0.90      6           4      19      24%
 0.92      5           1      20      20%
 0.98      4           0      21      16%   <- first clean row
```

There is no good row. The only threshold with zero false hits catches 16% of
real paraphrases. Grouping by rewording style shows why:

```
  match  family        n   mean
  True   punctuation   5  0.989
  False  negation      5  0.840   <-- negatives outrank...
  True   shortened     5  0.831
  False  direction     5  0.796
  False  antonym       5  0.699
  False  specific      5  0.614
  True   synonym       5  0.565   <-- ...real paraphrases
  True   reframe       5  0.492
  True   typos         5  0.364
  False  unrelated     5  0.020
```

**The classes are interleaved, not separated.** Wrong answers score higher
than right ones. Worst case:

```
0.976   'what does the free plan include'
        'what does the free plan not include'
```

Near-identical vectors, opposite meaning. Meanwhile a genuine paraphrase --
"how do I invite a teammate" / "my coworker needs access to our projects" --
scores 0.254. No scalar threshold separates 0.976 from 0.254 in the right
direction.

Two known weaknesses show up cleanly: **negation** (`not`, `can't`, `cannot`
barely move the vector) and **direction** (`to me` vs `to someone else`,
`from Jira` vs `to Jira`).

## What the layers actually contribute

Layer 1 catches **4/50 pairs and 0 false hits** -- normalization is safe. Those
are free and cannot be wrong, so they leave the threshold decision entirely.
Whole-system numbers:

```
thresh   exact  semantic  TOTAL  FALSE_HITS   hit_rate
 0.90       4         2      6           4      24%
 0.92       4         1      5           1      20%
 0.98       4         1      5           0      20%   <- ship this
```

Note Layer 2's isolated table looks *worse* after adding Layer 1 (16% -> 5%).
That is not a regression: Layer 1 took the easiest positives out of Layer 2's
pool. Read the combined table.

## Benchmark: with vs without, identical traffic

`benchmark.py`, 200 requests, 54 distinct, 30% unique never-repeating tail.
Two-stage verification on, accept 0.97.

```
                          no cache    with cache
  wall clock                160.7s         50.6s     69% less
  throughput (req/s)          1.24          3.95     3x
  mean latency               803ms         253ms     69% less
  p50 latency                804ms           1ms    100% less
  p95 latency                805ms         866ms      8% WORSE

  inference calls              200            47     76% less
  input tokens                1302           324     75% less
  output tokens               3702           888     76% less
  cost / 1k requests        $0.198        $0.048     76% less

  hit rate                       -     76%  (64% exact + 12% semantic)
  vectors in RAM                 -     70.5 KB (1536 bytes/entry)
  sqlite on disk                 -     112 KB
  model RAM                      -     532 MB
  CPU time                    0.0s     3.4s
```

**Read the three uncomfortable rows before the good ones.**

**p95 got worse.** Cache misses now pay an embedding *and* a verifier call
before the generation they were always going to make. Median latency collapses
to 1ms and the tail gets slightly slower. If you are p95-bound rather than
cost-bound, that is a real regression, not a rounding error.

**532 MB of RAM, and CPU goes from nothing to 3.4s.** The cache does not make
work disappear, it moves it: off a metered GPU and onto your own box. That is
usually the trade you want -- local CPU is far cheaper than inference -- but
"76% less compute" would be false. It is 76% less *inference*.

**Semantic matching contributes 12 of the 76 points.** Most of the win is
plain exact-match caching. The semantic layer is what the whole project is
about, and it is the minority of the benefit -- worth knowing before you
attribute the headline number to the interesting part.

Caveats on the good rows: generation is **modeled**, not measured (see the
header comment in `benchmark.py`) -- `llm.py`'s fake backend sleeps 800ms and
derives token counts from text length, priced at Sonnet 5 list. The hit rate
is real; what each hit is worth depends on your model. And 54 distinct queries
over 200 requests is a favorable repeat distribution. Rerun with
`LLM_BACKEND=anthropic` and your own traffic before quoting the cost line.

## Replay: the mechanism works

100 requests replayed through the live service at threshold 0.98:

```
LLM calls        22
hit rate         78%   (exact 74% + semantic 4%)
p50 exact hit    0.7ms
p50 semantic hit 14.5ms
p50 miss         824ms
cost             $0.0065  (no cache: ~$0.0297)
```

**Do not read 78% as a production hit rate.** The traffic is 100 draws with
replacement from a 50-string vocabulary, so exact repeats are wildly
over-represented -- that is what the 74% measures. What this run *does* prove
is that the layering works end to end and that the per-layer latencies are
what the design claims (1000x between Layer 1 and Layer 3). A real hit rate
needs real traffic, or at minimum a held-out paraphrase set with a realistic
repeat distribution. See the TODO in `replay.py`.

## Reading the sweep: the errors are not symmetric

- **False miss** -- should have matched, didn't. Costs one API call, ~$0.002.
  The user waits the normal amount of time. Nobody notices.
- **False hit** -- shouldn't have matched, did. A user asking how to *cancel*
  is told how to *upgrade*, instantly and confidently.

So the rule is not "maximize accuracy." Accuracy weights both equally and they
are not equal. Pick the lowest threshold with **zero false hits**, then back
off one notch. (Accuracy alone would pick ~0.80, which serves 8 wrong answers.)

## The fix: stop asking one number to answer two questions

A single cosine score was being asked both "is this roughly about the same
thing" and "does this mean the same thing." Those are different questions.

The reason it can only answer the first: a bi-encoder compresses each text
into a fixed vector **without knowing what it will be compared against**, so
it keeps the features generally useful against an arbitrary counterpart --
which is topic. `not` is a low-magnitude feature with total consequence, so it
gets discarded. A cross-encoder sees both texts jointly and can attend across
them, so `not` in B can attend to its absence in A.

So split the job. Retrieval **narrows** (topical similarity is exactly right
for that); a second stage **decides**.

### Measured: retrieval was never the problem

`recall.py` asks the retrieval question -- is the correct entry in the top k,
at *any* score -- against a 24-entry corpus:

```
     k   recall@k
     1       92%
     3       96%
     5      100%
```

**100% at k=5.** Every answerable query surfaces its target. The information
was there the whole time; the threshold was throwing it away. The `synonym`
family averages 0.565 cosine -- rejected by any safe threshold -- yet has
median rank 1.

### Measured: the verifier inverts the failures

The same three pairs, scored both ways:

```
pair                                        cosine   cross-encoder
'free plan include' / 'free plan NOT ...'    0.976       0.001
'cancel subscription' / 'upgrade ...'        0.933       0.003
'reset my password' / 'i forgot my ...'      0.351       0.962
```

Cosine ranks both wrong answers above the right one. The cross-encoder
inverts all three.

### Measured: end to end

`eval_two_stage.py`, 25 answerable queries + 25 that must be refused:

```
accept   hits  FALSE_HITS  rejects   hit_rate
 0.800     13           1       24       52%
 0.900     12           1       24       48%
 0.970     10           0       25       40%   <- ship this
 0.990      3           0       25       12%
```

**40% hit rate at zero false hits, against 20% for the single-threshold
design.** Both are safe; one serves twice as much.

### Then it failed on held-out data

The 50 pairs above were written by the same person who then tuned `ACCEPT`
against them. On `eval_heldout.py` -- 25 paraphrase pairs written afterwards
by someone else -- the local verifier recovers **1/25 (4%)**. It returns 0.000
on pairs like `'Does it support SSO?'` / `'Can I log in with single
sign-on?'`. The single-threshold design scores 0/25 on the same set.

Three local cross-encoders were tried; the best clean cutoff recovers 4-8%.
Not a tuning problem: these models match their training distribution rather
than reason about meaning.

**The split survived; the model did not.** On the same held-out set:

```
retrieval    recall@5 = 100%   (25/25 targets reach the shortlist)
verification 4%                (1/25 accepted)
```

Every right answer was handed to the verifier. The verifier threw it away. So
the two-stage argument is intact and stage 2 needs a model that actually reads
-- an LLM, not a fine-tuned classifier. That is wired up
(`VERIFY_BACKEND=anthropic`, Haiku 4.5) and **untested**, because this machine
has no API key.

Do not quote the 40% figure. It is an in-sample number.

### The consequence that inverts your instinct

Once verification exists, retrieval should **over-suggest**. A tight retriever
is now the liability: anything it drops never reaches the verifier and is
unrecoverable, while anything it wrongly includes gets thrown out. So
`THRESHOLD` stops being the decision and becomes `FLOOR = 0.15`, which exists
only to avoid verifying garbage on a cold cache.

This is the part people get wrong retrofitting a verifier: they see bad hits
and tighten the threshold, which is exactly backwards.

Negation is just the most visible case. Version numbers, dosages, dates,
units, and comparative direction all fail identically, for the same reason.

## Still worth trying

**Better embeddings.** `all-MiniLM-L6-v2` is small and old. Re-run with
`EMBED_BACKEND=openai` -- `embedder.py` is swappable for exactly this. It
would raise recall@k, which is the ceiling on everything downstream.

## Deploying

Container host, not serverless. Two reasons serverless does not work here:
the torch dependency is ~4.8GB against a 500MB function limit, and more
fundamentally a cache needs state that outlives a request -- an ephemeral
filesystem means an empty cache on every cold start, which is strictly worse
than no cache at all.

```bash
docker build -t semantic-cache .
docker run -p 8000:8000 -v semcache:/data \
  -e ANTHROPIC_API_KEY=sk-ant-... semantic-cache
```

On Railway / Render / Fly: point it at this repo, add a persistent volume
mounted at `/data`, and set the key in the platform's environment-variable UI
(`.env` is gitignored and never ships).

Verified on a real build (image 2.47GB, 415MB resident):

```
llm       cached=False  sim=0.000  3694.8ms  'how do I reset my password'
exact     cached=True   sim=1.000     2.3ms  'How do I reset my password?'
exact     cached=True   sim=1.000     1.9ms  'HOW DO I   reset my password'
semantic  cached=True   sim=0.605   270.6ms  'how do I change my login credentials'
llm       cached=False  sim=0.518   867.0ms  'how do I reset my API key'
```

The last two lines are the thesis: 0.605 served, 0.518 refused. Absolute
similarity is not the decision. Cache survives `docker restart` with the
volume mounted.

**Two things that will bite you:**

*Memory depends on which verifier you run.* `VERIFY_BACKEND=crossencoder`
loads both models, ~532MB resident, which will not fit a 512MB free tier.
`VERIFY_BACKEND=anthropic` never loads the cross-encoder at all (it is
imported lazily), leaving only the embedder at ~200MB. Since the local
verifier is measured at 4% recovery on held-out data, the Anthropic backend is
both the better and the smaller choice.

*Without a volume at `/data`, the cache resets on every deploy.* It will still
serve requests; the hit rate just starts at zero each time.

## Layout

| file | what it is |
|---|---|
| `eval_pairs.py` | 50 labeled pairs. The spec -- written before the service |
| `sweep.py` | Layer 1 coverage, threshold sweep, per-family breakdown |
| `normalize.py` | Layer 1: normalization + hashing |
| `embedder.py` | text -> normalized vector, swappable backend |
| `store.py` | SQLite; indexed hash lookup + in-memory matrix search |
| `cache.py` | the three layers, poisoning guard, TTL |
| `llm.py` | the expensive call; `fake` backend by default |
| `metrics.py` | in-memory counters |
| `main.py` | FastAPI: `POST /v1/ask`, `GET /v1/stats` |
| `replay.py` | traffic replay -> hit rate, latency, cost |

Search is a brute-force matrix multiply: 100k entries x 384 dims is ~5ms in
NumPy. An approximate index (sqlite-vec, HNSW) only pays off in the millions,
and swapping one in touches only `store.search`.

## Two design choices worth defending

**TTL is a hard filter, not a score.** A common pattern multiplies similarity
by freshness (`confidence = similarity x (1 - age/ttl)`) and gates the product.
Run the arithmetic: with a 0.75 gate, a valid 0.90-similarity entry falls below
it once age exceeds 17% of TTL -- 10 minutes on a 1-hour TTL -- which silently
guts the hit rate. It also conflates two unrelated things. A semantically wrong
match does not become right by being fresh.

**Layer 1 is one indexed lookup.** `UNIQUE INDEX (tenant, query_hash)`, one
SELECT. The layering is only justified if the cheap layer is actually cheap; a
"fast path" that scans every entry is not a fast path.

## Left as TODO (marked in the code)

- `store.evict_expired` / `evict_lru` -- TTL filters on read but nothing
  deletes. Entries accumulate forever.
- `cache.is_cacheable` -- "what's my account balance" must never be shared
  across users. The store is already partitioned by tenant.
- `cache._verify` -- the two-stage check above.
- `replay.build_traffic` -- currently replays the eval set, which flatters the
  hit rate. Needs a held-out paraphrase set.
