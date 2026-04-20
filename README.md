# distributed-embedding-cache

> Exploring whether predictive precomputation can beat LRU for RAG embedding caches.

## The question I'm exploring

Most production RAG systems treat embedding lookup as a cache problem — but they apply
caching strategies designed for exact-match workloads (LRU, LFU). Embeddings have a
weirder access pattern: a query for `"how do I deploy a model on GKE"` semantically
overlaps with `"deploy LLM kubernetes guide"`, even though they're cache-misses
under exact-match keying.

The vLLM PagedAttention paper showed how much memory waste exists when KV-cache
allocation assumes worst-case sequence lengths. I started wondering: do embedding
caches in retrieval pipelines have an analogous waste problem? And if so, can a small
predictor model proactively warm the cache for likely-next queries?

## Why I care

I keep seeing the same friction in deployed RAG: the embedding model is fast, but
the **cache layer** in front of it is dumb. Cache hit rate is the lever for both
**latency** and **carbon cost** of inference — every cache hit is one fewer GPU
forward pass. I wanted to see how far you can push hit rate with a learned predictor
before the predictor itself eats your savings.

## What's in here

A working scaffold with three pieces:

- `src/core/cache_engine.py` — async cache with semantic + exact-match lookup
- `src/core/similarity_router.py` — routes near-miss queries to nearest cached
  embedding above a similarity threshold
- `src/core/embedding_predictor.py` — lightweight model that predicts likely next
  queries from session history and pre-warms the cache

Plus the boring stuff: Redis cluster backend, gRPC server, Prometheus exporter,
Terraform for GCE + a small k8s manifest. I built this as much to learn the
**ops surface** of a production cache as the algorithm itself.

## What I'm finding (so far)

I haven't run a real benchmark on production-like traffic yet — I have synthetic
sessions only. From those:

- Similarity routing is fragile. A threshold of `0.92` cosine looks safe in
  isolation but breaks down when queries are paraphrases of *different* underlying
  intents (e.g. `"reset password"` vs `"reset settings"` are 0.89-0.93 in
  `text-embedding-3-small`).
- The predictor adds latency budget I didn't account for. If precompute happens
  on the request path, you save nothing. It has to be background-only, which
  means staleness is now a thing I have to reason about.
- LRU is a stronger baseline than I expected for sessions with low semantic drift.

## What I'd do next

- Replace synthetic traffic with a real dataset (MS MARCO sessions or BEIR query logs)
- Measure carbon cost properly — wall-clock latency isn't the right denominator,
  GPU-seconds-saved per cache lookup is
- Try a lighter predictor (n-gram or sparse retrieval over history) before
  reaching for a neural one
- Compare against vLLM's PagedAttention-style block reuse adapted for embeddings

## Status

Early exploration. The architecture works end-to-end, the experimental
results don't exist yet. Treat the numbers in any docstring as aspirational
until I have real benchmarks.

## References

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM, 2023)
- Belady, *A study of replacement algorithms for a virtual-storage computer* (IBM, 1966)
- Reimers & Gurevych, *Sentence-BERT* (2019) — for the similarity threshold intuition
