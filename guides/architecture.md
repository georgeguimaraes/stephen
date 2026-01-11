# Architecture

Stephen implements ColBERT (Contextual Late Interaction over BERT), a neural retrieval architecture that uses per-token embeddings for fine-grained semantic matching.

## How ColBERT Works

Traditional dense retrieval compresses each text into a single vector:

```
Query: "functional programming" → [0.1, 0.3, -0.2, ...]  (one vector)
Doc:   "Elixir is functional"   → [0.2, 0.1, -0.1, ...]  (one vector)
Score: dot_product(query_vec, doc_vec)
```

ColBERT keeps one embedding per token:

```
Query: "functional programming"
  → "functional" → [0.1, 0.3, ...]
  → "programming" → [0.2, -0.1, ...]

Doc: "Elixir is functional"
  → "Elixir" → [0.0, 0.2, ...]
  → "is" → [0.1, 0.0, ...]
  → "functional" → [0.1, 0.3, ...]
```

## MaxSim Scoring

For each query token, find the maximum similarity to any document token:

```
"functional" best matches "functional" → 0.95
"programming" best matches "Elixir" → 0.42

Final score = 0.95 + 0.42 = 1.37
```

This captures that "functional" has a perfect match while "programming" has a weaker match, providing more nuanced scoring than a single vector comparison.

## Module Overview

| Module | Purpose |
|--------|---------|
| `Stephen.Encoder` | BERT-based text to per-token embeddings |
| `Stephen.Scorer` | MaxSim and score normalization |
| `Stephen.Index` | HNSWLib-backed ANN search |
| `Stephen.Plaid` | Centroid-based inverted index |
| `Stephen.Index.Compressed` | PLAID + residual compression |
| `Stephen.Compression` | Quantization codebook |
| `Stephen.Chunker` | Long document handling |
| `Stephen.Retriever` | High-level retrieval orchestration |

## Retrieval Pipeline

### Standard Flow

```elixir
# 1. Encode query to per-token embeddings
query_embeddings = Encoder.encode_query(encoder, "functional programming")

# 2. Find candidate documents via ANN search
candidates = Index.search_tokens(index, query_embeddings, k: 50)

# 3. Rerank candidates with full MaxSim
results = candidates
|> Enum.map(fn doc_id ->
  doc_embeddings = Index.get_embeddings(index, doc_id)
  score = Scorer.max_sim(query_embeddings, doc_embeddings)
  %{doc_id: doc_id, score: score}
end)
|> Enum.sort_by(& &1.score, :desc)
```

### Two-Stage Retrieval

For large collections, use a fast first stage (BM25, dense retrieval) then rerank with ColBERT:

```elixir
# Stage 1: Fast candidate retrieval (external)
candidates = MySearch.bm25_search(query, top_k: 100)

# Stage 2: ColBERT reranking
results = Stephen.rerank(encoder, index, query, candidates)
```

## Query Augmentation

ColBERT pads queries with [MASK] tokens to a fixed length (default 32). The model learns to use these positions for query expansion, matching additional relevant terms.

```elixir
# Input: "functional programming"
# Internal: "[Q] functional programming [MASK] [MASK] ... [MASK]"
#           ↑                           ↑─────────────────────↑
#           Query marker                Padding to 32 tokens
```

## Document Markers

Both queries and documents are prefixed with markers:

- Queries: `[Q]` prefix
- Documents: `[D]` prefix

These help the model distinguish query vs document context during encoding.
