# Stephen

ColBERT-style neural retrieval library for Elixir.

Stephen brings state-of-the-art semantic search to the Elixir ecosystem using per-token embeddings and late interaction scoring (MaxSim). Named after the ColBERT architecture (Contextual Late Interaction over BERT), it provides fine-grained matching between queries and documents for superior retrieval quality.

## Features

- **Per-token embeddings**: Better granularity than pooled embeddings
- **Late interaction scoring**: MaxSim matches query tokens individually to document tokens
- **Residual compression**: 4-32x memory reduction with minimal quality loss
- **PLAID indexing**: Sub-linear search time for large collections
- **Dynamic updates**: Add, delete, and update documents in the index
- **GPU acceleration**: Optional EXLA backend for faster encoding

## Installation

Add `stephen` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:stephen, "~> 0.1.0"},
    {:exla, "~> 0.9"}  # optional, for GPU acceleration
  ]
end
```

To enable GPU acceleration, configure your default backend:

```elixir
# config/config.exs
config :nx, default_backend: EXLA.Backend
```

## Quick Start

```elixir
# Load encoder (downloads model on first use)
{:ok, encoder} = Stephen.load_encoder()

# Create index
index = Stephen.new_index(encoder)

# Index some documents
documents = [
  {"doc1", "Elixir is a functional programming language"},
  {"doc2", "Python is popular for machine learning"},
  {"doc3", "Rust provides memory safety without garbage collection"}
]
index = Stephen.index(encoder, index, documents)

# Search
results = Stephen.search(encoder, index, "functional programming", top_k: 3)
# => [%{doc_id: "doc1", score: 15.2}, %{doc_id: "doc3", score: 11.8}, ...]

# Save and load index
:ok = Stephen.save_index(index, "my_index")
{:ok, index} = Stephen.load_index("my_index")
```

## Architecture

### How ColBERT Works

Traditional dense retrieval encodes queries and documents into single vectors, then computes similarity. ColBERT instead:

1. Encodes each token separately (per-token embeddings)
2. Computes similarity between every query-document token pair
3. For each query token, takes the maximum similarity (MaxSim)
4. Sums all maximum similarities as the final score

This captures nuanced relevance that single-vector methods miss.

### Modules

| Module | Purpose |
|--------|---------|
| `Stephen.Encoder` | Text to per-token embeddings using BERT models |
| `Stephen.Scorer` | MaxSim scoring between query and document embeddings |
| `Stephen.Index` | HNSWLib-backed ANN search with dynamic updates |
| `Stephen.Plaid` | Centroid-based inverted index for faster retrieval |
| `Stephen.Index.Compressed` | PLAID + residual compression for memory efficiency |
| `Stephen.Compression` | 8-bit, 2-bit, and 1-bit residual quantization |
| `Stephen.Chunker` | Long document handling with overlapping chunks |
| `Stephen.Retriever` | Two-stage retrieval orchestration |

## Index Types

Stephen provides three index implementations for different use cases:

### Standard Index

Best for small to medium collections where memory is not a concern.

```elixir
index = Stephen.Index.new(embedding_dim: 128, max_elements: 100_000)
index = Stephen.Index.add(index, "doc1", embeddings)
candidates = Stephen.Index.search_tokens(index, query_embeddings, 50)
```

### PLAID Index

Best for larger collections where you need faster search.

```elixir
plaid = Stephen.Plaid.new(embedding_dim: 128, num_centroids: 1024)
plaid = Stephen.Plaid.index_documents(plaid, documents)
results = Stephen.Plaid.search(plaid, query_embeddings, top_k: 10, nprobe: 32)
```

### Compressed Index

Best for large collections where memory is a concern. Combines PLAID candidate generation with residual compression.

```elixir
index = Stephen.Index.Compressed.new(embedding_dim: 128)
index = Stephen.Index.Compressed.train(index, all_embeddings, residual_bits: 8)
index = Stephen.Index.Compressed.add(index, "doc1", embeddings)
results = Stephen.Index.Compressed.search(index, query_embeddings, top_k: 10)
```

Compression ratios:

- 8-bit: ~4x compression
- 2-bit: ~16x compression
- 1-bit: ~32x compression

## Dynamic Updates

All index types support dynamic document management:

```elixir
# Add documents
index = Index.add(index, "doc1", embeddings)
index = Index.add_all(index, [{"doc2", emb2}, {"doc3", emb3}])

# Remove documents
index = Index.delete(index, "doc1")
index = Index.delete_all(index, ["doc2", "doc3"])

# Update documents (delete + add)
index = Index.update(index, "doc1", new_embeddings)

# Check existence
Index.has_doc?(index, "doc1")
```

## Long Documents

Use the Chunker for documents exceeding the model's max token length:

```elixir
# Chunk documents with overlap
{chunks, mapping} = Stephen.Chunker.chunk_documents(documents,
  max_length: 180,
  stride: 90
)

# Index chunks
index = Stephen.index(encoder, index, chunks)

# Search and merge results back to documents
chunk_results = Stephen.search(encoder, index, "query")
doc_results = Stephen.Chunker.merge_results(chunk_results, mapping, aggregation: :max)
```

## Reranking

Use Stephen to rerank candidates from a faster first-stage retriever (like BM25):

```elixir
# Get candidates from BM25 or other retriever
candidates = ["doc1", "doc5", "doc12", "doc7"]

# Rerank with Stephen
results = Stephen.rerank(encoder, index, "query text", candidates)
```

## Batch Queries

Search multiple queries efficiently with batch encoding:

```elixir
# Batch search: encode all queries together, then search
queries = ["functional programming", "machine learning", "web development"]
results = Stephen.Retriever.batch_search(encoder, index, queries, top_k: 5)
# results[0] contains top 5 results for "functional programming"
# results[1] contains top 5 results for "machine learning"
# etc.

# Batch rerank: rerank different candidate sets for each query
queries_and_candidates = [
  {"programming languages", ["doc1", "doc2", "doc3"]},
  {"data science", ["doc2", "doc4", "doc5"]}
]
results = Stephen.Retriever.batch_rerank(encoder, index, queries_and_candidates)

# Multi-query scoring: score multiple queries against multiple documents
query_embeddings = [query1_emb, query2_emb]
doc_embeddings = [doc1_emb, doc2_emb, doc3_emb]
scores = Stephen.Scorer.multi_max_sim(query_embeddings, doc_embeddings)
# scores[i][j] = score of query i against doc j
```

## Configuration

### Encoder Options

```elixir
{:ok, encoder} = Stephen.Encoder.load(
  model: "sentence-transformers/all-MiniLM-L6-v2",  # HuggingFace model
  projection_dim: 128,        # Optional dimension reduction
  max_length: 512,            # Max tokens per document
  query_max_length: 32,       # Query padding length
  skip_punctuation: true,     # Filter punctuation tokens
  dedupe_threshold: 0.99      # Remove near-duplicate embeddings
)
```

### Using Official ColBERT Models

Stephen can load official ColBERT model weights directly, including the trained projection layer:

```elixir
# Load the official ColBERTv2 model with trained weights
{:ok, encoder} = Stephen.Encoder.load(model: "colbert-ir/colbertv2.0")

# Uses BERT-base (768 dims) with trained 768->128 projection
encoder.embedding_dim  # => 768
encoder.output_dim     # => 128
```

When you specify a ColBERT model, Stephen automatically:
1. Downloads the model weights from HuggingFace
2. Loads the BERT backbone via Bumblebee
3. Extracts the trained projection weights from the SafeTensors file

Supported ColBERT models:
- `colbert-ir/colbertv2.0` (recommended)
- `colbert-ir/colbertv1.0`

### Index Options

```elixir
index = Stephen.Index.new(
  embedding_dim: 128,
  space: :cosine,             # or :l2
  max_elements: 100_000,
  m: 16,                      # HNSW M parameter
  ef_construction: 200        # HNSW ef_construction
)
```

### PLAID Options

```elixir
plaid = Stephen.Plaid.new(
  embedding_dim: 128,
  num_centroids: 1024         # More centroids = better precision, slower
)

results = Stephen.Plaid.search(plaid, query,
  top_k: 10,
  nprobe: 32                  # More probes = better recall, slower
)
```

## Python ColBERT Compatibility

Stephen aims for feature parity with the Python ColBERT implementation:

- [x] Per-token embeddings
- [x] MaxSim scoring
- [x] Query padding with [MASK] tokens
- [x] Batch encoding
- [x] Punctuation filtering
- [x] 8-bit residual compression
- [x] PLAID indexing
- [x] Dynamic index updates
- [x] Multi-vector batch queries
- [x] Official ColBERT model loading
- [ ] Distillation training

## Dependencies

- **bumblebee**: HuggingFace model loading
- **nx**: Tensor operations
- **axon**: Neural network execution
- **hnswlib**: Approximate nearest neighbor search
- **exla** (optional): GPU acceleration

## License

MIT
