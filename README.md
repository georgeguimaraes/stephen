# Stephen

ColBERT-style neural retrieval for Elixir.

Stephen implements late interaction retrieval using per-token embeddings and MaxSim scoring. Instead of compressing text into a single vector, it keeps one embedding per token, enabling fine-grained semantic matching.

## Installation

```elixir
def deps do
  [
    {:stephen, "~> 0.1.0"},
    {:exla, "~> 0.9"}  # optional, for GPU acceleration
  ]
end
```

For GPU acceleration:

```elixir
# config/config.exs
config :nx, default_backend: EXLA.Backend
```

## Quick Start

```elixir
# Load encoder (downloads model on first use)
{:ok, encoder} = Stephen.load_encoder()

# Create index and add documents
index = Stephen.new_index(encoder)
index = Stephen.index(encoder, index, [
  {"doc1", "Elixir is a functional programming language"},
  {"doc2", "Python is popular for machine learning"},
  {"doc3", "Rust provides memory safety"}
])

# Search
results = Stephen.search(encoder, index, "functional programming")
# => [%{doc_id: "doc1", score: 15.2}, ...]

# Save/load
:ok = Stephen.save_index(index, "my_index")
{:ok, index} = Stephen.load_index("my_index")
```

## Why ColBERT?

Traditional dense retrieval compresses each text into a single vector. ColBERT keeps per-token embeddings and matches query tokens to document tokens individually:

1. Each token gets its own embedding
2. For each query token, find the best-matching document token (MaxSim)
3. Sum these maximum similarities

This captures nuanced relevance that single-vector methods miss.

## Reranking

Use Stephen to rerank candidates from a faster first-stage retriever:

```elixir
# From indexed documents
results = Stephen.rerank(encoder, index, "query", ["doc1", "doc5", "doc12"])

# From raw text (no index needed)
candidates = [
  {"doc1", "Elixir is functional"},
  {"doc2", "Python is dynamic"}
]
results = Stephen.rerank_texts(encoder, "functional programming", candidates)
```

## Query Expansion (PRF)

Improve recall with pseudo-relevance feedback:

```elixir
results = Stephen.search_with_prf(encoder, index, "machine learning")

# Tune expansion parameters
results = Stephen.search_with_prf(encoder, index, query,
  feedback_docs: 5,
  expansion_tokens: 15,
  expansion_weight: 0.3
)
```

PRF uses top-ranked documents to expand the query with related terms, finding documents that may not match the exact query.

## Index Types

| Index | Use Case |
|-------|----------|
| `Stephen.Index` | Small-medium collections, fast updates |
| `Stephen.Plaid` | Larger collections, sub-linear search |
| `Stephen.Index.Compressed` | Memory-constrained, 4-32x compression |

## Documentation

See the [guides](guides/) for detailed documentation:

- [Architecture](guides/architecture.md): how ColBERT and Stephen work
- [Index Types](guides/index_types.md): choosing and configuring indexes
- [Compression](guides/compression.md): residual quantization for memory efficiency
- [Chunking](guides/chunking.md): handling long documents
- [Configuration](guides/configuration.md): encoder and index options

## License

Copyright (c) 2025 George Guimarães

Licensed under the MIT License. See [LICENSE](LICENSE) for details.
