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
  {"colbert", "Stephen Colbert hosted The Colbert Report before The Late Show"},
  {"conan", "Conan O'Brien is known for his self-deprecating humor and tall hair"},
  {"seth", "Seth Meyers was head writer at SNL before hosting Late Night"}
])

# Search
results = Stephen.search(encoder, index, "late night comedy")
# => [%{doc_id: "colbert", score: 15.2}, ...]

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
  {"colbert", "Stephen Colbert interviews politicians with satirical wit"},
  {"conan", "Conan O'Brien traveled the world for his travel show"}
]
results = Stephen.rerank_texts(encoder, "political satire comedy", candidates)
```

## Query Expansion (PRF)

Improve recall with pseudo-relevance feedback:

```elixir
results = Stephen.search_with_prf(encoder, index, "late night hosts")

# Tune expansion parameters
results = Stephen.search_with_prf(encoder, index, query,
  feedback_docs: 5,
  expansion_tokens: 15,
  expansion_weight: 0.3
)
```

PRF uses top-ranked documents to expand the query with related terms, finding documents that may not match the exact query.

## Debugging Scores

Understand why a document scored the way it did:

```elixir
explanation = Stephen.explain(encoder, "satirical comedy", "Colbert is a satirical host")

# Print formatted explanation
explanation |> Stephen.Scorer.format_explanation() |> IO.puts()
# Score: 15.20
#
# Query Token          -> Doc Token            Similarity
# --------------------------------------------------------
# satirical            -> satirical            0.95
# comedy               -> host                 0.72
# ...
```

This shows which query tokens matched which document tokens and their similarity scores.

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
