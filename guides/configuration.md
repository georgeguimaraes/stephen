# Configuration

## Encoder Options

```elixir
{:ok, encoder} = Stephen.Encoder.load(
  model: "colbert-ir/colbertv2.0",
  max_query_length: 32,
  max_doc_length: 180,
  projection_dim: 128
)
```

### Model Selection

| Model | Description |
|-------|-------------|
| `colbert-ir/colbertv2.0` | Official ColBERT v2 with trained projection (recommended) |
| `colbert-ir/colbertv1.0` | Original ColBERT model |
| `bert-base-uncased` | Standard BERT (requires random projection) |
| Any HuggingFace BERT | Custom models work with random projection |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `:model` | `colbert-ir/colbertv2.0` | HuggingFace model name |
| `:max_query_length` | 32 | Query padding length |
| `:max_doc_length` | 180 | Max document tokens |
| `:projection_dim` | 128 | Output dimension (nil to disable) |
| `:base_module` | auto-detected | Override Bumblebee module |

### ColBERT vs Standard Models

When loading official ColBERT models, Stephen:

1. Downloads the model from HuggingFace
2. Loads BERT backbone via Bumblebee
3. Extracts trained projection weights from SafeTensors

For other models, Stephen initializes a random projection matrix.

## Document Encoding Options

```elixir
embeddings = Stephen.Encoder.encode_document(encoder, text,
  skip_punctuation?: true,
  deduplicate?: true
)
```

| Option | Default | Description |
|--------|---------|-------------|
| `:skip_punctuation?` | false | Filter punctuation token embeddings |
| `:deduplicate?` | false | Remove near-duplicate embeddings |

## Index Options

### Standard Index

```elixir
index = Stephen.Index.new(
  embedding_dim: 128,
  space: :cosine,
  max_tokens: 100_000,
  m: 16,
  ef_construction: 200
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `:embedding_dim` | required | Must match encoder output |
| `:space` | `:cosine` | Distance metric (`:cosine` or `:l2`) |
| `:max_tokens` | 100,000 | Maximum token embeddings |
| `:m` | 16 | HNSW connectivity |
| `:ef_construction` | 200 | Build quality |

### PLAID Index

```elixir
plaid = Stephen.Plaid.new(
  embedding_dim: 128,
  num_centroids: 1024
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `:embedding_dim` | required | Must match encoder output |
| `:num_centroids` | 1024 | Number of clusters |

### Compressed Index

```elixir
index = Stephen.Index.Compressed.new(
  embedding_dim: 128,
  num_centroids: 1024,
  compression_centroids: 2048,
  residual_bits: 8
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `:embedding_dim` | required | Must match encoder output |
| `:num_centroids` | 1024 | PLAID centroids |
| `:compression_centroids` | 2048 | Compression codebook size |
| `:residual_bits` | 8 | Quantization depth (1, 2, 4, or 8) |

## Search Options

```elixir
results = Stephen.search(encoder, index, query,
  top_k: 10,
  rerank?: true,
  candidates_per_token: 50
)
```

| Option | Default | Description |
|--------|---------|-------------|
| `:top_k` | 10 | Number of results |
| `:rerank?` | true | Full MaxSim reranking |
| `:candidates_per_token` | 50 | ANN candidates (Index) |
| `:nprobe` | 32 | Centroids to probe (PLAID) |

## GPU Acceleration

Enable EXLA for GPU acceleration:

```elixir
# mix.exs
{:exla, "~> 0.9"}

# config/config.exs
config :nx, default_backend: EXLA.Backend
```

For CPU-only with EXLA optimizations:

```elixir
config :nx, default_backend: {EXLA.Backend, client: :host}
```

## Batch Processing

For large document sets, process in batches to manage memory:

```elixir
documents
|> Stream.chunk_every(100)
|> Enum.reduce(index, fn batch, acc ->
  Stephen.index(encoder, acc, batch)
end)
```

For batch queries:

```elixir
results = Stephen.Retriever.batch_search(encoder, index, queries, top_k: 10)
```
