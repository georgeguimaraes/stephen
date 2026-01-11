# Compression

Stephen implements ColBERTv2-style residual compression for memory-efficient storage of token embeddings.

## How It Works

Instead of storing full float32 embeddings (512 bytes for 128 dimensions), we store:

1. **Centroid ID** (2 bytes): which cluster centroid is closest
2. **Quantized Residual** (variable): the difference from the centroid

To reconstruct: `embedding ≈ centroid[id] + dequantize(residual)`

## Compression Levels

| Bits | Storage per Token | Compression Ratio | Quality |
|------|-------------------|-------------------|---------|
| 8 | 130 bytes | ~4x | Excellent |
| 2 | 34 bytes | ~16x | Good |
| 1 | 18 bytes | ~32x | Acceptable |

## Training the Codebook

Compression requires training a codebook on representative embeddings:

```elixir
# Collect sample embeddings
sample_embeddings = documents
|> Enum.take(1000)
|> Enum.map(fn {_id, text} -> Encoder.encode_document(encoder, text) end)
|> Nx.concatenate(axis: 0)

# Train compression codebook
compression = Stephen.Compression.train(sample_embeddings,
  num_centroids: 2048,
  residual_bits: 8,
  iterations: 20
)
```

More centroids generally improves quality but increases memory for the codebook itself.

## Using Compressed Index

```elixir
# Create index with compression settings
index = Stephen.Index.Compressed.new(
  embedding_dim: 128,
  num_centroids: 1024,        # PLAID centroids
  compression_centroids: 2048, # compression codebook size
  residual_bits: 8
)

# Train on sample data
index = Stephen.Index.Compressed.train(index, sample_embeddings)

# Add documents (stored compressed)
index = Stephen.Index.Compressed.add(index, "doc1", embeddings)

# Search (decompresses candidates on-the-fly)
results = Stephen.Index.Compressed.search(index, query_embeddings, top_k: 10)
```

## Standalone Compression

You can also use compression independently:

```elixir
# Train codebook
compression = Stephen.Compression.train(embeddings,
  num_centroids: 2048,
  residual_bits: 8
)

# Compress embeddings
compressed = Stephen.Compression.compress(compression, embeddings)
# => %{centroid_ids: tensor, residuals: tensor}

# Decompress
reconstructed = Stephen.Compression.decompress(compression, compressed)

# Compute approximate similarity directly
score = Stephen.Compression.approximate_similarity(
  compression,
  query_embeddings,
  compressed_doc
)
```

## Saving and Loading

```elixir
# Save codebook
:ok = Stephen.Compression.save(compression, "codebook.bin")

# Load codebook
{:ok, compression} = Stephen.Compression.load("codebook.bin")
```

## Memory Calculation

For a collection with N documents averaging T tokens each:

| Setting | Memory |
|---------|--------|
| Uncompressed (f32) | N × T × dim × 4 bytes |
| 8-bit | N × T × (2 + dim) bytes |
| 2-bit | N × T × (2 + dim/4) bytes |
| 1-bit | N × T × (2 + dim/8) bytes |

Example: 100K documents, 100 tokens/doc, 128-dim embeddings:

| Setting | Memory |
|---------|--------|
| Uncompressed | 5.1 GB |
| 8-bit | 1.3 GB |
| 2-bit | 340 MB |
| 1-bit | 180 MB |

## Quality vs Compression Tradeoff

Lower bit depths reduce memory but increase approximation error. Recommendations:

- **8-bit**: Default choice, minimal quality loss
- **2-bit**: When memory is constrained, acceptable for most use cases
- **1-bit**: Extreme compression, test quality on your data first
