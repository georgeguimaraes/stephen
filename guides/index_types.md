# Index Types

Stephen provides three index implementations for different scale and memory requirements.

## Standard Index

`Stephen.Index` uses HNSWLib for approximate nearest neighbor search. Best for small to medium collections with frequent updates.

```elixir
index = Stephen.Index.new(
  embedding_dim: 128,
  space: :cosine,        # or :l2
  max_tokens: 100_000,   # maximum token embeddings
  m: 16,                 # HNSW M parameter
  ef_construction: 200   # HNSW build quality
)

# Add documents
index = Stephen.Index.add(index, "doc1", embeddings)

# Search
candidates = Stephen.Index.search_tokens(index, query_embeddings, 50)
```

### When to Use

- Collections under ~10K documents
- Need fast add/delete/update operations
- Memory is not a primary concern

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `:embedding_dim` | required | Embedding dimension |
| `:space` | `:cosine` | Distance metric (`:cosine` or `:l2`) |
| `:max_tokens` | 100,000 | Maximum token embeddings to store |
| `:m` | 16 | HNSW graph connectivity |
| `:ef_construction` | 200 | Index build quality |

## PLAID Index

`Stephen.Plaid` uses centroid-based inverted lists for sub-linear search time. Best for larger collections.

```elixir
plaid = Stephen.Plaid.new(
  embedding_dim: 128,
  num_centroids: 1024
)

# Index documents (trains centroids on first call)
plaid = Stephen.Plaid.index_documents(plaid, [
  {"doc1", embeddings1},
  {"doc2", embeddings2}
])

# Search
results = Stephen.Plaid.search(plaid, query_embeddings,
  top_k: 10,
  nprobe: 32
)
```

### How It Works

1. Cluster all document token embeddings into K centroids
2. Build inverted lists: centroid → [doc_ids with tokens near that centroid]
3. At query time, find nearest centroids for query tokens
4. Retrieve candidate docs from inverted lists
5. Rerank candidates with full MaxSim

### When to Use

- Collections over ~10K documents
- Search speed is critical
- Can tolerate slightly lower recall

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `:embedding_dim` | required | Embedding dimension |
| `:num_centroids` | 1024 | Number of clusters |
| `:nprobe` | 32 | Centroids to probe per search |

Higher `num_centroids` improves precision but slows search. Higher `nprobe` improves recall but slows search.

## Compressed Index

`Stephen.Index.Compressed` combines PLAID candidate generation with residual compression for memory efficiency.

```elixir
index = Stephen.Index.Compressed.new(
  embedding_dim: 128,
  num_centroids: 1024,
  compression_centroids: 2048,
  residual_bits: 8
)

# Train compression codebook (requires sample embeddings)
index = Stephen.Index.Compressed.train(index, training_embeddings)

# Add documents (stores compressed)
index = Stephen.Index.Compressed.add(index, "doc1", embeddings)

# Search (decompresses on-the-fly)
results = Stephen.Index.Compressed.search(index, query_embeddings, top_k: 10)
```

### Compression Levels

| Bits | Compression | Quality Impact |
|------|-------------|----------------|
| 8 | ~4x | Minimal |
| 2 | ~16x | Moderate |
| 1 | ~32x | Noticeable |

### When to Use

- Large collections with memory constraints
- Willing to trade some quality for memory
- Can train codebook on representative sample

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `:embedding_dim` | required | Embedding dimension |
| `:num_centroids` | 1024 | PLAID centroids |
| `:compression_centroids` | 2048 | Compression codebook size |
| `:residual_bits` | 8 | Quantization depth |

## Dynamic Updates

All index types support add, delete, and update:

```elixir
# Add
index = Index.add(index, "doc1", embeddings)

# Delete
index = Index.delete(index, "doc1")

# Update (delete + add)
index = Index.update(index, "doc1", new_embeddings)

# Batch operations
index = Index.add_all(index, [{"doc2", emb2}, {"doc3", emb3}])
index = Index.delete_all(index, ["doc2", "doc3"])
```

## Persistence

All indexes can be saved and loaded:

```elixir
:ok = Stephen.Index.save(index, "/path/to/index")
{:ok, index} = Stephen.Index.load("/path/to/index")
```

The save format includes all metadata needed to reconstruct the index.
