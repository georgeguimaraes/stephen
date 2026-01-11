# Chunking

BERT-based models have a maximum token limit (typically 512 tokens). Stephen's Chunker handles documents that exceed this limit by splitting them into overlapping chunks.

## Basic Usage

```elixir
documents = [
  {"doc1", "Very long document text..."},
  {"doc2", "Another long document..."}
]

# Split into chunks
{chunks, mapping} = Stephen.Chunker.chunk_documents(documents,
  max_length: 180,  # tokens per chunk
  stride: 90        # overlap between chunks
)

# chunks: [{"doc1_chunk_0", "..."}, {"doc1_chunk_1", "..."}, ...]
# mapping: %{"doc1_chunk_0" => "doc1", "doc1_chunk_1" => "doc1", ...}
```

## Indexing Chunks

Index chunks like regular documents:

```elixir
index = Stephen.index(encoder, index, chunks)
```

## Merging Results

After search, merge chunk results back to document-level scores:

```elixir
chunk_results = Stephen.search(encoder, index, "query")
# => [%{doc_id: "doc1_chunk_2", score: 15.2}, ...]

doc_results = Stephen.Chunker.merge_results(chunk_results, mapping,
  aggregation: :max  # or :sum, :avg
)
# => [%{doc_id: "doc1", score: 15.2}, ...]
```

## Aggregation Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `:max` | Highest chunk score | Short queries, specific matches |
| `:sum` | Sum of all chunk scores | Longer queries, topic matching |
| `:avg` | Average chunk score | Balanced ranking |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `:max_length` | 180 | Maximum tokens per chunk |
| `:stride` | 90 | Overlap between consecutive chunks |

### Choosing Parameters

- **max_length**: Should be less than encoder's max (default 180 for ColBERT's 512 limit)
- **stride**: Smaller values = more overlap = better boundary handling but more chunks

A stride of `max_length / 2` (50% overlap) is a common choice.

## Example: Full Pipeline

```elixir
# 1. Load encoder
{:ok, encoder} = Stephen.load_encoder()

# 2. Prepare documents
documents = [
  {"paper1", File.read!("paper1.txt")},
  {"paper2", File.read!("paper2.txt")}
]

# 3. Chunk long documents
{chunks, mapping} = Stephen.Chunker.chunk_documents(documents,
  max_length: 180,
  stride: 90
)

# 4. Create and populate index
index = Stephen.new_index(encoder)
index = Stephen.index(encoder, index, chunks)

# 5. Search
chunk_results = Stephen.search(encoder, index, "comedy monologue")

# 6. Merge to document-level
doc_results = Stephen.Chunker.merge_results(chunk_results, mapping)
```

## Handling Very Long Documents

For documents with hundreds of chunks, consider:

1. **Pre-filtering**: Use metadata or BM25 to limit candidates before ColBERT
2. **Hierarchical chunking**: Larger chunks for first pass, smaller for reranking
3. **Passage-level storage**: Store only relevant passages, not full documents
