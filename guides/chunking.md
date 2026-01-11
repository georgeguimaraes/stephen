# Chunking

BERT-based models have a maximum token limit (typically 512 tokens). Stephen's Chunker handles documents that exceed this limit by splitting them into overlapping chunks.

Stephen uses [text_chunker](https://hex.pm/packages/text_chunker) for sentence-aware recursive chunking, which splits at semantic boundaries (sentences, paragraphs) similar to LangChain's approach. Research shows ColBERT performs best with sentence-aware chunking.

## Basic Usage

```elixir
documents = [
  {"doc1", "Very long document text..."},
  {"doc2", "Another long document..."}
]

# Chunk documents
{chunks, mapping} = Stephen.Chunker.chunk_documents(documents)

# With custom size (characters, not tokens)
{chunks, mapping} = Stephen.Chunker.chunk_documents(documents,
  chunk_size: 500,     # ~100-125 tokens
  chunk_overlap: 100
)

# For markdown documents
{chunks, mapping} = Stephen.Chunker.chunk_documents(documents,
  format: :markdown
)
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
# => [%{doc_id: "doc1__chunk_2", score: 15.2}, ...]

doc_results = Stephen.Chunker.merge_results(chunk_results, mapping,
  aggregation: :max  # or :sum, :mean
)
# => [%{doc_id: "doc1", score: 15.2}, ...]
```

## Aggregation Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `:max` | Highest chunk score | Short queries, specific matches |
| `:sum` | Sum of all chunk scores | Longer queries, topic matching |
| `:mean` | Average chunk score | Balanced ranking |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `:chunk_size` | 500 | Target chunk size in characters |
| `:chunk_overlap` | 100 | Overlap between chunks in characters |
| `:format` | `:plaintext` | Text format (`:plaintext` or `:markdown`) |

## Choosing Parameters

- **chunk_size**: ~500 characters ≈ 100-125 tokens, ColBERT's sweet spot
- **chunk_overlap**: 100-200 characters provides good context preservation
- **format**: Use `:markdown` for markdown docs to split at headings

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
{chunks, mapping} = Stephen.Chunker.chunk_documents(documents)

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
