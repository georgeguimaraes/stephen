# Reranking

Stephen can be used as a reranker to improve results from a faster first-stage retriever like BM25, Elasticsearch, or a dense retriever.

## Why Rerank?

ColBERT's per-token matching is more accurate than single-vector similarity but slower. A common pattern is two-stage retrieval:

1. **First stage**: Fast retrieval of 100-1000 candidates (BM25, dense retrieval)
2. **Second stage**: Rerank candidates with ColBERT MaxSim

This gives you ColBERT-quality results at scale.

## Reranking Indexed Documents

If your candidates are already in a Stephen index:

```elixir
{:ok, encoder} = Stephen.load_encoder()

# Get candidates from first-stage retriever
candidates = [:doc1, :doc5, :doc12, :doc23]

# Rerank with ColBERT
results = Stephen.rerank(encoder, index, "my query", candidates)
# => [%{doc_id: :doc5, score: 18.2}, %{doc_id: :doc1, score: 12.1}, ...]
```

## Reranking Raw Text

For candidates from external sources (Elasticsearch, Postgres full-text, etc.), use `rerank_texts/4`:

```elixir
# Results from Elasticsearch or BM25
candidates = [
  {"colbert", "Stephen Colbert does satirical political commentary on The Late Show"},
  {"conan", "Conan O'Brien is known for absurdist comedy bits and celebrity interviews"},
  {"seth", "Seth Meyers hosts Late Night and does A Closer Look political segments"}
]

results = Stephen.rerank_texts(encoder, "political comedy", candidates)
# => [%{doc_id: "colbert", score: 18.5}, %{doc_id: "seth", score: 12.1}, ...]
```

Documents are encoded on-the-fly, so this is slower than reranking indexed docs but doesn't require pre-indexing.

## Options

Both reranking functions accept a `:top_k` option:

```elixir
# Return only top 5
results = Stephen.rerank(encoder, index, query, candidates, top_k: 5)
results = Stephen.rerank_texts(encoder, query, candidates, top_k: 5)
```

## Batch Reranking

For multiple queries, batch them to share overhead:

```elixir
queries_and_candidates = [
  {"political satire", ["doc1", "doc5", "doc12"]},
  {"late night comedy", ["doc2", "doc7", "doc15"]}
]

results = Stephen.Retriever.batch_rerank(encoder, index, queries_and_candidates)
# results[0] = results for first query
# results[1] = results for second query
```

## Pre-computed Embeddings

If you're reranking multiple candidate sets with the same query, encode once:

```elixir
query_embeddings = Stephen.Encoder.encode_query(encoder, "my query")

# Rerank different candidate sets with same query
results1 = Stephen.Retriever.rerank_with_embeddings(query_embeddings, index, candidates1)
results2 = Stephen.Retriever.rerank_with_embeddings(query_embeddings, index, candidates2)
```

## Integration Example

Here's a complete example combining BM25 with ColBERT reranking:

```elixir
defmodule MySearch do
  def search(query, opts \\ []) do
    top_k = Keyword.get(opts, :top_k, 10)

    # First stage: BM25 retrieves 100 candidates
    candidates = bm25_search(query, limit: 100)

    # Second stage: ColBERT reranks to top_k
    Stephen.rerank_texts(encoder(), query, candidates, top_k: top_k)
  end

  defp bm25_search(query, opts) do
    # Your BM25/Elasticsearch/Postgres full-text search
    # Returns [{id, text}, ...]
  end

  defp encoder do
    # Cache encoder in application state
    MyApp.get_encoder()
  end
end
```

## When to Use Reranking

Use reranking when:

- You have an existing search system you want to improve
- Your collection is too large for full ColBERT search
- You need sub-100ms latency (rerank 100 candidates vs search millions)
- You want to combine keyword and semantic matching

Skip reranking when:

- Your collection is small enough for direct Stephen search
- You don't have a good first-stage retriever
- Quality of first-stage candidates is already poor (garbage in, garbage out)
