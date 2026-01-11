defmodule Stephen do
  @moduledoc """
  ColBERT-style neural retrieval for Elixir.

  Stephen implements late interaction retrieval using per-token
  embeddings and MaxSim scoring, enabling high-quality semantic search.

  ## Quick Start

      # Load the encoder
      {:ok, encoder} = Stephen.load_encoder()

      # Create an index
      index = Stephen.new_index(encoder)

      # Index some documents
      documents = [
        {"doc1", "The quick brown fox jumps over the lazy dog"},
        {"doc2", "Machine learning is a subset of artificial intelligence"},
        {"doc3", "Elixir is a functional programming language"}
      ]
      index = Stephen.index(encoder, index, documents)

      # Search
      results = Stephen.search(encoder, index, "programming languages")
      # => [%{doc_id: "doc3", score: 12.5}, ...]

  ## Architecture

  Stephen uses the ColBERT architecture:

  1. **Encoder**: Converts text to per-token embeddings using BERT
  2. **Index**: Stores document embeddings with ANN search via HNSWLib
  3. **Scorer**: Computes MaxSim scores for late interaction
  4. **Retriever**: Orchestrates the full retrieval pipeline
  """

  alias Stephen.{Encoder, Index, Retriever}

  @type encoder :: Encoder.encoder()
  @type index :: Index.t()
  @type doc_id :: term()
  @type search_result :: Retriever.search_result()

  @doc """
  Loads the encoder model.

  ## Options
    * `:model` - HuggingFace model name (default: sentence-transformers/all-MiniLM-L6-v2)

  ## Examples

      {:ok, encoder} = Stephen.load_encoder()
      {:ok, encoder} = Stephen.load_encoder(model: "bert-base-uncased")
  """
  @spec load_encoder(keyword()) :: {:ok, encoder()} | {:error, term()}
  defdelegate load_encoder(opts \\ []), to: Encoder, as: :load

  @doc """
  Creates a new empty index.

  ## Options
    * `:max_tokens` - Maximum number of token embeddings (default: 100_000)
    * `:m` - HNSW M parameter (default: 16)
    * `:ef_construction` - HNSW ef_construction (default: 200)

  ## Examples

      index = Stephen.new_index(encoder)
      index = Stephen.new_index(encoder, max_tokens: 1_000_000)
  """
  @spec new_index(encoder(), keyword()) :: index()
  def new_index(encoder, opts \\ []) do
    opts = Keyword.put(opts, :embedding_dim, Encoder.embedding_dim(encoder))
    Index.new(opts)
  end

  @doc """
  Indexes documents.

  ## Arguments
    * `encoder` - Loaded encoder
    * `index` - Document index
    * `documents` - List of `{doc_id, text}` tuples

  ## Examples

      documents = [{"doc1", "Hello world"}, {"doc2", "Goodbye world"}]
      index = Stephen.index(encoder, index, documents)
  """
  @spec index(encoder(), index(), [{doc_id(), String.t()}]) :: index()
  defdelegate index(encoder, index, documents), to: Retriever, as: :index_documents

  @doc """
  Searches for documents matching a query.

  ## Options
    * `:top_k` - Number of results to return (default: 10)
    * `:rerank?` - Whether to rerank with full MaxSim (default: true)

  ## Examples

      results = Stephen.search(encoder, index, "hello")
      # => [%{doc_id: "doc1", score: 15.2}, %{doc_id: "doc2", score: 8.1}]
  """
  @spec search(encoder(), index(), String.t(), keyword()) :: [search_result()]
  defdelegate search(encoder, index, query, opts \\ []), to: Retriever

  @doc """
  Reranks a list of documents against a query.

  Useful when you have candidate documents from another source
  (e.g., keyword search) and want to rerank them semantically.

  ## Examples

      results = Stephen.rerank(encoder, index, "hello", ["doc1", "doc2"])
  """
  @spec rerank(encoder(), index(), String.t(), [doc_id()]) :: [search_result()]
  defdelegate rerank(encoder, index, query, doc_ids), to: Retriever

  @doc """
  Reranks raw text documents against a query without requiring an index.

  This is useful for reranking results from external sources like BM25,
  Elasticsearch, or any other retrieval system. Documents are encoded
  on-the-fly and scored using ColBERT's MaxSim.

  ## Arguments
    * `encoder` - Loaded encoder
    * `query` - Query string
    * `documents` - List of `{id, text}` tuples to rerank

  ## Options
    * `:top_k` - Number of results to return (default: all)

  ## Returns
    List of `%{doc_id: term(), score: float()}` sorted by score descending.

  ## Examples

      # Rerank search results from Elasticsearch
      candidates = [
        {"doc1", "Elixir is a dynamic, functional language"},
        {"doc2", "Python is a popular programming language"},
        {"doc3", "Erlang powers distributed systems"}
      ]
      results = Stephen.rerank_texts(encoder, "functional programming", candidates)
      # => [%{doc_id: "doc1", score: 18.5}, %{doc_id: "doc2", score: 12.1}, ...]

      # Return only top 2
      results = Stephen.rerank_texts(encoder, query, candidates, top_k: 2)
  """
  @spec rerank_texts(encoder(), String.t(), [{doc_id(), String.t()}], keyword()) :: [
          search_result()
        ]
  def rerank_texts(encoder, query, documents, opts \\ []) do
    Retriever.rerank_texts(encoder, query, documents, opts)
  end

  @doc """
  Saves an index to disk.

  ## Examples

      :ok = Stephen.save_index(index, "/path/to/index")
  """
  @spec save_index(index(), Path.t()) :: :ok | {:error, term()}
  defdelegate save_index(index, path), to: Index, as: :save

  @doc """
  Loads an index from disk.

  ## Examples

      {:ok, index} = Stephen.load_index("/path/to/index")
  """
  @spec load_index(Path.t()) :: {:ok, index()} | {:error, term()}
  defdelegate load_index(path), to: Index, as: :load

  @doc """
  Searches with pseudo-relevance feedback (PRF) for query expansion.

  PRF improves recall by expanding the query with information from
  top-ranked documents. Useful when you want to find more relevant
  documents that may not match the exact query terms.

  ## Options
    * `:top_k` - Final results to return (default: 10)
    * `:feedback_docs` - Number of docs for feedback (default: 3)
    * `:expansion_tokens` - Tokens to add from feedback (default: 10)
    * `:expansion_weight` - Weight for expansion vs original (default: 0.5)

  ## Examples

      results = Stephen.search_with_prf(encoder, index, "machine learning")

      # Tune PRF parameters
      results = Stephen.search_with_prf(encoder, index, query,
        feedback_docs: 5,
        expansion_weight: 0.3
      )
  """
  @spec search_with_prf(encoder(), index(), String.t(), keyword()) :: [search_result()]
  defdelegate search_with_prf(encoder, index, query, opts \\ []), to: Retriever

  @doc """
  Explains why a document scored the way it did for a query.

  Returns detailed information about which query tokens matched which
  document tokens, useful for debugging and understanding retrieval results.

  ## Arguments
    * `encoder` - Loaded encoder
    * `query` - Query string
    * `doc_text` - Document text

  ## Returns
    Map containing:
    * `:score` - Total MaxSim score
    * `:matches` - List of match details for each query token

  ## Examples

      explanation = Stephen.explain(encoder, "functional programming", "Elixir is functional")
      # => %{
      #   score: 15.2,
      #   matches: [
      #     %{query_token: "functional", doc_token: "functional", similarity: 0.95, ...},
      #     ...
      #   ]
      # }

      # Print formatted explanation
      explanation |> Stephen.Scorer.format_explanation() |> IO.puts()
  """
  @spec explain(encoder(), String.t(), String.t()) :: map()
  def explain(encoder, query, doc_text) do
    query_emb = Encoder.encode_query(encoder, query)
    doc_emb = Encoder.encode_document(encoder, doc_text)
    query_tokens = Encoder.tokenize(encoder, query, type: :query)
    doc_tokens = Encoder.tokenize(encoder, doc_text, type: :document)

    Stephen.Scorer.explain(query_emb, doc_emb, query_tokens, doc_tokens)
  end
end
