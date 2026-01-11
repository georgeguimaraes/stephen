defmodule StephenColbert do
  @moduledoc """
  ColBERT-style neural retrieval for Elixir.

  StephenColbert implements late interaction retrieval using per-token
  embeddings and MaxSim scoring, enabling high-quality semantic search.

  ## Quick Start

      # Load the encoder
      {:ok, encoder} = StephenColbert.load_encoder()

      # Create an index
      index = StephenColbert.new_index(encoder)

      # Index some documents
      documents = [
        {"doc1", "The quick brown fox jumps over the lazy dog"},
        {"doc2", "Machine learning is a subset of artificial intelligence"},
        {"doc3", "Elixir is a functional programming language"}
      ]
      index = StephenColbert.index(encoder, index, documents)

      # Search
      results = StephenColbert.search(encoder, index, "programming languages")
      # => [%{doc_id: "doc3", score: 12.5}, ...]

  ## Architecture

  StephenColbert uses the ColBERT architecture:

  1. **Encoder**: Converts text to per-token embeddings using BERT
  2. **Index**: Stores document embeddings with ANN search via HNSWLib
  3. **Scorer**: Computes MaxSim scores for late interaction
  4. **Retriever**: Orchestrates the full retrieval pipeline
  """

  alias StephenColbert.{Encoder, Index, Retriever}

  @type encoder :: Encoder.encoder()
  @type index :: Index.t()
  @type doc_id :: term()
  @type search_result :: Retriever.search_result()

  @doc """
  Loads the encoder model.

  ## Options
    * `:model` - HuggingFace model name (default: sentence-transformers/all-MiniLM-L6-v2)

  ## Examples

      {:ok, encoder} = StephenColbert.load_encoder()
      {:ok, encoder} = StephenColbert.load_encoder(model: "bert-base-uncased")
  """
  @spec load_encoder(keyword()) :: {:ok, encoder()} | {:error, term()}
  defdelegate load_encoder(opts \\ []), to: Encoder, as: :load

  @doc """
  Creates a new empty index.

  ## Options
    * `:max_elements` - Maximum number of token embeddings (default: 100_000)
    * `:m` - HNSW M parameter (default: 16)
    * `:ef_construction` - HNSW ef_construction (default: 200)

  ## Examples

      index = StephenColbert.new_index(encoder)
      index = StephenColbert.new_index(encoder, max_elements: 1_000_000)
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
      index = StephenColbert.index(encoder, index, documents)
  """
  @spec index(encoder(), index(), [{doc_id(), String.t()}]) :: index()
  defdelegate index(encoder, index, documents), to: Retriever, as: :index_documents

  @doc """
  Searches for documents matching a query.

  ## Options
    * `:top_k` - Number of results to return (default: 10)
    * `:rerank` - Whether to rerank with full MaxSim (default: true)

  ## Examples

      results = StephenColbert.search(encoder, index, "hello")
      # => [%{doc_id: "doc1", score: 15.2}, %{doc_id: "doc2", score: 8.1}]
  """
  @spec search(encoder(), index(), String.t(), keyword()) :: [search_result()]
  defdelegate search(encoder, index, query, opts \\ []), to: Retriever

  @doc """
  Reranks a list of documents against a query.

  Useful when you have candidate documents from another source
  (e.g., keyword search) and want to rerank them semantically.

  ## Examples

      results = StephenColbert.rerank(encoder, index, "hello", ["doc1", "doc2"])
  """
  @spec rerank(encoder(), index(), String.t(), [doc_id()]) :: [search_result()]
  defdelegate rerank(encoder, index, query, doc_ids), to: Retriever

  @doc """
  Saves an index to disk.

  ## Examples

      :ok = StephenColbert.save_index(index, "/path/to/index")
  """
  @spec save_index(index(), Path.t()) :: :ok | {:error, term()}
  defdelegate save_index(index, path), to: Index, as: :save

  @doc """
  Loads an index from disk.

  ## Examples

      {:ok, index} = StephenColbert.load_index("/path/to/index")
  """
  @spec load_index(Path.t()) :: {:ok, index()} | {:error, term()}
  defdelegate load_index(path), to: Index, as: :load
end
