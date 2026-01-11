defmodule Stephen.Retriever do
  @moduledoc """
  High-level retrieval operations combining encoding, indexing, and scoring.

  Implements the full ColBERT retrieval pipeline:
  1. Encode query to per-token embeddings
  2. Search ANN index for candidate documents
  3. Rerank candidates using full MaxSim scoring

  ## Two-Stage Retrieval

  For large collections, use a two-stage retrieval approach:
  1. First stage: Fast candidate retrieval (BM25, dense retriever, etc.)
  2. Second stage: Rerank candidates with ColBERT MaxSim

      # Get candidates from first stage (e.g., BM25)
      candidates = MySearch.bm25_search(query, top_k: 100)

      # Rerank with ColBERT
      results = Stephen.Retriever.rerank(encoder, index, query, candidates)
  """

  alias Stephen.{Encoder, Index, Scorer}
  alias Stephen.Index.Compressed, as: CompressedIndex
  alias Stephen.Plaid

  @type search_result :: %{
          doc_id: term(),
          score: float()
        }

  @doc """
  Searches the index for documents matching the query.

  ## Arguments
    * `encoder` - Loaded encoder from `Encoder.load/1`
    * `index` - Document index from `Index.new/1`
    * `query` - Query string

  ## Options
    * `:top_k` - Number of results to return (default: 10)
    * `:candidates_per_token` - ANN candidates per query token (default: 50)
    * `:rerank` - Whether to rerank with full MaxSim (default: true)

  ## Returns
    List of `%{doc_id: term(), score: float()}` sorted by score descending.
  """
  @spec search(
          Encoder.encoder(),
          Index.t() | Plaid.t() | CompressedIndex.t(),
          String.t(),
          keyword()
        ) :: [search_result()]
  def search(encoder, index, query, opts \\ []) do
    query_embeddings = Encoder.encode_query(encoder, query)
    search_with_embeddings(query_embeddings, index, opts)
  end

  @doc """
  Reranks a list of documents against a query using full MaxSim scoring.

  Supports multiple index types: Index, Plaid, and Index.Compressed.

  ## Arguments
    * `encoder` - Loaded encoder
    * `index` - Document index (Index, Plaid, or Index.Compressed)
    * `query` - Query string
    * `doc_ids` - List of document IDs to rerank

  ## Options
    * `:top_k` - Number of results to return (default: all)

  ## Returns
    List of `%{doc_id: term(), score: float()}` sorted by score descending.

  ## Examples

      # Rerank BM25 candidates
      candidates = [:doc1, :doc2, :doc3]
      results = Stephen.Retriever.rerank(encoder, index, "my query", candidates)

      # Return only top 5
      results = Stephen.Retriever.rerank(encoder, index, query, candidates, top_k: 5)
  """
  @spec rerank(
          Encoder.encoder(),
          Index.t() | Plaid.t() | CompressedIndex.t(),
          String.t(),
          [term()],
          keyword()
        ) :: [search_result()]
  def rerank(encoder, index, query, doc_ids, opts \\ []) do
    top_k = Keyword.get(opts, :top_k, length(doc_ids))
    query_embeddings = Encoder.encode_query(encoder, query)

    doc_ids
    |> Enum.map(fn doc_id ->
      doc_embeddings = get_embeddings_from_index(index, doc_id)

      if doc_embeddings do
        score = Scorer.max_sim(query_embeddings, doc_embeddings)
        %{doc_id: doc_id, score: score}
      else
        %{doc_id: doc_id, score: 0.0}
      end
    end)
    |> Enum.sort_by(& &1.score, :desc)
    |> Enum.take(top_k)
  end

  @doc """
  Reranks documents with pre-computed query embeddings.

  Useful when reranking multiple candidate sets with the same query,
  or when query embeddings are already available.

  ## Arguments
    * `query_embeddings` - Pre-computed query embeddings tensor
    * `index` - Document index
    * `doc_ids` - List of document IDs to rerank

  ## Options
    * `:top_k` - Number of results to return (default: all)

  ## Returns
    List of `%{doc_id: term(), score: float()}` sorted by score descending.
  """
  @spec rerank_with_embeddings(
          Nx.Tensor.t(),
          Index.t() | Plaid.t() | CompressedIndex.t(),
          [term()],
          keyword()
        ) :: [search_result()]
  def rerank_with_embeddings(query_embeddings, index, doc_ids, opts \\ []) do
    top_k = Keyword.get(opts, :top_k, length(doc_ids))

    doc_ids
    |> Enum.map(fn doc_id ->
      doc_embeddings = get_embeddings_from_index(index, doc_id)

      if doc_embeddings do
        score = Scorer.max_sim(query_embeddings, doc_embeddings)
        %{doc_id: doc_id, score: score}
      else
        %{doc_id: doc_id, score: 0.0}
      end
    end)
    |> Enum.sort_by(& &1.score, :desc)
    |> Enum.take(top_k)
  end

  @doc """
  Batch reranks multiple queries against their candidate documents.

  ## Arguments
    * `encoder` - Loaded encoder
    * `index` - Document index
    * `queries_and_candidates` - List of {query, doc_ids} tuples

  ## Options
    * `:top_k` - Number of results per query (default: 10)

  ## Returns
    List of result lists, one per query.
  """
  @spec batch_rerank(
          Encoder.encoder(),
          Index.t() | Plaid.t() | CompressedIndex.t(),
          [{String.t(), [term()]}],
          keyword()
        ) :: [[search_result()]]
  def batch_rerank(encoder, index, queries_and_candidates, opts \\ []) do
    top_k = Keyword.get(opts, :top_k, 10)

    queries_and_candidates
    |> Enum.map(fn {query, doc_ids} ->
      rerank(encoder, index, query, doc_ids, top_k: top_k)
    end)
  end

  @doc """
  Reranks raw text documents against a query without requiring an index.

  Documents are encoded on-the-fly and scored using ColBERT's MaxSim.
  Useful for reranking results from external sources like BM25 or Elasticsearch.

  ## Arguments
    * `encoder` - Loaded encoder
    * `query` - Query string
    * `documents` - List of `{id, text}` tuples to rerank

  ## Options
    * `:top_k` - Number of results to return (default: all)

  ## Returns
    List of `%{doc_id: term(), score: float()}` sorted by score descending.

  ## Examples

      candidates = [
        {"doc1", "Elixir is a dynamic, functional language"},
        {"doc2", "Python is a popular programming language"}
      ]
      results = rerank_texts(encoder, "functional programming", candidates)
  """
  @spec rerank_texts(Encoder.encoder(), String.t(), [{term(), String.t()}], keyword()) ::
          [search_result()]
  def rerank_texts(encoder, query, documents, opts \\ []) do
    query_embeddings = Encoder.encode_query(encoder, query)
    rerank_texts_with_embeddings(query_embeddings, encoder, documents, opts)
  end

  @doc """
  Reranks raw text documents with pre-computed query embeddings.

  Useful when reranking multiple candidate sets with the same query.

  ## Arguments
    * `query_embeddings` - Pre-computed query embeddings tensor
    * `encoder` - Loaded encoder (for encoding documents)
    * `documents` - List of `{id, text}` tuples to rerank

  ## Options
    * `:top_k` - Number of results to return (default: all)

  ## Returns
    List of `%{doc_id: term(), score: float()}` sorted by score descending.
  """
  @spec rerank_texts_with_embeddings(
          Nx.Tensor.t(),
          Encoder.encoder(),
          [{term(), String.t()}],
          keyword()
        ) ::
          [search_result()]
  def rerank_texts_with_embeddings(query_embeddings, encoder, documents, opts \\ []) do
    top_k = Keyword.get(opts, :top_k, length(documents))

    documents
    |> Enum.map(fn {doc_id, text} ->
      doc_embeddings = Encoder.encode_document(encoder, text)
      score = Scorer.max_sim(query_embeddings, doc_embeddings)
      %{doc_id: doc_id, score: score}
    end)
    |> Enum.sort_by(& &1.score, :desc)
    |> Enum.take(top_k)
  end

  @doc """
  Searches for documents matching multiple queries.

  Efficiently encodes all queries together, then searches the index
  for each query independently.

  ## Arguments
    * `encoder` - Loaded encoder
    * `index` - Document index (Index, Plaid, or Index.Compressed)
    * `queries` - List of query strings

  ## Options
    * `:top_k` - Number of results per query (default: 10)
    * `:candidates_per_token` - ANN candidates per query token (default: 50)
    * `:rerank` - Whether to rerank with full MaxSim (default: true)

  ## Returns
    List of result lists, one per query.

  ## Examples

      results = Stephen.Retriever.batch_search(encoder, index, ["query 1", "query 2"])
      # results[0] contains top_k results for "query 1"
      # results[1] contains top_k results for "query 2"
  """
  @spec batch_search(
          Encoder.encoder(),
          Index.t() | Plaid.t() | CompressedIndex.t(),
          [String.t()],
          keyword()
        ) :: [[search_result()]]
  def batch_search(encoder, index, queries, opts \\ []) do
    # Batch encode all queries
    query_embeddings_list = Encoder.encode_queries(encoder, queries)

    # Search for each query
    Enum.map(query_embeddings_list, fn query_embeddings ->
      search_with_embeddings(query_embeddings, index, opts)
    end)
  end

  @doc """
  Searches for documents matching a query using pre-computed embeddings.

  ## Arguments
    * `query_embeddings` - Pre-computed query embeddings tensor
    * `index` - Document index (Index, Plaid, or Index.Compressed)

  ## Options
    * `:top_k` - Number of results to return (default: 10)
    * `:candidates_per_token` - ANN candidates per query token (default: 50)
    * `:rerank` - Whether to rerank with full MaxSim (default: true)
    * `:nprobe` - Number of centroids to probe for Plaid/Compressed (default: 32)

  ## Returns
    List of `%{doc_id: term(), score: float()}` sorted by score descending.
  """
  @spec search_with_embeddings(
          Nx.Tensor.t(),
          Index.t() | Plaid.t() | CompressedIndex.t(),
          keyword()
        ) :: [search_result()]
  def search_with_embeddings(query_embeddings, index, opts \\ [])

  def search_with_embeddings(query_embeddings, %Index{} = index, opts) do
    top_k = Keyword.get(opts, :top_k, 10)
    candidates_per_token = Keyword.get(opts, :candidates_per_token, 50)
    rerank = Keyword.get(opts, :rerank, true)

    candidates = Index.search_tokens(index, query_embeddings, candidates_per_token)

    if rerank do
      rerank_candidates(query_embeddings, candidates, index, top_k)
    else
      format_candidates_as_results(candidates, top_k)
    end
  end

  def search_with_embeddings(query_embeddings, %Plaid{} = index, opts) do
    top_k = Keyword.get(opts, :top_k, 10)
    nprobe = Keyword.get(opts, :nprobe, 32)

    Plaid.search(index, query_embeddings, top_k: top_k, nprobe: nprobe)
  end

  def search_with_embeddings(query_embeddings, %CompressedIndex{} = index, opts) do
    top_k = Keyword.get(opts, :top_k, 10)
    nprobe = Keyword.get(opts, :nprobe, 32)

    CompressedIndex.search(index, query_embeddings, top_k: top_k, nprobe: nprobe)
  end

  defp format_candidates_as_results(candidates, top_k) do
    candidates
    |> Enum.sort_by(fn {_doc_id, count} -> count end, :desc)
    |> Enum.take(top_k)
    |> Enum.map(fn {doc_id, count} -> %{doc_id: doc_id, score: count / 1.0} end)
  end

  @doc """
  Indexes a list of documents.

  ## Arguments
    * `encoder` - Loaded encoder
    * `index` - Document index
    * `documents` - List of `{doc_id, text}` tuples

  ## Returns
    Updated index with all documents added.
  """
  @spec index_documents(Encoder.encoder(), Index.t(), [{term(), String.t()}]) :: Index.t()
  def index_documents(encoder, index, documents) do
    documents
    |> Enum.reduce(index, fn {doc_id, text}, acc ->
      embeddings = Encoder.encode_document(encoder, text)
      Index.add(acc, doc_id, embeddings)
    end)
  end

  # Rerank candidates with full MaxSim scoring
  defp rerank_candidates(query_embeddings, candidates, index, top_k) do
    candidates
    |> Map.keys()
    |> Enum.map(fn doc_id ->
      doc_embeddings = get_embeddings_from_index(index, doc_id)
      score = Scorer.max_sim(query_embeddings, doc_embeddings)
      %{doc_id: doc_id, score: score}
    end)
    |> Enum.sort_by(& &1.score, :desc)
    |> Enum.take(top_k)
  end

  # Helper to get embeddings from different index types
  defp get_embeddings_from_index(%Index{} = index, doc_id) do
    Index.get_embeddings(index, doc_id)
  end

  defp get_embeddings_from_index(%Plaid{} = index, doc_id) do
    Plaid.get_embeddings(index, doc_id)
  end

  defp get_embeddings_from_index(%CompressedIndex{} = index, doc_id) do
    CompressedIndex.get_embeddings(index, doc_id)
  end

  defp get_embeddings_from_index(_index, _doc_id), do: nil
end
