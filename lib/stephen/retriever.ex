defmodule Stephen.Retriever do
  @moduledoc """
  High-level retrieval operations combining encoding, indexing, and scoring.

  Implements the full ColBERT retrieval pipeline:
  1. Encode query to per-token embeddings
  2. Search ANN index for candidate documents
  3. Rerank candidates using full MaxSim scoring
  """

  alias Stephen.{Encoder, Index, Scorer}

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
  @spec search(Encoder.encoder(), Index.t(), String.t(), keyword()) :: [search_result()]
  def search(encoder, index, query, opts \\ []) do
    top_k = Keyword.get(opts, :top_k, 10)
    candidates_per_token = Keyword.get(opts, :candidates_per_token, 50)
    rerank = Keyword.get(opts, :rerank, true)

    # Encode query
    query_embeddings = Encoder.encode_query(encoder, query)

    # Get candidate documents via ANN search
    candidates = Index.search_tokens(index, query_embeddings, candidates_per_token)

    if rerank do
      # Rerank candidates with full MaxSim scoring
      rerank_candidates(query_embeddings, candidates, index, top_k)
    else
      # Return candidates sorted by token match count
      candidates
      |> Enum.sort_by(fn {_doc_id, count} -> count end, :desc)
      |> Enum.take(top_k)
      |> Enum.map(fn {doc_id, count} -> %{doc_id: doc_id, score: count / 1.0} end)
    end
  end

  @doc """
  Reranks a list of documents against a query using full MaxSim scoring.

  ## Arguments
    * `encoder` - Loaded encoder
    * `index` - Document index
    * `query` - Query string
    * `doc_ids` - List of document IDs to rerank

  ## Returns
    List of `%{doc_id: term(), score: float()}` sorted by score descending.
  """
  @spec rerank(Encoder.encoder(), Index.t(), String.t(), [term()]) :: [search_result()]
  def rerank(encoder, index, query, doc_ids) do
    query_embeddings = Encoder.encode_query(encoder, query)

    doc_ids
    |> Enum.map(fn doc_id ->
      doc_embeddings = Index.get_embeddings(index, doc_id)

      if doc_embeddings do
        score = Scorer.max_sim(query_embeddings, doc_embeddings)
        %{doc_id: doc_id, score: score}
      else
        %{doc_id: doc_id, score: 0.0}
      end
    end)
    |> Enum.sort_by(& &1.score, :desc)
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
      doc_embeddings = Index.get_embeddings(index, doc_id)
      score = Scorer.max_sim(query_embeddings, doc_embeddings)
      %{doc_id: doc_id, score: score}
    end)
    |> Enum.sort_by(& &1.score, :desc)
    |> Enum.take(top_k)
  end
end
