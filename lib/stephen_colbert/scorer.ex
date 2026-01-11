defmodule StephenColbert.Scorer do
  @moduledoc """
  Implements ColBERT's late interaction scoring mechanism (MaxSim).

  MaxSim computes the relevance score between a query and document by:
  1. Computing cosine similarity between all query-document token pairs
  2. For each query token, taking the maximum similarity to any document token
  3. Summing these maximum similarities

  This "late interaction" approach captures fine-grained token-level matching
  while remaining efficient for retrieval.
  """

  import Nx.Defn

  @type score :: float()

  @doc """
  Computes the MaxSim score between query and document embeddings.

  ## Arguments
    * `query_embeddings` - Tensor of shape {query_len, dim}
    * `doc_embeddings` - Tensor of shape {doc_len, dim}

  ## Returns
    A scalar float representing the relevance score.

  ## Examples

      score = StephenColbert.Scorer.max_sim(query_emb, doc_emb)
  """
  @spec max_sim(Nx.Tensor.t(), Nx.Tensor.t()) :: score()
  def max_sim(query_embeddings, doc_embeddings) do
    max_sim_nx(query_embeddings, doc_embeddings)
    |> Nx.to_number()
  end

  @doc """
  Computes MaxSim scores for a query against multiple documents.

  ## Arguments
    * `query_embeddings` - Tensor of shape {query_len, dim}
    * `doc_embeddings_list` - List of tensors, each of shape {doc_len, dim}

  ## Returns
    List of scores in the same order as the input documents.
  """
  @spec max_sim_batch(Nx.Tensor.t(), [Nx.Tensor.t()]) :: [score()]
  def max_sim_batch(query_embeddings, doc_embeddings_list) do
    Enum.map(doc_embeddings_list, &max_sim(query_embeddings, &1))
  end

  @doc """
  Ranks documents by their MaxSim scores against a query.

  ## Arguments
    * `query_embeddings` - Tensor of shape {query_len, dim}
    * `doc_embeddings_list` - List of {doc_id, embeddings} tuples

  ## Returns
    List of {doc_id, score} tuples sorted by score descending.
  """
  @spec rank(Nx.Tensor.t(), [{term(), Nx.Tensor.t()}]) :: [{term(), score()}]
  def rank(query_embeddings, doc_embeddings_list) do
    doc_embeddings_list
    |> Enum.map(fn {doc_id, embeddings} ->
      {doc_id, max_sim(query_embeddings, embeddings)}
    end)
    |> Enum.sort_by(fn {_id, score} -> score end, :desc)
  end

  # Nx defn for efficient MaxSim computation
  defn max_sim_nx(query_embeddings, doc_embeddings) do
    # Compute similarity matrix: {query_len, doc_len}
    # Since embeddings are L2 normalized, dot product = cosine similarity
    similarity_matrix = Nx.dot(query_embeddings, Nx.transpose(doc_embeddings))

    # For each query token, take max similarity across all doc tokens
    max_similarities = Nx.reduce_max(similarity_matrix, axes: [1])

    # Sum the max similarities
    Nx.sum(max_similarities)
  end

  @doc """
  Computes the similarity matrix between query and document tokens.

  Useful for visualization and debugging.

  ## Returns
    Tensor of shape {query_len, doc_len} with cosine similarities.
  """
  @spec similarity_matrix(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def similarity_matrix(query_embeddings, doc_embeddings) do
    Nx.dot(query_embeddings, Nx.transpose(doc_embeddings))
  end
end
