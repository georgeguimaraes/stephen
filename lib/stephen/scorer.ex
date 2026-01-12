defmodule Stephen.Scorer do
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

      score = Stephen.Scorer.max_sim(query_emb, doc_emb)
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
  Computes MaxSim scores for multiple queries against multiple documents.

  Each query is scored against each document, returning a matrix of scores.

  ## Arguments
    * `query_embeddings_list` - List of query tensors, each of shape {query_len, dim}
    * `doc_embeddings_list` - List of document tensors, each of shape {doc_len, dim}

  ## Returns
    List of lists where result[i][j] is the score of query i against doc j.

  ## Examples

      scores = Stephen.Scorer.multi_max_sim(queries, docs)
      # scores[0][1] is score of first query against second doc
  """
  @spec multi_max_sim([Nx.Tensor.t()], [Nx.Tensor.t()]) :: [[score()]]
  def multi_max_sim(query_embeddings_list, doc_embeddings_list) do
    Enum.map(query_embeddings_list, fn query_emb ->
      max_sim_batch(query_emb, doc_embeddings_list)
    end)
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

  @doc """
  Explains the MaxSim scoring between query and document.

  Returns detailed information about which query tokens matched which
  document tokens, useful for debugging and understanding retrieval results.

  ## Arguments
    * `query_embeddings` - Query token embeddings
    * `doc_embeddings` - Document token embeddings
    * `query_tokens` - List of query token strings
    * `doc_tokens` - List of document token strings

  ## Returns
    Map containing:
    * `:score` - Total MaxSim score
    * `:matches` - List of match details for each query token, including:
      * `:query_token` - Query token string
      * `:query_index` - Query token index
      * `:doc_token` - Best matching document token string
      * `:doc_index` - Best matching document token index
      * `:similarity` - Cosine similarity (contribution to score)

  ## Examples

      query_emb = Encoder.encode_query(encoder, "satirical comedy")
      doc_emb = Encoder.encode_document(encoder, "Colbert is satirical")
      query_tokens = Encoder.tokenize(encoder, "satirical comedy", type: :query)
      doc_tokens = Encoder.tokenize(encoder, "Colbert is satirical")

      explanation = Scorer.explain(query_emb, doc_emb, query_tokens, doc_tokens)
      # => %{
      #   score: 15.2,
      #   matches: [
      #     %{query_token: "satirical", doc_token: "satirical", similarity: 0.95, ...},
      #     %{query_token: "comedy", doc_token: "Colbert", similarity: 0.42, ...},
      #     ...
      #   ]
      # }
  """
  @spec explain(Nx.Tensor.t(), Nx.Tensor.t(), [String.t()], [String.t()]) :: map()
  def explain(query_embeddings, doc_embeddings, query_tokens, doc_tokens) do
    sim_matrix = similarity_matrix(query_embeddings, doc_embeddings)

    # For each query token, find the best matching doc token
    max_similarities = Nx.reduce_max(sim_matrix, axes: [1])
    best_doc_indices = Nx.argmax(sim_matrix, axis: 1)

    max_sims_list = Nx.to_flat_list(max_similarities)
    best_indices_list = Nx.to_flat_list(best_doc_indices)

    matches =
      query_tokens
      |> Enum.with_index()
      |> Enum.zip(Enum.zip(max_sims_list, best_indices_list))
      |> Enum.map(fn {{query_token, query_idx}, {similarity, doc_idx}} ->
        doc_token =
          if doc_idx < length(doc_tokens) do
            Enum.at(doc_tokens, doc_idx)
          else
            "[OUT_OF_RANGE]"
          end

        %{
          query_token: query_token,
          query_index: query_idx,
          doc_token: doc_token,
          doc_index: doc_idx,
          similarity: similarity
        }
      end)

    score = Enum.sum(max_sims_list)

    %{
      score: score,
      matches: matches
    }
  end

  @doc """
  Formats an explanation for display.

  Takes the output of `explain/4` and returns a human-readable string.

  ## Options
    * `:top_k` - Only show top-k matches by similarity (default: all)
    * `:skip_special` - Skip special tokens like [CLS], [SEP], [MASK] (default: true)
    * `:min_similarity` - Only show matches above threshold (default: 0.0)

  ## Examples

      explanation = Scorer.explain(query_emb, doc_emb, query_tokens, doc_tokens)
      IO.puts(Scorer.format_explanation(explanation))
      # Score: 15.20
      #
      # Query Token          -> Doc Token            Similarity
      # --------------------------------------------------------
      # satirical           -> satirical           0.95
      # comedy               -> host                0.72
      # ...
  """
  @spec format_explanation(map(), keyword()) :: String.t()
  def format_explanation(explanation, opts \\ []) do
    top_k = Keyword.get(opts, :top_k, nil)
    skip_special = Keyword.get(opts, :skip_special, true)
    min_sim = Keyword.get(opts, :min_similarity, 0.0)

    special_tokens = MapSet.new(["[CLS]", "[SEP]", "[PAD]", "[MASK]", "[Q]", "[D]"])

    matches =
      explanation.matches
      |> Enum.reject(fn m ->
        skip_special and MapSet.member?(special_tokens, m.query_token)
      end)
      |> Enum.filter(fn m -> m.similarity >= min_sim end)
      |> Enum.sort_by(& &1.similarity, :desc)

    matches = if top_k, do: Enum.take(matches, top_k), else: matches

    header = "Score: #{Float.round(explanation.score, 2)}\n\n"

    col_header =
      String.pad_trailing("Query Token", 20) <>
        " -> " <>
        String.pad_trailing("Doc Token", 20) <> " Similarity\n"

    separator = String.duplicate("-", 60) <> "\n"

    rows =
      Enum.map_join(matches, "\n", fn m ->
        q = String.pad_trailing(String.slice(m.query_token, 0, 20), 20)
        d = String.pad_trailing(String.slice(m.doc_token, 0, 20), 20)
        s = Float.round(m.similarity, 4)
        "#{q} -> #{d} #{s}"
      end)

    header <> col_header <> separator <> rows
  end

  @doc """
  Normalizes a MaxSim score to [0, 1] range.

  Since embeddings are L2-normalized, the maximum per-token similarity is 1.0.
  The theoretical maximum score is therefore `query_length`.

  ## Arguments
    * `score` - Raw MaxSim score
    * `query_length` - Number of query tokens used in scoring

  ## Returns
    Normalized score in [0, 1] range.

  ## Examples

      raw_score = Stephen.Scorer.max_sim(query_emb, doc_emb)
      normalized = Stephen.Scorer.normalize(raw_score, 32)
      # => 0.73
  """
  @spec normalize(score(), pos_integer()) :: float()
  def normalize(score, query_length) when query_length > 0 do
    score / query_length
  end

  @doc """
  Normalizes search results to [0, 1] range.

  Takes a list of search results and normalizes their scores based on
  the query length. Useful for setting thresholds or comparing results
  across different queries.

  ## Arguments
    * `results` - List of `%{doc_id: term(), score: float()}` maps
    * `query_length` - Number of query tokens used in scoring

  ## Returns
    Results with normalized scores.

  ## Examples

      results = Stephen.search(encoder, index, "late night comedy")
      normalized = Stephen.Scorer.normalize_results(results, 32)
      high_quality = Enum.filter(normalized, & &1.score > 0.7)
  """
  @spec normalize_results([map()], pos_integer()) :: [map()]
  def normalize_results(results, query_length) when query_length > 0 do
    Enum.map(results, fn result ->
      %{result | score: result.score / query_length}
    end)
  end

  @doc """
  Normalizes results using min-max scaling within the result set.

  Scales scores so the highest is 1.0 and lowest is 0.0. Useful when
  you want relative ranking within results rather than absolute scores.

  ## Arguments
    * `results` - List of `%{doc_id: term(), score: float()}` maps

  ## Returns
    Results with scores scaled to [0, 1] range.

  ## Examples

      results = Stephen.search(encoder, index, query)
      normalized = Stephen.Scorer.normalize_minmax(results)
  """
  @spec normalize_minmax([map()]) :: [map()]
  def normalize_minmax([]), do: []
  def normalize_minmax([single]), do: [%{single | score: 1.0}]

  def normalize_minmax(results) do
    scores = Enum.map(results, & &1.score)
    min_score = Enum.min(scores)
    max_score = Enum.max(scores)
    range = max_score - min_score

    if range == 0 do
      Enum.map(results, fn result -> %{result | score: 1.0} end)
    else
      Enum.map(results, fn result ->
        %{result | score: (result.score - min_score) / range}
      end)
    end
  end

  # Multi-Query Fusion Functions

  @doc """
  Fuses scores from multiple queries using the specified strategy.

  Combines scores from multiple query variants (e.g., query expansions,
  reformulations) into a single ranking.

  ## Arguments
    * `query_embeddings_list` - List of query embedding tensors
    * `doc_embeddings` - Document embedding tensor
    * `strategy` - Fusion strategy: `:max`, `:avg`, or `{:weighted, weights}`

  ## Strategies
    * `:max` - Takes the maximum score across all queries (good for OR semantics)
    * `:avg` - Averages scores across queries (good for ensemble)
    * `{:weighted, weights}` - Weighted average with custom weights per query

  ## Examples

      # Query expansion: original + synonyms
      queries = [
        Encoder.encode_query(encoder, "late night host"),
        Encoder.encode_query(encoder, "talk show comedian"),
        Encoder.encode_query(encoder, "comedy television")
      ]
      score = Scorer.fuse_queries(queries, doc_emb, :max)

      # Weighted fusion: prioritize original query
      score = Scorer.fuse_queries(queries, doc_emb, {:weighted, [0.6, 0.2, 0.2]})
  """
  @spec fuse_queries([Nx.Tensor.t()], Nx.Tensor.t(), :max | :avg | {:weighted, [float()]}) ::
          score()
  def fuse_queries([], _doc_embeddings, _strategy), do: 0.0

  def fuse_queries(query_embeddings_list, doc_embeddings, strategy) do
    scores = Enum.map(query_embeddings_list, &max_sim(&1, doc_embeddings))

    case strategy do
      :max ->
        Enum.max(scores)

      :avg ->
        Enum.sum(scores) / length(scores)

      {:weighted, weights} when length(weights) == length(scores) ->
        scores
        |> Enum.zip(weights)
        |> Enum.map(fn {score, weight} -> score * weight end)
        |> Enum.sum()
    end
  end

  @doc """
  Fuses and ranks documents using multiple queries.

  Scores each document against all queries and combines using the specified
  fusion strategy, returning ranked results.

  ## Arguments
    * `query_embeddings_list` - List of query embedding tensors
    * `doc_embeddings_list` - List of `{doc_id, embeddings}` tuples
    * `strategy` - Fusion strategy: `:max`, `:avg`, or `{:weighted, weights}`

  ## Returns
    List of `%{doc_id: term(), score: float()}` maps sorted by score descending.

  ## Examples

      queries = [query1_emb, query2_emb]
      docs = [{"doc1", emb1}, {"doc2", emb2}]
      results = Scorer.fuse_and_rank(queries, docs, :avg)
  """
  @spec fuse_and_rank(
          [Nx.Tensor.t()],
          [{term(), Nx.Tensor.t()}],
          :max | :avg | {:weighted, [float()]}
        ) ::
          [map()]
  def fuse_and_rank(query_embeddings_list, doc_embeddings_list, strategy) do
    doc_embeddings_list
    |> Enum.map(fn {doc_id, doc_emb} ->
      score = fuse_queries(query_embeddings_list, doc_emb, strategy)
      %{doc_id: doc_id, score: score}
    end)
    |> Enum.sort_by(& &1.score, :desc)
  end

  @doc """
  Reciprocal Rank Fusion (RRF) for combining multiple ranked lists.

  RRF is a robust fusion method that combines rankings rather than raw scores,
  making it effective when score distributions differ across queries.

  ## Arguments
    * `ranked_lists` - List of ranked result lists, each `[%{doc_id: term(), score: float()}, ...]`
    * `k` - Smoothing constant (default: 60). Higher values reduce the impact of top ranks.

  ## Returns
    Fused results sorted by RRF score descending.

  ## Examples

      results1 = Retriever.search_with_embeddings(query1, index)
      results2 = Retriever.search_with_embeddings(query2, index)
      fused = Scorer.reciprocal_rank_fusion([results1, results2])

  ## References
    Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).
    Reciprocal rank fusion outperforms condorcet and individual rank learning methods.
  """
  @spec reciprocal_rank_fusion([[map()]], pos_integer()) :: [map()]
  def reciprocal_rank_fusion(ranked_lists, k \\ 60)
  def reciprocal_rank_fusion([], _k), do: []

  def reciprocal_rank_fusion(ranked_lists, k) do
    ranked_lists
    |> Enum.flat_map(fn results ->
      results
      |> Enum.with_index(1)
      |> Enum.map(fn {result, rank} ->
        {result.doc_id, 1.0 / (k + rank)}
      end)
    end)
    |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))
    |> Enum.map(fn {doc_id, rrf_scores} ->
      %{doc_id: doc_id, score: Enum.sum(rrf_scores)}
    end)
    |> Enum.sort_by(& &1.score, :desc)
  end
end
