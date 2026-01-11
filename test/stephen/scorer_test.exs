defmodule Stephen.ScorerTest do
  use ExUnit.Case, async: true

  alias Stephen.Scorer

  describe "max_sim/2" do
    test "computes max similarity for identical embeddings" do
      # Two identical embeddings should have high similarity
      embeddings = Nx.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
      score = Scorer.max_sim(embeddings, embeddings)

      # Each query token matches itself perfectly (similarity = 1.0)
      # Sum of 2 tokens = 2.0
      assert_in_delta score, 2.0, 0.01
    end

    test "computes max similarity for orthogonal embeddings" do
      query = Nx.tensor([[1.0, 0.0, 0.0]])
      doc = Nx.tensor([[0.0, 1.0, 0.0]])

      score = Scorer.max_sim(query, doc)

      # Orthogonal vectors have 0 cosine similarity
      assert_in_delta score, 0.0, 0.01
    end

    test "computes max similarity for multiple query tokens" do
      # Query with 2 tokens
      query = Nx.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
      # Doc with 3 tokens, where first matches query[0] and second matches query[1]
      doc = Nx.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]])

      score = Scorer.max_sim(query, doc)

      # Query token 0 best matches doc token 0 (sim = 1.0)
      # Query token 1 best matches doc token 1 (sim = 1.0)
      # Total = 2.0
      assert_in_delta score, 2.0, 0.01
    end

    test "handles normalized embeddings correctly" do
      # L2 normalized vectors
      query = Nx.tensor([[0.6, 0.8, 0.0]])
      doc = Nx.tensor([[0.6, 0.8, 0.0], [1.0, 0.0, 0.0]])

      score = Scorer.max_sim(query, doc)

      # Query matches doc[0] perfectly (same vector)
      assert_in_delta score, 1.0, 0.01
    end
  end

  describe "max_sim_batch/2" do
    test "scores multiple documents" do
      query = Nx.tensor([[1.0, 0.0, 0.0]])

      docs = [
        Nx.tensor([[1.0, 0.0, 0.0]]),
        Nx.tensor([[0.0, 1.0, 0.0]]),
        Nx.tensor([[0.5, 0.5, 0.0]])
      ]

      scores = Scorer.max_sim_batch(query, docs)

      assert length(scores) == 3
      assert_in_delta Enum.at(scores, 0), 1.0, 0.01
      assert_in_delta Enum.at(scores, 1), 0.0, 0.01
    end
  end

  describe "rank/2" do
    test "ranks documents by score descending" do
      query = Nx.tensor([[1.0, 0.0, 0.0]])

      docs = [
        {"doc_low", Nx.tensor([[0.0, 1.0, 0.0]])},
        {"doc_high", Nx.tensor([[1.0, 0.0, 0.0]])},
        {"doc_mid", Nx.tensor([[0.5, 0.5, 0.0]])}
      ]

      ranked = Scorer.rank(query, docs)

      assert length(ranked) == 3
      assert elem(Enum.at(ranked, 0), 0) == "doc_high"
      assert elem(Enum.at(ranked, 2), 0) == "doc_low"
    end
  end

  describe "similarity_matrix/2" do
    test "returns matrix with correct shape" do
      query = Nx.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
      doc = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])

      matrix = Scorer.similarity_matrix(query, doc)

      assert Nx.shape(matrix) == {3, 2}
    end
  end

  describe "multi_max_sim/2" do
    test "scores multiple queries against multiple documents" do
      queries = [
        Nx.tensor([[1.0, 0.0, 0.0]]),
        Nx.tensor([[0.0, 1.0, 0.0]])
      ]

      docs = [
        Nx.tensor([[1.0, 0.0, 0.0]]),
        Nx.tensor([[0.0, 1.0, 0.0]]),
        Nx.tensor([[0.5, 0.5, 0.0]])
      ]

      scores = Scorer.multi_max_sim(queries, docs)

      # Returns list of lists: scores[query_idx][doc_idx]
      assert length(scores) == 2
      assert length(Enum.at(scores, 0)) == 3

      # Query 0 matches doc 0 best
      assert_in_delta Enum.at(Enum.at(scores, 0), 0), 1.0, 0.01
      assert_in_delta Enum.at(Enum.at(scores, 0), 1), 0.0, 0.01

      # Query 1 matches doc 1 best
      assert_in_delta Enum.at(Enum.at(scores, 1), 0), 0.0, 0.01
      assert_in_delta Enum.at(Enum.at(scores, 1), 1), 1.0, 0.01
    end

    test "handles empty query list" do
      docs = [Nx.tensor([[1.0, 0.0, 0.0]])]
      scores = Scorer.multi_max_sim([], docs)
      assert scores == []
    end

    test "handles empty doc list" do
      queries = [Nx.tensor([[1.0, 0.0, 0.0]])]
      scores = Scorer.multi_max_sim(queries, [])
      assert scores == [[]]
    end
  end
end
