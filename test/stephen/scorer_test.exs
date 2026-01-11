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

  describe "normalize/2" do
    test "normalizes score by query length" do
      # Score of 16 with 32 tokens = 0.5
      assert Scorer.normalize(16.0, 32) == 0.5
    end

    test "perfect match gives 1.0" do
      # If all tokens match perfectly, score = query_length
      assert Scorer.normalize(32.0, 32) == 1.0
    end

    test "zero score gives 0.0" do
      assert Scorer.normalize(0.0, 32) == 0.0
    end
  end

  describe "normalize_results/2" do
    test "normalizes all result scores" do
      results = [
        %{doc_id: "a", score: 24.0},
        %{doc_id: "b", score: 16.0},
        %{doc_id: "c", score: 8.0}
      ]

      normalized = Scorer.normalize_results(results, 32)

      assert Enum.at(normalized, 0).score == 0.75
      assert Enum.at(normalized, 1).score == 0.5
      assert Enum.at(normalized, 2).score == 0.25
    end

    test "preserves doc_ids" do
      results = [%{doc_id: "test", score: 16.0}]
      normalized = Scorer.normalize_results(results, 32)

      assert Enum.at(normalized, 0).doc_id == "test"
    end
  end

  describe "normalize_minmax/1" do
    test "scales scores to [0, 1] range" do
      results = [
        %{doc_id: "a", score: 30.0},
        %{doc_id: "b", score: 20.0},
        %{doc_id: "c", score: 10.0}
      ]

      normalized = Scorer.normalize_minmax(results)

      assert Enum.at(normalized, 0).score == 1.0
      assert Enum.at(normalized, 1).score == 0.5
      assert Enum.at(normalized, 2).score == 0.0
    end

    test "handles empty list" do
      assert Scorer.normalize_minmax([]) == []
    end

    test "handles single result" do
      results = [%{doc_id: "a", score: 15.0}]
      normalized = Scorer.normalize_minmax(results)

      assert Enum.at(normalized, 0).score == 1.0
    end

    test "handles all equal scores" do
      results = [
        %{doc_id: "a", score: 10.0},
        %{doc_id: "b", score: 10.0}
      ]

      normalized = Scorer.normalize_minmax(results)

      assert Enum.at(normalized, 0).score == 1.0
      assert Enum.at(normalized, 1).score == 1.0
    end
  end

  describe "fuse_queries/3" do
    test "fuses with :max strategy" do
      query1 = Nx.tensor([[1.0, 0.0, 0.0]])
      query2 = Nx.tensor([[0.0, 1.0, 0.0]])
      doc = Nx.tensor([[1.0, 0.0, 0.0]])

      score = Scorer.fuse_queries([query1, query2], doc, :max)

      # query1 matches perfectly (1.0), query2 doesn't match (0.0)
      # max = 1.0
      assert_in_delta score, 1.0, 0.01
    end

    test "fuses with :avg strategy" do
      query1 = Nx.tensor([[1.0, 0.0, 0.0]])
      query2 = Nx.tensor([[0.0, 1.0, 0.0]])
      doc = Nx.tensor([[1.0, 0.0, 0.0]])

      score = Scorer.fuse_queries([query1, query2], doc, :avg)

      # query1 matches perfectly (1.0), query2 doesn't match (0.0)
      # avg = 0.5
      assert_in_delta score, 0.5, 0.01
    end

    test "fuses with :weighted strategy" do
      query1 = Nx.tensor([[1.0, 0.0, 0.0]])
      query2 = Nx.tensor([[0.0, 1.0, 0.0]])
      doc = Nx.tensor([[1.0, 0.0, 0.0]])

      # Weight query1 at 0.8, query2 at 0.2
      score = Scorer.fuse_queries([query1, query2], doc, {:weighted, [0.8, 0.2]})

      # 1.0 * 0.8 + 0.0 * 0.2 = 0.8
      assert_in_delta score, 0.8, 0.01
    end

    test "handles empty query list" do
      doc = Nx.tensor([[1.0, 0.0, 0.0]])
      assert Scorer.fuse_queries([], doc, :max) == 0.0
    end
  end

  describe "fuse_and_rank/3" do
    test "ranks documents by fused scores" do
      query1 = Nx.tensor([[1.0, 0.0, 0.0]])
      query2 = Nx.tensor([[0.0, 1.0, 0.0]])

      docs = [
        {"doc_x", Nx.tensor([[1.0, 0.0, 0.0]])},
        {"doc_y", Nx.tensor([[0.0, 1.0, 0.0]])},
        {"doc_both", Nx.tensor([[0.707, 0.707, 0.0]])}
      ]

      results = Scorer.fuse_and_rank([query1, query2], docs, :avg)

      assert length(results) == 3
      # doc_both should rank highest with avg strategy (matches both queries partially)
      assert hd(results).doc_id == "doc_both"
    end

    test "respects :max strategy for ranking" do
      query1 = Nx.tensor([[1.0, 0.0, 0.0]])
      query2 = Nx.tensor([[0.0, 0.0, 1.0]])

      docs = [
        {"doc_a", Nx.tensor([[1.0, 0.0, 0.0]])},
        {"doc_b", Nx.tensor([[0.5, 0.5, 0.0]])}
      ]

      results = Scorer.fuse_and_rank([query1, query2], docs, :max)

      # doc_a matches query1 perfectly (max=1.0)
      # doc_b matches query1 partially (max~=0.707)
      assert hd(results).doc_id == "doc_a"
    end
  end

  describe "reciprocal_rank_fusion/2" do
    test "combines ranked lists" do
      list1 = [
        %{doc_id: "a", score: 10.0},
        %{doc_id: "b", score: 8.0},
        %{doc_id: "c", score: 6.0}
      ]

      list2 = [
        %{doc_id: "b", score: 10.0},
        %{doc_id: "a", score: 8.0},
        %{doc_id: "c", score: 6.0}
      ]

      fused = Scorer.reciprocal_rank_fusion([list1, list2])

      assert length(fused) == 3
      # Both a and b appear in top 2 of both lists, should have highest RRF scores
      top_ids = Enum.take(fused, 2) |> Enum.map(& &1.doc_id)
      assert "a" in top_ids
      assert "b" in top_ids
    end

    test "handles empty list" do
      assert Scorer.reciprocal_rank_fusion([]) == []
    end

    test "handles single ranked list" do
      list = [
        %{doc_id: "a", score: 10.0},
        %{doc_id: "b", score: 8.0}
      ]

      fused = Scorer.reciprocal_rank_fusion([list])

      assert length(fused) == 2
      # Ranking should be preserved
      assert hd(fused).doc_id == "a"
    end

    test "handles documents appearing in only one list" do
      list1 = [%{doc_id: "a", score: 10.0}]
      list2 = [%{doc_id: "b", score: 10.0}]

      fused = Scorer.reciprocal_rank_fusion([list1, list2])

      assert length(fused) == 2
      doc_ids = Enum.map(fused, & &1.doc_id)
      assert "a" in doc_ids
      assert "b" in doc_ids
    end

    test "custom k parameter affects scores" do
      list = [
        %{doc_id: "a", score: 10.0},
        %{doc_id: "b", score: 8.0}
      ]

      fused_k60 = Scorer.reciprocal_rank_fusion([list], 60)
      fused_k10 = Scorer.reciprocal_rank_fusion([list], 10)

      # Lower k gives higher RRF scores
      assert hd(fused_k10).score > hd(fused_k60).score
    end
  end

  describe "explain/4" do
    test "returns score and matches" do
      query = Nx.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
      doc = Nx.tensor([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 1.0, 0.0]])

      query_tokens = ["hello", "world"]
      doc_tokens = ["hello", "there", "world"]

      explanation = Scorer.explain(query, doc, query_tokens, doc_tokens)

      assert is_float(explanation.score)
      assert length(explanation.matches) == 2

      first_match = hd(explanation.matches)
      assert first_match.query_token == "hello"
      assert first_match.query_index == 0
      assert first_match.doc_token == "hello"
      assert first_match.doc_index == 0
      assert_in_delta first_match.similarity, 1.0, 0.01
    end

    test "finds best matching doc token for each query token" do
      query = Nx.tensor([[0.0, 1.0, 0.0]])
      doc = Nx.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

      explanation = Scorer.explain(query, doc, ["query"], ["a", "b", "c"])

      match = hd(explanation.matches)
      # Query matches doc token at index 2 ("c")
      assert match.doc_index == 2
      assert match.doc_token == "c"
    end
  end

  describe "format_explanation/2" do
    test "formats explanation as string" do
      explanation = %{
        score: 15.5,
        matches: [
          %{
            query_token: "hello",
            query_index: 0,
            doc_token: "hello",
            doc_index: 0,
            similarity: 0.95
          },
          %{
            query_token: "world",
            query_index: 1,
            doc_token: "earth",
            doc_index: 2,
            similarity: 0.72
          }
        ]
      }

      output = Scorer.format_explanation(explanation)

      assert output =~ "Score: 15.5"
      assert output =~ "hello"
      assert output =~ "0.95"
    end

    test "skips special tokens by default" do
      explanation = %{
        score: 10.0,
        matches: [
          %{
            query_token: "[CLS]",
            query_index: 0,
            doc_token: "[CLS]",
            doc_index: 0,
            similarity: 1.0
          },
          %{
            query_token: "hello",
            query_index: 1,
            doc_token: "hello",
            doc_index: 1,
            similarity: 0.9
          }
        ]
      }

      output = Scorer.format_explanation(explanation)

      refute output =~ "[CLS]"
      assert output =~ "hello"
    end

    test "respects top_k option" do
      explanation = %{
        score: 10.0,
        matches: [
          %{query_token: "a", query_index: 0, doc_token: "x", doc_index: 0, similarity: 0.9},
          %{query_token: "b", query_index: 1, doc_token: "y", doc_index: 1, similarity: 0.8},
          %{query_token: "c", query_index: 2, doc_token: "z", doc_index: 2, similarity: 0.7}
        ]
      }

      output = Scorer.format_explanation(explanation, top_k: 2)

      assert output =~ "0.9"
      assert output =~ "0.8"
      refute output =~ "0.7"
    end

    test "respects min_similarity option" do
      explanation = %{
        score: 10.0,
        matches: [
          %{query_token: "a", query_index: 0, doc_token: "x", doc_index: 0, similarity: 0.9},
          %{query_token: "b", query_index: 1, doc_token: "y", doc_index: 1, similarity: 0.3}
        ]
      }

      output = Scorer.format_explanation(explanation, min_similarity: 0.5)

      assert output =~ "0.9"
      refute output =~ "0.3"
    end
  end
end
