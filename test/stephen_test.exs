defmodule StephenTest do
  use ExUnit.Case

  @moduletag :integration
  @moduletag timeout: :infinity

  describe "full pipeline" do
    @tag :slow
    test "indexes and searches documents" do
      # This test requires downloading model weights, so skip in CI
      # To run: mix test --include slow

      {:ok, encoder} = Stephen.load_encoder()
      index = Stephen.new_index(encoder)

      documents = [
        {"doc1", "The quick brown fox jumps over the lazy dog"},
        {"doc2", "Machine learning is a subset of artificial intelligence"},
        {"doc3", "Elixir is a functional programming language built on Erlang"}
      ]

      index = Stephen.index(encoder, index, documents)

      # Search for programming-related content
      results = Stephen.search(encoder, index, "programming languages", top_k: 3)

      assert length(results) == 3
      assert is_float(hd(results).score)

      # The programming document should rank high
      doc_ids = Enum.map(results, & &1.doc_id)
      assert "doc3" in doc_ids
    end

    @tag :slow
    test "reranks documents" do
      {:ok, encoder} = Stephen.load_encoder()
      index = Stephen.new_index(encoder)

      documents = [
        {"doc1", "Python is popular for data science"},
        {"doc2", "Elixir excels at concurrent programming"},
        {"doc3", "JavaScript runs in web browsers"}
      ]

      index = Stephen.index(encoder, index, documents)

      # Rerank specific documents
      results =
        Stephen.rerank(encoder, index, "concurrent systems", ["doc1", "doc2", "doc3"])

      assert length(results) == 3
      # Elixir doc should rank highest for "concurrent systems"
      assert hd(results).doc_id == "doc2"
    end

    @tag :slow
    test "batch search with multiple queries" do
      {:ok, encoder} = Stephen.load_encoder()
      index = Stephen.new_index(encoder)

      documents = [
        {"doc1", "Python is popular for data science and machine learning"},
        {"doc2", "Elixir excels at concurrent programming and fault tolerance"},
        {"doc3", "JavaScript runs in web browsers and Node.js servers"},
        {"doc4", "Rust provides memory safety without garbage collection"}
      ]

      index = Stephen.index(encoder, index, documents)

      # Batch search multiple queries
      queries = ["machine learning", "concurrent systems", "memory management"]
      results = Stephen.Retriever.batch_search(encoder, index, queries, top_k: 2)

      assert length(results) == 3
      assert length(Enum.at(results, 0)) == 2
      assert length(Enum.at(results, 1)) == 2
      assert length(Enum.at(results, 2)) == 2

      # First query about ML should rank doc1 high
      ml_results = Enum.at(results, 0)
      ml_doc_ids = Enum.map(ml_results, & &1.doc_id)
      assert "doc1" in ml_doc_ids

      # Second query about concurrency should rank doc2 high
      concurrent_results = Enum.at(results, 1)
      concurrent_doc_ids = Enum.map(concurrent_results, & &1.doc_id)
      assert "doc2" in concurrent_doc_ids

      # Third query about memory should rank doc4 high
      memory_results = Enum.at(results, 2)
      memory_doc_ids = Enum.map(memory_results, & &1.doc_id)
      assert "doc4" in memory_doc_ids
    end

    @tag :slow
    test "batch rerank with multiple query-candidate pairs" do
      {:ok, encoder} = Stephen.load_encoder()
      index = Stephen.new_index(encoder)

      documents = [
        {"doc1", "Python is the best language for data science and analytics"},
        {"doc2", "Elixir excels at concurrent programming and fault tolerance"},
        {"doc3", "JavaScript runs in web browsers"},
        {"doc4", "Rust provides memory safety"}
      ]

      index = Stephen.index(encoder, index, documents)

      # Batch rerank with different candidates for each query
      queries_and_candidates = [
        {"python data science analytics", ["doc1", "doc2", "doc3"]},
        {"concurrent fault tolerant distributed systems", ["doc2", "doc3", "doc4"]}
      ]

      results = Stephen.Retriever.batch_rerank(encoder, index, queries_and_candidates)

      assert length(results) == 2

      # Each result set should have 3 docs (all candidates)
      first_results = Enum.at(results, 0)
      second_results = Enum.at(results, 1)
      assert length(first_results) == 3
      assert length(second_results) == 3

      # All results should have scores
      assert Enum.all?(first_results, &is_float(&1.score))
      assert Enum.all?(second_results, &is_float(&1.score))

      # Results should be sorted by score descending
      first_scores = Enum.map(first_results, & &1.score)
      second_scores = Enum.map(second_results, & &1.score)
      assert first_scores == Enum.sort(first_scores, :desc)
      assert second_scores == Enum.sort(second_scores, :desc)
    end

    @tag :slow
    test "search_with_embeddings allows pre-computed query embeddings" do
      {:ok, encoder} = Stephen.load_encoder()
      index = Stephen.new_index(encoder)

      documents = [
        {"doc1", "Elixir is a functional programming language"},
        {"doc2", "Python is used for machine learning"}
      ]

      index = Stephen.index(encoder, index, documents)

      # Pre-compute query embeddings
      query_embeddings = Stephen.Encoder.encode_query(encoder, "functional programming")

      # Search with pre-computed embeddings
      results = Stephen.Retriever.search_with_embeddings(query_embeddings, index, top_k: 2)

      assert length(results) == 2
      assert hd(results).doc_id == "doc1"
    end
  end
end
