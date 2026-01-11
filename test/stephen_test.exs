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
        {"colbert", "Stephen Colbert hosted The Colbert Report before The Late Show"},
        {"conan", "Conan O'Brien is known for his self-deprecating humor and remotes"},
        {"seth", "Seth Meyers was head writer at SNL before hosting Late Night"}
      ]

      index = Stephen.index(encoder, index, documents)

      # Search for late night related content
      results = Stephen.search(encoder, index, "late night comedy", top_k: 3)

      assert length(results) == 3
      assert is_float(hd(results).score)

      # The late night documents should be found
      doc_ids = Enum.map(results, & &1.doc_id)
      assert "colbert" in doc_ids
    end

    @tag :slow
    test "reranks documents" do
      {:ok, encoder} = Stephen.load_encoder()
      index = Stephen.new_index(encoder)

      documents = [
        {"colbert", "Stephen Colbert does satirical political comedy"},
        {"conan", "Conan O'Brien is known for absurdist comedy sketches"},
        {"letterman", "David Letterman pioneered the modern late night format"}
      ]

      index = Stephen.index(encoder, index, documents)

      # Rerank specific documents
      results =
        Stephen.rerank(encoder, index, "political satire", ["colbert", "conan", "letterman"])

      assert length(results) == 3
      # Colbert should rank highest for "political satire"
      assert hd(results).doc_id == "colbert"
    end

    @tag :slow
    test "batch search with multiple queries" do
      {:ok, encoder} = Stephen.load_encoder()
      index = Stephen.new_index(encoder)

      documents = [
        {"colbert", "Stephen Colbert does satirical political commentary on The Late Show"},
        {"conan", "Conan O'Brien traveled the world for Conan Without Borders specials"},
        {"seth", "Seth Meyers hosts Late Night and does A Closer Look segments"},
        {"oliver", "John Oliver hosts Last Week Tonight with in-depth investigative comedy"}
      ]

      index = Stephen.index(encoder, index, documents)

      # Batch search multiple queries
      queries = ["political satire", "travel comedy", "investigative journalism"]
      results = Stephen.Retriever.batch_search(encoder, index, queries, top_k: 2)

      assert length(results) == 3
      assert length(Enum.at(results, 0)) == 2
      assert length(Enum.at(results, 1)) == 2
      assert length(Enum.at(results, 2)) == 2

      # First query about political satire should rank colbert high
      satire_results = Enum.at(results, 0)
      satire_doc_ids = Enum.map(satire_results, & &1.doc_id)
      assert "colbert" in satire_doc_ids

      # Second query about travel should rank conan high
      travel_results = Enum.at(results, 1)
      travel_doc_ids = Enum.map(travel_results, & &1.doc_id)
      assert "conan" in travel_doc_ids

      # Third query about investigative should rank oliver high
      investigative_results = Enum.at(results, 2)
      investigative_doc_ids = Enum.map(investigative_results, & &1.doc_id)
      assert "oliver" in investigative_doc_ids
    end

    @tag :slow
    test "batch rerank with multiple query-candidate pairs" do
      {:ok, encoder} = Stephen.load_encoder()
      index = Stephen.new_index(encoder)

      documents = [
        {"colbert", "Stephen Colbert does satirical political commentary nightly"},
        {"conan", "Conan O'Brien is known for absurdist comedy and travel shows"},
        {"seth", "Seth Meyers does political analysis on Late Night"},
        {"oliver", "John Oliver does investigative comedy journalism"}
      ]

      index = Stephen.index(encoder, index, documents)

      # Batch rerank with different candidates for each query
      queries_and_candidates = [
        {"satirical political commentary", ["colbert", "conan", "seth"]},
        {"absurdist comedy travel shows", ["conan", "seth", "oliver"]}
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
        {"colbert", "Stephen Colbert is known for satirical political comedy"},
        {"conan", "Conan O'Brien does absurdist comedy sketches and remotes"}
      ]

      index = Stephen.index(encoder, index, documents)

      # Pre-compute query embeddings
      query_embeddings = Stephen.Encoder.encode_query(encoder, "political satire")

      # Search with pre-computed embeddings
      results = Stephen.Retriever.search_with_embeddings(query_embeddings, index, top_k: 2)

      assert length(results) == 2
      assert hd(results).doc_id == "colbert"
    end
  end
end
