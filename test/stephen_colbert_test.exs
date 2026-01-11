defmodule StephenColbertTest do
  use ExUnit.Case

  @moduletag :integration

  describe "full pipeline" do
    @tag :slow
    test "indexes and searches documents" do
      # This test requires downloading model weights, so skip in CI
      # To run: mix test --include slow

      {:ok, encoder} = StephenColbert.load_encoder()
      index = StephenColbert.new_index(encoder)

      documents = [
        {"doc1", "The quick brown fox jumps over the lazy dog"},
        {"doc2", "Machine learning is a subset of artificial intelligence"},
        {"doc3", "Elixir is a functional programming language built on Erlang"}
      ]

      index = StephenColbert.index(encoder, index, documents)

      # Search for programming-related content
      results = StephenColbert.search(encoder, index, "programming languages", top_k: 3)

      assert length(results) == 3
      assert is_float(hd(results).score)

      # The programming document should rank high
      doc_ids = Enum.map(results, & &1.doc_id)
      assert "doc3" in doc_ids
    end

    @tag :slow
    test "reranks documents" do
      {:ok, encoder} = StephenColbert.load_encoder()
      index = StephenColbert.new_index(encoder)

      documents = [
        {"doc1", "Python is popular for data science"},
        {"doc2", "Elixir excels at concurrent programming"},
        {"doc3", "JavaScript runs in web browsers"}
      ]

      index = StephenColbert.index(encoder, index, documents)

      # Rerank specific documents
      results =
        StephenColbert.rerank(encoder, index, "concurrent systems", ["doc1", "doc2", "doc3"])

      assert length(results) == 3
      # Elixir doc should rank highest for "concurrent systems"
      assert hd(results).doc_id == "doc2"
    end
  end
end
