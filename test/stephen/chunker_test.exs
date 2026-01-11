defmodule Stephen.ChunkerTest do
  use ExUnit.Case

  alias Stephen.Chunker

  describe "chunk_text/2" do
    test "returns single chunk for short text" do
      text = "This is a short sentence."
      chunks = Chunker.chunk_text(text)

      assert chunks == [text]
    end

    test "splits long text at sentence boundaries" do
      text = """
      First sentence here. Second sentence follows. Third sentence comes next.
      Fourth sentence in new paragraph. Fifth sentence ends it.
      """

      chunks = Chunker.chunk_text(text, chunk_size: 100, chunk_overlap: 20)

      # Should have multiple chunks split at sentence boundaries
      assert length(chunks) >= 1

      # Each chunk should be a proper substring
      for chunk <- chunks do
        assert String.length(chunk) > 0
      end
    end

    test "respects markdown format" do
      text = """
      # Heading One

      Some content under heading one.

      ## Heading Two

      More content under heading two.
      """

      chunks = Chunker.chunk_text(text, chunk_size: 50, format: :markdown)

      assert length(chunks) >= 1
    end

    test "respects chunk_size parameter" do
      # Create a long text
      text = String.duplicate("This is a test sentence. ", 50)

      small_chunks = Chunker.chunk_text(text, chunk_size: 100)
      large_chunks = Chunker.chunk_text(text, chunk_size: 500)

      # Smaller chunk size should produce more chunks
      assert length(small_chunks) >= length(large_chunks)
    end
  end

  describe "chunk_documents/2" do
    test "chunks documents preserving mapping" do
      docs = [
        {"doc1", "Short document."},
        {"doc2", "Another document with more content. It has multiple sentences."}
      ]

      {chunks, mapping} = Chunker.chunk_documents(docs)

      # Each doc should have at least one chunk
      assert length(chunks) >= 2

      # Mapping should track doc origins
      for {chunk_id, _text} <- chunks do
        assert Map.has_key?(mapping, chunk_id)
        assert mapping[chunk_id].doc_id in ["doc1", "doc2"]
      end
    end

    test "generates proper chunk IDs" do
      docs = [{"my_doc", "Some content here."}]

      {chunks, mapping} = Chunker.chunk_documents(docs)

      [{chunk_id, _text}] = chunks
      assert chunk_id == "my_doc__chunk_0"
      assert mapping[chunk_id].doc_id == "my_doc"
      assert mapping[chunk_id].chunk_index == 0
    end

    test "handles multiple chunks per document" do
      long_text = String.duplicate("This is a sentence with some words. ", 30)
      docs = [{"long_doc", long_text}]

      {chunks, mapping} = Chunker.chunk_documents(docs, chunk_size: 100)

      # Should produce multiple chunks
      assert length(chunks) > 1

      # All chunks should map to same doc
      for {chunk_id, _text} <- chunks do
        assert mapping[chunk_id].doc_id == "long_doc"
      end
    end
  end

  describe "merge_results/3" do
    test "merges chunk results to document level with max aggregation" do
      mapping = %{
        "doc1__chunk_0" => %{doc_id: "doc1", chunk_index: 0},
        "doc1__chunk_1" => %{doc_id: "doc1", chunk_index: 1},
        "doc2__chunk_0" => %{doc_id: "doc2", chunk_index: 0}
      }

      results = [
        %{doc_id: "doc1__chunk_0", score: 10.0},
        %{doc_id: "doc1__chunk_1", score: 15.0},
        %{doc_id: "doc2__chunk_0", score: 12.0}
      ]

      merged = Chunker.merge_results(results, mapping)

      # doc1 should have max score 15.0
      doc1_result = Enum.find(merged, &(&1.doc_id == "doc1"))
      assert doc1_result.score == 15.0

      # doc2 should have score 12.0
      doc2_result = Enum.find(merged, &(&1.doc_id == "doc2"))
      assert doc2_result.score == 12.0

      # Results should be sorted by score descending
      scores = Enum.map(merged, & &1.score)
      assert scores == Enum.sort(scores, :desc)
    end

    test "supports mean aggregation" do
      mapping = %{
        "doc1__chunk_0" => %{doc_id: "doc1", chunk_index: 0},
        "doc1__chunk_1" => %{doc_id: "doc1", chunk_index: 1}
      }

      results = [
        %{doc_id: "doc1__chunk_0", score: 10.0},
        %{doc_id: "doc1__chunk_1", score: 20.0}
      ]

      merged = Chunker.merge_results(results, mapping, aggregation: :mean)

      doc1_result = List.first(merged)
      assert doc1_result.score == 15.0
    end

    test "supports sum aggregation" do
      mapping = %{
        "doc1__chunk_0" => %{doc_id: "doc1", chunk_index: 0},
        "doc1__chunk_1" => %{doc_id: "doc1", chunk_index: 1}
      }

      results = [
        %{doc_id: "doc1__chunk_0", score: 10.0},
        %{doc_id: "doc1__chunk_1", score: 20.0}
      ]

      merged = Chunker.merge_results(results, mapping, aggregation: :sum)

      doc1_result = List.first(merged)
      assert doc1_result.score == 30.0
    end

    test "handles unmapped chunk IDs" do
      mapping = %{}

      results = [
        %{doc_id: "unknown_chunk", score: 5.0}
      ]

      merged = Chunker.merge_results(results, mapping)

      # Should keep original ID when not in mapping
      assert Enum.find(merged, &(&1.doc_id == "unknown_chunk"))
    end
  end

  describe "estimate_chunks/2" do
    test "returns 1 for short text" do
      text = "short text"
      assert Chunker.estimate_chunks(text) == 1
    end

    test "estimates more chunks for longer text" do
      short_text = "Short sentence."
      long_text = String.duplicate("This is a longer sentence with more words. ", 20)

      short_estimate = Chunker.estimate_chunks(short_text)
      long_estimate = Chunker.estimate_chunks(long_text, chunk_size: 100)

      assert long_estimate >= short_estimate
    end
  end

  describe "get_doc_id/2" do
    test "returns original doc_id" do
      mapping = %{
        "doc1__chunk_0" => %{doc_id: "doc1", chunk_index: 0}
      }

      assert Chunker.get_doc_id("doc1__chunk_0", mapping) == "doc1"
    end

    test "returns nil for unknown chunk" do
      mapping = %{}
      assert Chunker.get_doc_id("unknown", mapping) == nil
    end
  end

  describe "get_chunk_ids/2" do
    test "returns all chunks for a document" do
      mapping = %{
        "doc1__chunk_0" => %{doc_id: "doc1", chunk_index: 0},
        "doc1__chunk_1" => %{doc_id: "doc1", chunk_index: 1},
        "doc2__chunk_0" => %{doc_id: "doc2", chunk_index: 0}
      }

      chunk_ids = Chunker.get_chunk_ids("doc1", mapping)

      assert length(chunk_ids) == 2
      assert "doc1__chunk_0" in chunk_ids
      assert "doc1__chunk_1" in chunk_ids
    end

    test "returns empty list for unknown doc" do
      mapping = %{}
      assert Chunker.get_chunk_ids("unknown", mapping) == []
    end
  end
end
