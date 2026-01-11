defmodule Stephen.ChunkerTest do
  use ExUnit.Case

  alias Stephen.Chunker

  describe "chunk_text/2" do
    test "returns single chunk for short text" do
      text = "This is a short sentence."
      chunks = Chunker.chunk_text(text, max_length: 10)

      assert chunks == [text]
    end

    test "splits long text into overlapping chunks" do
      # Create text with 20 words
      text = Enum.map_join(1..20, " ", fn i -> "word#{i}" end)

      chunks = Chunker.chunk_text(text, max_length: 10, stride: 5)

      # Should have 4 chunks: 0-9, 5-14, 10-19, 15-19
      assert length(chunks) == 4

      # First chunk should have first 10 words
      first_chunk = List.first(chunks)
      assert String.contains?(first_chunk, "word1")
      assert String.contains?(first_chunk, "word10")

      # Chunks should overlap
      second_chunk = Enum.at(chunks, 1)
      assert String.contains?(second_chunk, "word6")
      assert String.contains?(second_chunk, "word10")
    end

    test "uses custom tokenizer" do
      text = "hello,world,this,is,a,test"
      tokenizer = fn t -> String.split(t, ",") end

      chunks = Chunker.chunk_text(text, max_length: 3, stride: 2, tokenizer: tokenizer)

      # 6 tokens, max_length 3, stride 2 = 3 chunks (0-2, 2-4, 4-5)
      assert length(chunks) == 3
    end
  end

  describe "chunk_documents/2" do
    test "chunks multiple documents" do
      docs = [
        {"doc1", Enum.map_join(1..5, " ", fn i -> "word#{i}" end)},
        {"doc2", Enum.map_join(1..15, " ", fn i -> "term#{i}" end)}
      ]

      {chunks, mapping} = Chunker.chunk_documents(docs, max_length: 10, stride: 5)

      # doc1 has 5 words -> 1 chunk
      # doc2 has 15 words -> 3 chunks (0-9, 5-14, 10-14)
      assert length(chunks) == 4

      # Verify mapping
      assert Map.has_key?(mapping, "doc1__chunk_0")
      assert Map.has_key?(mapping, "doc2__chunk_0")
      assert Map.has_key?(mapping, "doc2__chunk_1")
      assert Map.has_key?(mapping, "doc2__chunk_2")

      assert mapping["doc1__chunk_0"].doc_id == "doc1"
      assert mapping["doc2__chunk_0"].doc_id == "doc2"
      assert mapping["doc2__chunk_1"].doc_id == "doc2"
      assert mapping["doc2__chunk_2"].doc_id == "doc2"
    end

    test "preserves chunk indices" do
      docs = [{"long_doc", Enum.map_join(1..30, " ", fn i -> "x#{i}" end)}]

      {_chunks, mapping} = Chunker.chunk_documents(docs, max_length: 10, stride: 5)

      # 30 words, max 10, stride 5 -> chunks at 0, 5, 10, 15, 20, 25
      assert mapping["long_doc__chunk_0"].chunk_index == 0
      assert mapping["long_doc__chunk_1"].chunk_index == 1
      assert mapping["long_doc__chunk_2"].chunk_index == 2
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
      assert Chunker.estimate_chunks(text, max_length: 10) == 1
    end

    test "estimates correct number of chunks" do
      # 20 words with max_length 10, stride 5
      text = Enum.map_join(1..20, " ", fn i -> "word#{i}" end)
      estimate = Chunker.estimate_chunks(text, max_length: 10, stride: 5)

      # Actual chunks: (20 - 10) / 5 + 2 = 4
      assert estimate == 4
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
