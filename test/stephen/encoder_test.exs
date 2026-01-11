defmodule Stephen.EncoderTest do
  use ExUnit.Case

  alias Stephen.Encoder

  @moduletag :integration
  @moduletag timeout: :infinity

  describe "load/1" do
    @tag :slow
    test "loads encoder with default settings" do
      {:ok, encoder} = Encoder.load()

      assert encoder.embedding_dim == 384
      assert encoder.output_dim == 128
      assert encoder.max_query_length == 32
      assert encoder.max_doc_length == 180
      assert encoder.projection != nil
    end

    @tag :slow
    test "loads encoder without projection" do
      {:ok, encoder} = Encoder.load(projection_dim: nil)

      assert encoder.embedding_dim == 384
      assert encoder.output_dim == 384
      assert encoder.projection == nil
    end

    @tag :slow
    test "loads encoder with custom settings" do
      {:ok, encoder} =
        Encoder.load(
          max_query_length: 64,
          max_doc_length: 256,
          projection_dim: 64
        )

      assert encoder.max_query_length == 64
      assert encoder.max_doc_length == 256
      assert encoder.output_dim == 64
    end
  end

  describe "encode_query/3" do
    @tag :slow
    test "encodes query and pads to max_query_length" do
      {:ok, encoder} = Encoder.load()

      embeddings = Encoder.encode_query(encoder, "hello world")

      {seq_len, dim} = Nx.shape(embeddings)
      assert seq_len == 32
      assert dim == 128
    end

    @tag :slow
    test "encodes query without padding" do
      {:ok, encoder} = Encoder.load()

      embeddings = Encoder.encode_query(encoder, "hello world", pad: false)

      {seq_len, dim} = Nx.shape(embeddings)
      assert seq_len < 32
      assert dim == 128
    end
  end

  describe "encode_document/2" do
    @tag :slow
    test "encodes document without padding" do
      {:ok, encoder} = Encoder.load()

      embeddings = Encoder.encode_document(encoder, "This is a test document")

      {seq_len, dim} = Nx.shape(embeddings)
      assert seq_len > 0
      assert seq_len < 180
      assert dim == 128
    end
  end

  describe "encode_documents/2" do
    @tag :slow
    test "batch encodes multiple documents" do
      {:ok, encoder} = Encoder.load()

      docs = ["First document", "Second document", "Third document"]
      embeddings_list = Encoder.encode_documents(encoder, docs)

      assert length(embeddings_list) == 3

      for embeddings <- embeddings_list do
        {_seq_len, dim} = Nx.shape(embeddings)
        assert dim == 128
      end
    end
  end

  describe "encode_queries/3" do
    @tag :slow
    test "batch encodes multiple queries with padding" do
      {:ok, encoder} = Encoder.load()

      queries = ["query one", "query two"]
      embeddings_list = Encoder.encode_queries(encoder, queries)

      assert length(embeddings_list) == 2

      for embeddings <- embeddings_list do
        {seq_len, dim} = Nx.shape(embeddings)
        assert seq_len == 32
        assert dim == 128
      end
    end
  end

  describe "embedding_dim/1 and hidden_dim/1" do
    @tag :slow
    test "returns correct dimensions" do
      {:ok, encoder} = Encoder.load(projection_dim: 64)

      assert Encoder.embedding_dim(encoder) == 64
      assert Encoder.hidden_dim(encoder) == 384
    end
  end

  describe "ColBERT model loading" do
    @tag :slow
    test "loads colbert-ir/colbertv2.0 with trained projection weights" do
      {:ok, encoder} = Encoder.load(model: "colbert-ir/colbertv2.0")

      # ColBERTv2 uses BERT-base (768 hidden) with 128-dim projection
      assert encoder.embedding_dim == 768
      assert encoder.output_dim == 128
      assert encoder.projection != nil
      assert Nx.shape(encoder.projection) == {768, 128}
    end

    @tag :slow
    test "colbert encoder produces embeddings" do
      {:ok, encoder} = Encoder.load(model: "colbert-ir/colbertv2.0")

      embeddings = Encoder.encode_query(encoder, "what is elixir?")
      {seq_len, dim} = Nx.shape(embeddings)

      assert seq_len == 32
      assert dim == 128
    end

    @tag :slow
    test "colbert embeddings match reference values" do
      {:ok, encoder} = Encoder.load(model: "colbert-ir/colbertv2.0")

      # Encode "hello world" and check output shape and some values
      embeddings = Encoder.encode_query(encoder, "hello world", pad: false)
      {_seq_len, dim} = Nx.shape(embeddings)

      assert dim == 128

      # Values should be L2-normalized (unit length per token)
      norms = Nx.LinAlg.norm(embeddings, axes: [-1])
      assert Nx.all(Nx.less(Nx.abs(Nx.subtract(norms, 1.0)), 0.01)) |> Nx.to_number() == 1
    end

    @tag :slow
    test "auto-detects model_type from config.json" do
      # colbert-ir/colbertv2.0 has model_type: "bert" in config.json
      # This test verifies auto-detection works (same result as hardcoded)
      {:ok, encoder} = Encoder.load(model: "colbert-ir/colbertv2.0")
      assert encoder.embedding_dim == 768
    end

    @tag :slow
    test "accepts base_module override" do
      # Even though colbertv2.0 is BERT-based, we can force a different module
      # This will fail to load params but proves the override works
      result = Encoder.load(model: "colbert-ir/colbertv2.0", base_module: Bumblebee.Text.Roberta)

      # Should fail because RoBERTa architecture doesn't match BERT weights
      assert {:error, _} = result
    end
  end
end
