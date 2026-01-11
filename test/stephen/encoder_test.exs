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
end
