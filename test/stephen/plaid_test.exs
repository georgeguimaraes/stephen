defmodule Stephen.PlaidTest do
  use ExUnit.Case

  alias Stephen.Plaid

  @moduletag :integration
  @moduletag timeout: :infinity

  describe "new/1" do
    test "creates empty PLAID index" do
      plaid = Plaid.new(embedding_dim: 128)

      assert plaid.embedding_dim == 128
      assert plaid.num_centroids == 1024
      assert plaid.doc_count == 0
      assert plaid.centroids == nil
      assert plaid.inverted_index == %{}
      assert plaid.doc_embeddings == %{}
    end

    test "creates index with custom num_centroids" do
      plaid = Plaid.new(embedding_dim: 64, num_centroids: 256)

      assert plaid.embedding_dim == 64
      assert plaid.num_centroids == 256
    end
  end

  describe "index_documents/2" do
    @tag :slow
    test "indexes documents and trains centroids" do
      plaid = Plaid.new(embedding_dim: 32, num_centroids: 8)

      # Create fake document embeddings
      docs = create_test_documents(5, 10, 32)
      plaid = Plaid.index_documents(plaid, docs)

      assert Plaid.size(plaid) == 5
      assert plaid.centroids != nil
      assert Nx.shape(plaid.centroids) == {8, 32}
    end

    @tag :slow
    test "preserves doc_id types" do
      plaid = Plaid.new(embedding_dim: 32, num_centroids: 4)

      key = Nx.Random.key(42)
      {emb1, key} = Nx.Random.normal(key, shape: {5, 32}, type: :f32)
      {emb2, _key} = Nx.Random.normal(key, shape: {5, 32}, type: :f32)

      docs = [
        {"doc_string", normalize(emb1)},
        {123, normalize(emb2)}
      ]

      plaid = Plaid.index_documents(plaid, docs)

      assert "doc_string" in Plaid.doc_ids(plaid)
      assert 123 in Plaid.doc_ids(plaid)
    end
  end

  describe "add_document/3" do
    @tag :slow
    test "adds document to existing index" do
      plaid = Plaid.new(embedding_dim: 32, num_centroids: 4)

      # First index some documents to train centroids
      docs = create_test_documents(3, 8, 32)
      plaid = Plaid.index_documents(plaid, docs)

      # Add another document
      key = Nx.Random.key(99)
      {new_emb, _key} = Nx.Random.normal(key, shape: {6, 32}, type: :f32)
      new_emb = normalize(new_emb)

      plaid = Plaid.add_document(plaid, "new_doc", new_emb)

      assert Plaid.size(plaid) == 4
      assert "new_doc" in Plaid.doc_ids(plaid)
    end
  end

  describe "search/3" do
    @tag :slow
    test "returns ranked results" do
      plaid = Plaid.new(embedding_dim: 32, num_centroids: 4)
      docs = create_test_documents(10, 8, 32)
      plaid = Plaid.index_documents(plaid, docs)

      # Create query embeddings
      key = Nx.Random.key(123)
      {query_emb, _key} = Nx.Random.normal(key, shape: {5, 32}, type: :f32)
      query_emb = normalize(query_emb)

      results = Plaid.search(plaid, query_emb, top_k: 5)

      assert length(results) == 5
      assert Enum.all?(results, &Map.has_key?(&1, :doc_id))
      assert Enum.all?(results, &Map.has_key?(&1, :score))

      # Results should be sorted by score descending
      scores = Enum.map(results, & &1.score)
      assert scores == Enum.sort(scores, :desc)
    end

    @tag :slow
    test "respects top_k parameter" do
      plaid = Plaid.new(embedding_dim: 32, num_centroids: 4)
      docs = create_test_documents(10, 8, 32)
      plaid = Plaid.index_documents(plaid, docs)

      key = Nx.Random.key(456)
      {query_emb, _key} = Nx.Random.normal(key, shape: {5, 32}, type: :f32)
      query_emb = normalize(query_emb)

      results = Plaid.search(plaid, query_emb, top_k: 3)
      assert length(results) == 3
    end

    @tag :slow
    test "custom nprobe parameter" do
      plaid = Plaid.new(embedding_dim: 32, num_centroids: 8)
      docs = create_test_documents(20, 8, 32)
      plaid = Plaid.index_documents(plaid, docs)

      key = Nx.Random.key(789)
      {query_emb, _key} = Nx.Random.normal(key, shape: {5, 32}, type: :f32)
      query_emb = normalize(query_emb)

      # Higher nprobe should still work
      results = Plaid.search(plaid, query_emb, top_k: 5, nprobe: 4)
      assert length(results) <= 5
    end
  end

  describe "get_embeddings/2" do
    @tag :slow
    test "retrieves document embeddings" do
      plaid = Plaid.new(embedding_dim: 32, num_centroids: 4)

      key = Nx.Random.key(42)
      {emb, _key} = Nx.Random.normal(key, shape: {10, 32}, type: :f32)
      emb = normalize(emb)

      plaid = Plaid.index_documents(plaid, [{"my_doc", emb}])

      retrieved = Plaid.get_embeddings(plaid, "my_doc")
      assert Nx.shape(retrieved) == {10, 32}
    end

    test "returns nil for unknown doc" do
      plaid = Plaid.new(embedding_dim: 32)
      assert Plaid.get_embeddings(plaid, "unknown") == nil
    end
  end

  # Helper functions

  defp create_test_documents(num_docs, seq_len, dim) do
    key = Nx.Random.key(42)

    Enum.map_reduce(1..num_docs, key, fn i, key ->
      {emb, new_key} = Nx.Random.normal(key, shape: {seq_len, dim}, type: :f32)
      emb = normalize(emb)
      {{"doc_#{i}", emb}, new_key}
    end)
    |> elem(0)
  end

  defp normalize(embeddings) do
    norm = Nx.LinAlg.norm(embeddings, axes: [-1], keep_axes: true)
    Nx.divide(embeddings, Nx.add(norm, 1.0e-9))
  end
end
