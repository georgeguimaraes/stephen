defmodule Stephen.Index.CompressedTest do
  use ExUnit.Case

  alias Stephen.Index.Compressed

  @moduletag :integration
  @moduletag timeout: :infinity

  describe "new/1" do
    test "creates empty compressed index" do
      index = Compressed.new(embedding_dim: 128)

      assert index.embedding_dim == 128
      assert index.num_centroids == 1024
      assert index.doc_count == 0
      assert index.trained? == false
      assert index.compression == nil
      assert index.centroids == nil
    end

    test "creates index with custom num_centroids" do
      index = Compressed.new(embedding_dim: 64, num_centroids: 256)

      assert index.embedding_dim == 64
      assert index.num_centroids == 256
    end
  end

  describe "train/3" do
    @tag :slow
    test "trains compression codebook and PLAID centroids" do
      index = Compressed.new(embedding_dim: 32, num_centroids: 8)

      # Create training data
      embeddings = create_test_embeddings(100, 32)

      trained_index = Compressed.train(index, embeddings, compression_centroids: 16)

      assert trained_index.trained? == true
      assert trained_index.compression != nil
      assert trained_index.centroids != nil
      assert Nx.shape(trained_index.centroids) == {8, 32}
    end
  end

  describe "add/3" do
    @tag :slow
    test "adds document to trained index" do
      index = Compressed.new(embedding_dim: 32, num_centroids: 4)

      # Train on some data
      training_data = create_test_embeddings(50, 32)
      index = Compressed.train(index, training_data, compression_centroids: 8)

      # Add a document
      doc_embeddings = create_test_embeddings(10, 32)
      index = Compressed.add(index, "doc1", doc_embeddings)

      assert Compressed.size(index) == 1
      assert "doc1" in Compressed.doc_ids(index)
    end

    test "raises error when index not trained" do
      index = Compressed.new(embedding_dim: 32)
      embeddings = create_test_embeddings(10, 32)

      assert_raise ArgumentError, ~r/must be trained/, fn ->
        Compressed.add(index, "doc1", embeddings)
      end
    end
  end

  describe "index_documents/2" do
    @tag :slow
    test "trains and indexes documents in one call" do
      index = Compressed.new(embedding_dim: 32, num_centroids: 4)

      docs = create_test_documents(5, 10, 32)
      index = Compressed.index_documents(index, docs)

      assert Compressed.size(index) == 5
      assert index.trained? == true
    end
  end

  describe "search/3" do
    @tag :slow
    test "returns ranked results" do
      index = Compressed.new(embedding_dim: 32, num_centroids: 4)
      docs = create_test_documents(10, 8, 32)
      index = Compressed.index_documents(index, docs)

      # Create query embeddings
      query_emb = create_test_embeddings(5, 32)

      results = Compressed.search(index, query_emb, top_k: 5)

      assert length(results) == 5
      assert Enum.all?(results, &Map.has_key?(&1, :doc_id))
      assert Enum.all?(results, &Map.has_key?(&1, :score))

      # Results should be sorted by score descending
      scores = Enum.map(results, & &1.score)
      assert scores == Enum.sort(scores, :desc)
    end

    @tag :slow
    test "respects top_k parameter" do
      index = Compressed.new(embedding_dim: 32, num_centroids: 4)
      docs = create_test_documents(10, 8, 32)
      index = Compressed.index_documents(index, docs)

      query_emb = create_test_embeddings(5, 32)
      results = Compressed.search(index, query_emb, top_k: 3)

      assert length(results) == 3
    end
  end

  describe "get_embeddings/2" do
    @tag :slow
    test "retrieves decompressed embeddings" do
      index = Compressed.new(embedding_dim: 32, num_centroids: 4)

      original_emb = create_test_embeddings(10, 32)
      docs = [{"my_doc", original_emb}]
      index = Compressed.index_documents(index, docs)

      retrieved = Compressed.get_embeddings(index, "my_doc")

      # Shape should match
      assert Nx.shape(retrieved) == {10, 32}

      # Due to compression, values won't be exact but should be similar
      # Check that cosine similarity is reasonable
      sim = cosine_similarity(original_emb[0], retrieved[0])
      assert sim > 0.5
    end

    test "returns nil for unknown doc" do
      index = Compressed.new(embedding_dim: 32)
      assert Compressed.get_embeddings(index, "unknown") == nil
    end
  end

  describe "get_compressed/2" do
    @tag :slow
    test "retrieves compressed representation" do
      index = Compressed.new(embedding_dim: 32, num_centroids: 4)
      docs = create_test_documents(1, 10, 32)
      index = Compressed.index_documents(index, docs)

      compressed = Compressed.get_compressed(index, "doc_1")

      assert Map.has_key?(compressed, :centroid_ids)
      assert Map.has_key?(compressed, :residuals)
      assert Nx.shape(compressed.centroid_ids) == {10}
    end
  end

  describe "save/2 and load/1" do
    @tag :slow
    test "saves and loads compressed index" do
      index = Compressed.new(embedding_dim: 32, num_centroids: 4)
      docs = create_test_documents(5, 8, 32)
      index = Compressed.index_documents(index, docs)

      path = Path.join(System.tmp_dir!(), "compressed_test_#{:rand.uniform(100_000)}.bin")

      try do
        :ok = Compressed.save(index, path)
        {:ok, loaded} = Compressed.load(path)

        assert loaded.num_centroids == index.num_centroids
        assert loaded.embedding_dim == index.embedding_dim
        assert loaded.doc_count == index.doc_count
        assert loaded.trained? == index.trained?
        assert Compressed.doc_ids(loaded) == Compressed.doc_ids(index)

        # Verify search still works after reload
        query_emb = create_test_embeddings(5, 32)

        original_results = Compressed.search(index, query_emb, top_k: 3)
        loaded_results = Compressed.search(loaded, query_emb, top_k: 3)

        assert length(original_results) == length(loaded_results)

        for {orig, loaded} <- Enum.zip(original_results, loaded_results) do
          assert orig.doc_id == loaded.doc_id
          assert_in_delta orig.score, loaded.score, 0.0001
        end
      after
        File.rm(path)
      end
    end
  end

  describe "stats/1" do
    @tag :slow
    test "returns compression statistics" do
      index = Compressed.new(embedding_dim: 32, num_centroids: 4)
      docs = create_test_documents(5, 10, 32)
      index = Compressed.index_documents(index, docs)

      stats = Compressed.stats(index)

      assert stats.doc_count == 5
      assert stats.embedding_dim == 32
      assert stats.trained == true
      assert stats.compressed_size_bytes > 0
      assert stats.compression_ratio > 0
    end
  end

  # Helper functions

  defp create_test_embeddings(num_tokens, dim) do
    key = Nx.Random.key(:rand.uniform(100_000))
    {emb, _key} = Nx.Random.normal(key, shape: {num_tokens, dim}, type: :f32)
    normalize(emb)
  end

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

  defp cosine_similarity(a, b) do
    dot = Nx.sum(Nx.multiply(a, b)) |> Nx.to_number()
    norm_a = Nx.LinAlg.norm(a) |> Nx.to_number()
    norm_b = Nx.LinAlg.norm(b) |> Nx.to_number()
    dot / (norm_a * norm_b + 1.0e-9)
  end
end
