defmodule Stephen.CompressionTest do
  use ExUnit.Case

  alias Stephen.Compression

  @moduletag :integration
  @moduletag timeout: :infinity

  describe "train/2" do
    @tag :slow
    test "trains compression codebook from embeddings" do
      # Create random normalized embeddings
      key = Nx.Random.key(42)
      {embeddings, _key} = Nx.Random.normal(key, shape: {100, 128}, type: :f32)
      embeddings = normalize(embeddings)

      compression = Compression.train(embeddings, num_centroids: 16, iterations: 5)

      assert compression.num_centroids == 16
      assert compression.embedding_dim == 128
      assert compression.residual_bits == 8
      assert Nx.shape(compression.centroids) == {16, 128}
    end

    @tag :slow
    test "trains from list of embeddings" do
      key = Nx.Random.key(42)
      {emb1, key} = Nx.Random.normal(key, shape: {50, 64}, type: :f32)
      {emb2, _key} = Nx.Random.normal(key, shape: {50, 64}, type: :f32)

      compression = Compression.train([emb1, emb2], num_centroids: 8, iterations: 3)

      assert compression.embedding_dim == 64
      assert Nx.shape(compression.centroids) == {8, 64}
    end
  end

  describe "compress/2 and decompress/2" do
    @tag :slow
    test "compresses and decompresses embeddings" do
      key = Nx.Random.key(42)
      {embeddings, _key} = Nx.Random.normal(key, shape: {50, 64}, type: :f32)
      embeddings = normalize(embeddings)

      compression = Compression.train(embeddings, num_centroids: 8, iterations: 5)
      compressed = Compression.compress(compression, embeddings)

      assert Map.has_key?(compressed, :centroid_ids)
      assert Map.has_key?(compressed, :residuals)
      assert Nx.shape(compressed.centroid_ids) == {50}
      assert Nx.shape(compressed.residuals) == {50, 64}

      # Decompress and verify shape
      decompressed = Compression.decompress(compression, compressed)
      assert Nx.shape(decompressed) == {50, 64}
    end

    @tag :slow
    test "decompressed embeddings are close to original" do
      key = Nx.Random.key(42)
      {embeddings, _key} = Nx.Random.normal(key, shape: {20, 32}, type: :f32)
      embeddings = normalize(embeddings)

      compression = Compression.train(embeddings, num_centroids: 8, iterations: 10)
      compressed = Compression.compress(compression, embeddings)
      decompressed = Compression.decompress(compression, compressed)

      # Check that reconstruction error is reasonable
      # (not exact due to quantization, but should be in same ballpark)
      diff = Nx.subtract(embeddings, decompressed)
      mse = Nx.mean(Nx.pow(diff, 2)) |> Nx.to_number()

      # MSE should be reasonably small (less than 1.0 for normalized vectors)
      assert mse < 1.0
    end
  end

  describe "approximate_similarity/3" do
    @tag :slow
    test "computes similarity using compressed representations" do
      key = Nx.Random.key(42)
      {doc_embeddings, key} = Nx.Random.normal(key, shape: {20, 64}, type: :f32)
      {query_embeddings, _key} = Nx.Random.normal(key, shape: {5, 64}, type: :f32)

      doc_embeddings = normalize(doc_embeddings)
      query_embeddings = normalize(query_embeddings)

      compression = Compression.train(doc_embeddings, num_centroids: 8, iterations: 5)
      compressed_doc = Compression.compress(compression, doc_embeddings)

      similarity =
        Compression.approximate_similarity(compression, query_embeddings, compressed_doc)

      # Should return one score per query token
      assert Nx.shape(similarity) == {5}
    end
  end

  describe "save/2 and load/1" do
    @tag :slow
    test "saves and loads compression codebook" do
      key = Nx.Random.key(42)
      {embeddings, _key} = Nx.Random.normal(key, shape: {50, 64}, type: :f32)

      compression = Compression.train(embeddings, num_centroids: 8, iterations: 3)

      path = Path.join(System.tmp_dir!(), "compression_test_#{:rand.uniform(100_000)}.bin")

      try do
        :ok = Compression.save(compression, path)
        {:ok, loaded} = Compression.load(path)

        assert loaded.num_centroids == compression.num_centroids
        assert loaded.embedding_dim == compression.embedding_dim
        assert loaded.residual_bits == compression.residual_bits
        assert Nx.shape(loaded.centroids) == Nx.shape(compression.centroids)
      after
        File.rm(path)
      end
    end
  end

  defp normalize(embeddings) do
    norm = Nx.LinAlg.norm(embeddings, axes: [-1], keep_axes: true)
    Nx.divide(embeddings, Nx.add(norm, 1.0e-9))
  end
end
