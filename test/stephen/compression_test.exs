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

  describe "1-bit compression" do
    @tag :slow
    test "trains and compresses with 1-bit residuals" do
      key = Nx.Random.key(42)
      {embeddings, _key} = Nx.Random.normal(key, shape: {50, 128}, type: :f32)
      embeddings = normalize(embeddings)

      compression =
        Compression.train(embeddings, num_centroids: 8, residual_bits: 1, iterations: 5)

      assert compression.residual_bits == 1
      assert compression.embedding_dim == 128

      compressed = Compression.compress(compression, embeddings)

      # 1-bit: 128 dims packed into 16 bytes
      assert Nx.shape(compressed.residuals) == {50, 16}
    end

    @tag :slow
    test "1-bit compression achieves expected compression ratio" do
      # 128-dim float32 = 512 bytes original
      # 1-bit: 2 bytes (centroid) + 16 bytes (packed) = 18 bytes
      # Compression ratio ~ 28x
      ratio = Compression.compression_ratio(128, 1)
      assert ratio > 25.0 and ratio < 30.0
    end

    @tag :slow
    test "1-bit round-trip preserves approximate values" do
      key = Nx.Random.key(42)
      {embeddings, _key} = Nx.Random.normal(key, shape: {20, 64}, type: :f32)
      embeddings = normalize(embeddings)

      compression =
        Compression.train(embeddings, num_centroids: 8, residual_bits: 1, iterations: 10)

      compressed = Compression.compress(compression, embeddings)
      decompressed = Compression.decompress(compression, compressed)

      assert Nx.shape(decompressed) == {20, 64}

      # 1-bit has lower quality, so we expect higher reconstruction error
      diff = Nx.subtract(embeddings, decompressed)
      mse = Nx.mean(Nx.pow(diff, 2)) |> Nx.to_number()

      # MSE will be higher than 8-bit but still reasonable
      assert mse < 2.0
    end
  end

  describe "2-bit compression" do
    @tag :slow
    test "trains and compresses with 2-bit residuals" do
      key = Nx.Random.key(42)
      {embeddings, _key} = Nx.Random.normal(key, shape: {50, 128}, type: :f32)
      embeddings = normalize(embeddings)

      compression =
        Compression.train(embeddings, num_centroids: 8, residual_bits: 2, iterations: 5)

      assert compression.residual_bits == 2
      assert compression.embedding_dim == 128

      compressed = Compression.compress(compression, embeddings)

      # 2-bit: 128 dims packed into 32 bytes (4 values per byte)
      assert Nx.shape(compressed.residuals) == {50, 32}
    end

    @tag :slow
    test "2-bit compression achieves expected compression ratio" do
      # 128-dim float32 = 512 bytes original
      # 2-bit: 2 bytes (centroid) + 32 bytes (packed) = 34 bytes
      # Compression ratio ~ 15x
      ratio = Compression.compression_ratio(128, 2)
      assert ratio > 13.0 and ratio < 17.0
    end

    @tag :slow
    test "2-bit round-trip preserves approximate values" do
      key = Nx.Random.key(42)
      {embeddings, _key} = Nx.Random.normal(key, shape: {20, 64}, type: :f32)
      embeddings = normalize(embeddings)

      compression =
        Compression.train(embeddings, num_centroids: 8, residual_bits: 2, iterations: 10)

      compressed = Compression.compress(compression, embeddings)
      decompressed = Compression.decompress(compression, compressed)

      assert Nx.shape(decompressed) == {20, 64}

      # 2-bit should have lower error than 1-bit
      diff = Nx.subtract(embeddings, decompressed)
      mse = Nx.mean(Nx.pow(diff, 2)) |> Nx.to_number()

      # MSE should be lower than 1-bit
      assert mse < 1.5
    end
  end

  describe "compression ratios" do
    test "compression_ratio/2 returns correct values" do
      # 8-bit: 128*4 / (2 + 128) = 512 / 130 = 3.94
      assert Compression.compression_ratio(128, 8) == 3.94

      # 4-bit: 512 / (2 + 64) = 512 / 66 = 7.76
      assert Compression.compression_ratio(128, 4) == 7.76

      # 2-bit: 512 / (2 + 32) = 512 / 34 = 15.06
      assert Compression.compression_ratio(128, 2) == 15.06

      # 1-bit: 512 / (2 + 16) = 512 / 18 = 28.44
      assert Compression.compression_ratio(128, 1) == 28.44
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
