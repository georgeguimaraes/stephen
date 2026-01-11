defmodule Stephen.Compression do
  @moduledoc """
  ColBERTv2-style residual compression for token embeddings.

  Compresses embeddings using centroid-based representation:
  1. Learn K centroids using K-means clustering
  2. For each embedding, store centroid ID + quantized residual
  3. Achieves ~6x compression while maintaining retrieval quality

  ## How it works

  Instead of storing full float32 embeddings, we store:
  - Centroid ID (2 bytes for 65536 centroids)
  - Quantized residual (1 byte per dimension)

  To reconstruct: embedding ≈ centroid[id] + dequantize(residual)
  """

  import Nx.Defn

  defstruct [:centroids, :num_centroids, :embedding_dim, :residual_bits]

  @type t :: %__MODULE__{
          centroids: Nx.Tensor.t(),
          num_centroids: pos_integer(),
          embedding_dim: pos_integer(),
          residual_bits: pos_integer()
        }

  @type compressed_embedding :: %{
          centroid_ids: Nx.Tensor.t(),
          residuals: Nx.Tensor.t()
        }

  @default_num_centroids 2048
  @default_residual_bits 8
  @default_kmeans_iterations 20

  @doc """
  Trains a compression codebook from a collection of embeddings.

  ## Arguments
    * `embeddings` - List of embedding tensors or single tensor of shape {n, dim}
    * `opts` - Options

  ## Options
    * `:num_centroids` - Number of centroids (default: #{@default_num_centroids})
    * `:residual_bits` - Bits for residual quantization (default: #{@default_residual_bits})
    * `:iterations` - K-means iterations (default: #{@default_kmeans_iterations})

  ## Returns
    A trained compression codebook struct.
  """
  @spec train([Nx.Tensor.t()] | Nx.Tensor.t(), keyword()) :: t()
  def train(embeddings, opts \\ []) do
    num_centroids = Keyword.get(opts, :num_centroids, @default_num_centroids)
    residual_bits = Keyword.get(opts, :residual_bits, @default_residual_bits)
    iterations = Keyword.get(opts, :iterations, @default_kmeans_iterations)

    # Flatten list of embeddings into single tensor
    all_embeddings =
      case embeddings do
        [_ | _] -> Nx.concatenate(embeddings, axis: 0)
        tensor -> tensor
      end

    {_n, embedding_dim} = Nx.shape(all_embeddings)

    # Train centroids using K-means
    centroids = kmeans(all_embeddings, num_centroids, iterations)

    %__MODULE__{
      centroids: centroids,
      num_centroids: num_centroids,
      embedding_dim: embedding_dim,
      residual_bits: residual_bits
    }
  end

  @doc """
  Compresses embeddings using the trained codebook.

  ## Arguments
    * `compression` - Trained compression codebook
    * `embeddings` - Tensor of shape {n, dim} to compress

  ## Returns
    Compressed embedding struct with centroid IDs and quantized residuals.
  """
  @spec compress(t(), Nx.Tensor.t()) :: compressed_embedding()
  def compress(compression, embeddings) do
    %{centroids: centroids, residual_bits: residual_bits} = compression

    # Find nearest centroid for each embedding
    centroid_ids = find_nearest_centroids(embeddings, centroids)

    # Compute residuals
    nearest_centroids = Nx.take(centroids, centroid_ids, axis: 0)
    residuals = Nx.subtract(embeddings, nearest_centroids)

    # Quantize residuals
    quantized_residuals = quantize_residuals(residuals, residual_bits)

    %{
      centroid_ids: centroid_ids,
      residuals: quantized_residuals
    }
  end

  @doc """
  Decompresses embeddings from compressed representation.

  ## Arguments
    * `compression` - Trained compression codebook
    * `compressed` - Compressed embedding struct

  ## Returns
    Reconstructed embeddings tensor of shape {n, dim}.
  """
  @spec decompress(t(), compressed_embedding()) :: Nx.Tensor.t()
  def decompress(compression, compressed) do
    %{centroids: centroids, residual_bits: residual_bits} = compression
    %{centroid_ids: centroid_ids, residuals: quantized_residuals} = compressed

    # Get centroids
    nearest_centroids = Nx.take(centroids, centroid_ids, axis: 0)

    # Dequantize residuals
    residuals = dequantize_residuals(quantized_residuals, residual_bits)

    # Reconstruct
    Nx.add(nearest_centroids, residuals)
  end

  @doc """
  Computes approximate similarity using compressed representations.

  Uses centroid lookup + residual correction for efficient scoring.
  """
  @spec approximate_similarity(t(), Nx.Tensor.t(), compressed_embedding()) :: Nx.Tensor.t()
  def approximate_similarity(compression, query_embeddings, compressed_doc) do
    # Decompress and compute similarity
    doc_embeddings = decompress(compression, compressed_doc)
    similarity_matrix = Nx.dot(query_embeddings, Nx.transpose(doc_embeddings))
    Nx.reduce_max(similarity_matrix, axes: [1])
  end

  @doc """
  Saves compression codebook to disk.
  """
  @spec save(t(), Path.t()) :: :ok
  def save(compression, path) do
    data = %{
      centroids:
        {Nx.shape(compression.centroids), Nx.type(compression.centroids),
         Nx.to_binary(compression.centroids)},
      num_centroids: compression.num_centroids,
      embedding_dim: compression.embedding_dim,
      residual_bits: compression.residual_bits
    }

    File.write!(path, :erlang.term_to_binary(data))
    :ok
  end

  @doc """
  Loads compression codebook from disk.
  """
  @spec load(Path.t()) :: {:ok, t()} | {:error, term()}
  def load(path) do
    with {:ok, binary} <- File.read(path) do
      data = :erlang.binary_to_term(binary)
      {shape, type, centroid_binary} = data.centroids

      centroids =
        centroid_binary
        |> Nx.from_binary(type)
        |> Nx.reshape(shape)

      {:ok,
       %__MODULE__{
         centroids: centroids,
         num_centroids: data.num_centroids,
         embedding_dim: data.embedding_dim,
         residual_bits: data.residual_bits
       }}
    end
  end

  # K-means clustering implementation
  defp kmeans(embeddings, k, iterations) do
    {n, dim} = Nx.shape(embeddings)

    # Initialize centroids by random selection from embeddings
    key = Nx.Random.key(42)
    {indices, _key} = Nx.Random.randint(key, 0, n, shape: {k})
    initial_centroids = Nx.take(embeddings, indices, axis: 0)

    # Run K-means iterations
    Enum.reduce(1..iterations, initial_centroids, fn _i, centroids ->
      # Assign each embedding to nearest centroid
      assignments = find_nearest_centroids(embeddings, centroids)

      # Update centroids as mean of assigned embeddings
      update_centroids(embeddings, assignments, k, dim)
    end)
  end

  # Find nearest centroid for each embedding
  defnp find_nearest_centroids(embeddings, centroids) do
    # embeddings: {n, dim}, centroids: {k, dim}
    # Compute pairwise distances
    # Using squared L2 distance: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    emb_sq = Nx.sum(Nx.pow(embeddings, 2), axes: [1], keep_axes: true)
    cent_sq = Nx.sum(Nx.pow(centroids, 2), axes: [1])
    dot_product = Nx.dot(embeddings, Nx.transpose(centroids))

    distances = emb_sq + cent_sq - 2 * dot_product

    # Return index of minimum distance for each embedding
    Nx.argmin(distances, axis: 1)
  end

  # Update centroids as mean of assigned points
  defp update_centroids(embeddings, assignments, k, _dim) do
    # This is a sequential operation for correctness
    # Could be optimized with scatter operations
    centroids =
      for i <- 0..(k - 1) do
        mask = Nx.equal(assignments, i)
        count = Nx.sum(mask) |> Nx.to_number()

        if count > 0 do
          # Expand mask to match embedding dimensions
          mask_expanded = Nx.reshape(mask, {:auto, 1})
          masked = Nx.multiply(embeddings, mask_expanded)
          sum = Nx.sum(masked, axes: [0])
          Nx.divide(sum, count)
        else
          # Keep centroid unchanged if no points assigned
          # Use a random embedding as fallback
          embeddings[0]
        end
      end

    Nx.stack(centroids)
  end

  # Quantize residuals to specified bit depth
  defp quantize_residuals(residuals, bits) do
    # Scale residuals to [0, 2^bits - 1] range
    max_val = Nx.reduce_max(Nx.abs(residuals))
    # Avoid division by zero
    max_val = Nx.max(max_val, 1.0e-9)

    # Scale to [-1, 1] then to [0, 2^bits - 1]
    levels = :math.pow(2, bits) - 1
    scaled = Nx.divide(residuals, max_val)
    quantized = Nx.round(Nx.multiply(Nx.add(scaled, 1), levels / 2))

    # Store as uint8 (assuming bits <= 8)
    Nx.as_type(quantized, :u8)
  end

  # Dequantize residuals back to float
  defp dequantize_residuals(quantized, bits) do
    levels = :math.pow(2, bits) - 1
    scaled = Nx.divide(Nx.as_type(quantized, :f32), levels / 2)
    Nx.subtract(scaled, 1)
  end
end
