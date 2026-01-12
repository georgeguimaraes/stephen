defmodule Stephen.Compression do
  @moduledoc """
  ColBERTv2-style residual compression for token embeddings.

  Compresses embeddings using centroid-based representation:
  1. Learn K centroids using K-means clustering
  2. For each embedding, store centroid ID + quantized residual
  3. Achieves compression while maintaining retrieval quality

  ## Compression Levels

  Supports multiple quantization bit depths via `:residual_bits`:

  - `residual_bits: 8` (default) - 8-bit quantization, ~4-6x compression
  - `residual_bits: 2` - 2-bit quantization, ~16x compression
  - `residual_bits: 1` - Binary/1-bit quantization, ~32x compression

  Lower bit depths trade retrieval quality for smaller index size.

  ## Storage Format

  For 128-dim embeddings:
  - 8-bit: 2 bytes (centroid) + 128 bytes (residuals) = 130 bytes
  - 2-bit: 2 bytes (centroid) + 32 bytes (packed) = 34 bytes
  - 1-bit: 2 bytes (centroid) + 16 bytes (packed) = 18 bytes

  ## How it works

  Instead of storing full float32 embeddings (512 bytes), we store:
  - Centroid ID (2 bytes for 65536 centroids)
  - Quantized residual (packed bits)

  To reconstruct: embedding ≈ centroid[id] + dequantize(residual)
  """

  import Nx.Defn

  alias Stephen.KMeans

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

    # Train centroids using K-means (L2 distance, no normalization)
    centroids =
      KMeans.train(all_embeddings, num_centroids,
        iterations: iterations,
        distance: :l2,
        normalize: false
      )

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

  # Find nearest centroid for each embedding using L2 distance
  defnp find_nearest_centroids(embeddings, centroids) do
    # embeddings: {n, dim}, centroids: {k, dim}
    # Using squared L2 distance: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    emb_sq = Nx.sum(Nx.pow(embeddings, 2), axes: [1], keep_axes: true)
    cent_sq = Nx.sum(Nx.pow(centroids, 2), axes: [1])
    dot_product = Nx.dot(embeddings, Nx.transpose(centroids))

    distances = emb_sq + cent_sq - 2 * dot_product
    Nx.argmin(distances, axis: 1)
  end

  # Quantize residuals to specified bit depth
  # Like Python ColBERT, uses fixed scale - centroids capture magnitude
  defp quantize_residuals(residuals, 1) do
    # 1-bit: sign only, pack 8 bits per byte
    # positive -> 1, negative/zero -> 0
    bits = Nx.greater(residuals, 0) |> Nx.as_type(:u8)
    pack_bits(bits, 1)
  end

  defp quantize_residuals(residuals, 2) do
    # 2-bit: 4 levels, pack 4 values per byte
    # Map residuals to [0, 3] range using fixed scale
    clamped = Nx.clip(residuals, -1.0, 1.0)
    # Map [-1, 1] to [0, 3]
    quantized = Nx.round(Nx.multiply(Nx.add(clamped, 1.0), 1.5))
    quantized = Nx.clip(quantized, 0, 3) |> Nx.as_type(:u8)
    pack_bits(quantized, 2)
  end

  defp quantize_residuals(residuals, bits) when bits in [4, 8] do
    # 4-bit or 8-bit: use fixed scale like Python ColBERT
    # Residuals are small (difference from centroid), so fixed range works
    clamped = Nx.clip(residuals, -1.0, 1.0)
    levels = :math.pow(2, bits) - 1
    # Map [-1, 1] to [0, levels]
    quantized = Nx.round(Nx.multiply(Nx.add(clamped, 1), levels / 2))

    if bits == 4 do
      pack_bits(Nx.as_type(quantized, :u8), 4)
    else
      Nx.as_type(quantized, :u8)
    end
  end

  # Dequantize residuals back to float using fixed scale (like Python ColBERT)
  defp dequantize_residuals(quantized, 1) do
    # 1-bit: unpack and map 0->-1, 1->+1
    bits = unpack_bits(quantized, 1)
    # Map {0, 1} to {-1, +1}
    Nx.subtract(Nx.multiply(bits, 2.0), 1.0)
  end

  defp dequantize_residuals(quantized, 2) do
    # 2-bit: unpack and map [0,3] to [-1, 1]
    values = unpack_bits(quantized, 2)
    # Map [0, 3] to [-1, 1]
    Nx.subtract(Nx.divide(values, 1.5), 1.0)
  end

  defp dequantize_residuals(quantized, bits) when bits in [4, 8] do
    values =
      if bits == 4 do
        unpack_bits(quantized, 4)
      else
        Nx.as_type(quantized, :f32)
      end

    levels = :math.pow(2, bits) - 1
    # Map [0, levels] to [-1, 1]
    Nx.subtract(Nx.divide(values, levels / 2), 1)
  end

  # Pack values into bytes
  # 1-bit: 8 values per byte
  # 2-bit: 4 values per byte
  # 4-bit: 2 values per byte
  defp pack_bits(values, 1) do
    {n, dim} = Nx.shape(values)
    # Ensure dim is multiple of 8
    padded_dim = ceil(dim / 8) * 8
    padding = padded_dim - dim

    padded =
      if padding > 0 do
        Nx.pad(values, 0, [{0, 0, 0}, {0, padding, 0}])
      else
        values
      end

    # Reshape to {n, dim/8, 8} and pack
    reshaped = Nx.reshape(padded, {n, div(padded_dim, 8), 8})

    # Pack bits: multiply by powers of 2 and sum
    weights = Nx.tensor([128, 64, 32, 16, 8, 4, 2, 1], type: :u8)
    packed = Nx.sum(Nx.multiply(reshaped, weights), axes: [2])
    Nx.as_type(packed, :u8)
  end

  defp pack_bits(values, 2) do
    {n, dim} = Nx.shape(values)
    # Ensure dim is multiple of 4
    padded_dim = ceil(dim / 4) * 4
    padding = padded_dim - dim

    padded =
      if padding > 0 do
        Nx.pad(values, 0, [{0, 0, 0}, {0, padding, 0}])
      else
        values
      end

    # Reshape to {n, dim/4, 4} and pack
    reshaped = Nx.reshape(padded, {n, div(padded_dim, 4), 4})

    # Pack 4 2-bit values per byte: multiply by powers of 4 and sum
    weights = Nx.tensor([64, 16, 4, 1], type: :u8)
    packed = Nx.sum(Nx.multiply(reshaped, weights), axes: [2])
    Nx.as_type(packed, :u8)
  end

  defp pack_bits(values, 4) do
    {n, dim} = Nx.shape(values)
    # Ensure dim is multiple of 2
    padded_dim = ceil(dim / 2) * 2
    padding = padded_dim - dim

    padded =
      if padding > 0 do
        Nx.pad(values, 0, [{0, 0, 0}, {0, padding, 0}])
      else
        values
      end

    # Reshape to {n, dim/2, 2} and pack
    reshaped = Nx.reshape(padded, {n, div(padded_dim, 2), 2})

    # Pack 2 4-bit values per byte
    weights = Nx.tensor([16, 1], type: :u8)
    packed = Nx.sum(Nx.multiply(reshaped, weights), axes: [2])
    Nx.as_type(packed, :u8)
  end

  # Unpack bytes to values
  defp unpack_bits(packed, 1) do
    {n, packed_dim} = Nx.shape(packed)
    dim = packed_dim * 8

    # Expand each byte to 8 bits
    expanded = Nx.reshape(packed, {n, packed_dim, 1})
    weights = Nx.tensor([[128, 64, 32, 16, 8, 4, 2, 1]], type: :u8)

    # Bitwise AND and check if > 0
    masked = Nx.bitwise_and(expanded, weights)
    bits = Nx.greater(masked, 0) |> Nx.as_type(:f32)
    Nx.reshape(bits, {n, dim})
  end

  defp unpack_bits(packed, 2) do
    {n, packed_dim} = Nx.shape(packed)
    dim = packed_dim * 4

    # Expand each byte to 4 2-bit values
    expanded = Nx.reshape(packed, {n, packed_dim, 1})

    # Extract each 2-bit value using shifts and masks
    # Value 0: bits 6-7, Value 1: bits 4-5, Value 2: bits 2-3, Value 3: bits 0-1
    shifts = Nx.tensor([[6, 4, 2, 0]], type: :u8)
    shifted = Nx.right_shift(expanded, shifts)
    values = Nx.bitwise_and(shifted, 3) |> Nx.as_type(:f32)
    Nx.reshape(values, {n, dim})
  end

  defp unpack_bits(packed, 4) do
    {n, packed_dim} = Nx.shape(packed)
    dim = packed_dim * 2

    # Expand each byte to 2 4-bit values
    expanded = Nx.reshape(packed, {n, packed_dim, 1})

    # Extract each 4-bit value
    shifts = Nx.tensor([[4, 0]], type: :u8)
    shifted = Nx.right_shift(expanded, shifts)
    values = Nx.bitwise_and(shifted, 15) |> Nx.as_type(:f32)
    Nx.reshape(values, {n, dim})
  end

  @doc """
  Returns the compression ratio for given settings.

  ## Examples

      iex> Stephen.Compression.compression_ratio(128, 8)
      3.94
      iex> Stephen.Compression.compression_ratio(128, 1)
      28.44
  """
  @spec compression_ratio(pos_integer(), pos_integer()) :: float()
  def compression_ratio(embedding_dim, residual_bits) do
    # float32
    original_bytes = embedding_dim * 4
    # uint16 for centroid ID
    centroid_bytes = 2

    residual_bytes =
      case residual_bits do
        1 -> div(embedding_dim, 8)
        2 -> div(embedding_dim, 4)
        4 -> div(embedding_dim, 2)
        8 -> embedding_dim
      end

    compressed_bytes = centroid_bytes + residual_bytes
    Float.round(original_bytes / compressed_bytes, 2)
  end
end
