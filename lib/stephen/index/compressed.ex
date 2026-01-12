defmodule Stephen.Index.Compressed do
  @moduledoc """
  Compressed document index using residual compression.

  Combines PLAID-style centroid indexing with ColBERTv2 residual compression
  for memory-efficient storage while maintaining retrieval quality.

  Instead of storing full float32 embeddings (~512 bytes per token), stores:
  - Centroid ID (2 bytes)
  - Quantized residual (dim bytes at 8 bits)

  This achieves ~4-6x compression ratio.

  ## Usage

      # Train compression on document embeddings
      index = Stephen.Index.Compressed.new(embedding_dim: 128)
      index = Stephen.Index.Compressed.train(index, all_doc_embeddings)

      # Add documents (stores compressed)
      index = Stephen.Index.Compressed.add(index, "doc1", embeddings1)
      index = Stephen.Index.Compressed.add(index, "doc2", embeddings2)

      # Search (decompresses on-the-fly)
      results = Stephen.Index.Compressed.search(index, query_embeddings, top_k: 10)
  """

  alias Stephen.Compression
  alias Stephen.KMeans
  alias Stephen.Scorer

  defstruct [
    :compression,
    :centroids,
    :inverted_index,
    :compressed_embeddings,
    :num_centroids,
    :embedding_dim,
    :doc_count,
    :trained?
  ]

  @type t :: %__MODULE__{
          compression: Compression.t() | nil,
          centroids: Nx.Tensor.t() | nil,
          inverted_index: %{non_neg_integer() => MapSet.t()},
          compressed_embeddings: %{term() => Compression.compressed_embedding()},
          num_centroids: pos_integer(),
          embedding_dim: pos_integer(),
          doc_count: non_neg_integer(),
          trained?: boolean()
        }

  @type doc_id :: term()

  @default_num_centroids 1024
  @default_compression_centroids 2048
  @default_nprobe 32

  @doc """
  Creates a new empty compressed index.

  ## Options
    * `:embedding_dim` - Dimension of embeddings (required)
    * `:num_centroids` - Number of PLAID centroids for candidate generation (default: #{@default_num_centroids})
    * `:compression_centroids` - Number of compression centroids (default: #{@default_compression_centroids})
  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    embedding_dim = Keyword.fetch!(opts, :embedding_dim)
    num_centroids = Keyword.get(opts, :num_centroids, @default_num_centroids)

    %__MODULE__{
      compression: nil,
      centroids: nil,
      inverted_index: %{},
      compressed_embeddings: %{},
      num_centroids: num_centroids,
      embedding_dim: embedding_dim,
      doc_count: 0,
      trained?: false
    }
  end

  @doc """
  Trains the compression codebook and PLAID centroids on document embeddings.

  Must be called before adding documents to the index.

  ## Arguments
    * `index` - The compressed index struct
    * `embeddings` - List of embedding tensors or single concatenated tensor

  ## Options
    * `:compression_centroids` - Number of compression centroids (default: #{@default_compression_centroids})
    * `:residual_bits` - Bits for residual quantization (default: 8)
  """
  @spec train(t(), [Nx.Tensor.t()] | Nx.Tensor.t(), keyword()) :: t()
  def train(index, embeddings, opts \\ []) do
    compression_centroids =
      Keyword.get(opts, :compression_centroids, @default_compression_centroids)

    residual_bits = Keyword.get(opts, :residual_bits, 8)

    # Flatten embeddings
    all_embeddings =
      case embeddings do
        [_ | _] -> Nx.concatenate(embeddings, axis: 0)
        tensor -> tensor
      end

    {n, _dim} = Nx.shape(all_embeddings)

    # Train compression codebook
    compression =
      Compression.train(all_embeddings,
        num_centroids: min(compression_centroids, n),
        residual_bits: residual_bits
      )

    # Train PLAID centroids for candidate generation
    plaid_centroids = train_plaid_centroids(all_embeddings, min(index.num_centroids, n))

    %{
      index
      | compression: compression,
        centroids: plaid_centroids,
        trained?: true
    }
  end

  @doc """
  Adds a document's embeddings to the index (stores compressed).

  The index must be trained before adding documents.

  ## Arguments
    * `index` - The compressed index struct
    * `doc_id` - Unique identifier for the document
    * `embeddings` - Tensor of shape {num_tokens, embedding_dim}
  """
  @spec add(t(), doc_id(), Nx.Tensor.t()) :: t()
  def add(%{trained?: false}, _doc_id, _embeddings) do
    raise ArgumentError, "Index must be trained before adding documents. Call train/3 first."
  end

  def add(index, doc_id, embeddings) do
    %{
      compression: compression,
      centroids: centroids,
      inverted_index: inv_idx,
      compressed_embeddings: comp_emb
    } = index

    # Compress embeddings
    compressed = Compression.compress(compression, embeddings)

    # Update PLAID inverted index
    centroid_ids = find_nearest_centroids(embeddings, centroids)
    unique_centroids = centroid_ids |> Nx.to_flat_list() |> Enum.uniq()

    new_inv_idx =
      Enum.reduce(unique_centroids, inv_idx, fn centroid_id, acc ->
        doc_set = Map.get(acc, centroid_id, MapSet.new())
        Map.put(acc, centroid_id, MapSet.put(doc_set, doc_id))
      end)

    %{
      index
      | inverted_index: new_inv_idx,
        compressed_embeddings: Map.put(comp_emb, doc_id, compressed),
        doc_count: index.doc_count + 1
    }
  end

  @doc """
  Adds multiple documents to the index.

  ## Arguments
    * `index` - The compressed index struct
    * `documents` - List of {doc_id, embeddings} tuples
  """
  @spec add_all(t(), [{doc_id(), Nx.Tensor.t()}]) :: t()
  def add_all(index, documents) do
    Enum.reduce(documents, index, fn {doc_id, embeddings}, acc ->
      add(acc, doc_id, embeddings)
    end)
  end

  @doc """
  Indexes documents: trains on all embeddings, then adds each document.

  Convenience function that combines train/3 and add/3.

  ## Arguments
    * `index` - The compressed index struct
    * `documents` - List of {doc_id, embeddings} tuples
  """
  @spec index_documents(t(), [{doc_id(), Nx.Tensor.t()}]) :: t()
  def index_documents(index, documents) do
    # Collect all embeddings for training
    all_embeddings = Enum.map(documents, fn {_id, emb} -> emb end)

    index
    |> train(all_embeddings)
    |> add_all(documents)
  end

  @doc """
  Searches the compressed index for documents matching a query.

  Uses PLAID-style candidate generation followed by reranking
  with decompressed embeddings.

  ## Arguments
    * `index` - The compressed index struct
    * `query_embeddings` - Query token embeddings
    * `opts` - Search options

  ## Options
    * `:top_k` - Number of results to return (default: 10)
    * `:nprobe` - Number of centroids to probe (default: #{@default_nprobe})

  ## Returns
    List of `%{doc_id: term(), score: float()}` sorted by score descending.
  """
  @spec search(t(), Nx.Tensor.t(), keyword()) :: [%{doc_id: doc_id(), score: float()}]
  def search(index, query_embeddings, opts \\ []) do
    top_k = Keyword.get(opts, :top_k, 10)
    nprobe = Keyword.get(opts, :nprobe, @default_nprobe)

    %{
      compression: compression,
      centroids: centroids,
      inverted_index: inv_idx,
      compressed_embeddings: comp_emb
    } = index

    # Get candidate documents via PLAID centroid lookup
    candidates = get_candidates(query_embeddings, centroids, inv_idx, nprobe)

    # Score candidates with full MaxSim (decompressing on-the-fly)
    candidates
    |> Enum.map(fn doc_id ->
      compressed = Map.get(comp_emb, doc_id)
      doc_embeddings = Compression.decompress(compression, compressed)
      score = Scorer.max_sim(query_embeddings, doc_embeddings)
      %{doc_id: doc_id, score: score}
    end)
    |> Enum.sort_by(& &1.score, :desc)
    |> Enum.take(top_k)
  end

  @doc """
  Removes a document from the index.

  ## Arguments
    * `index` - The compressed index struct
    * `doc_id` - The document ID to remove

  ## Returns
    Updated compressed index, or the original index if doc_id not found.
  """
  @spec delete(t(), doc_id()) :: t()
  def delete(index, doc_id) do
    case Map.get(index.compressed_embeddings, doc_id) do
      nil ->
        index

      compressed ->
        %{centroids: centroids, inverted_index: inv_idx, compression: compression} = index

        # Decompress to find which centroids this document was indexed under
        embeddings = Compression.decompress(compression, compressed)
        centroid_ids = find_nearest_centroids(embeddings, centroids)
        unique_centroids = centroid_ids |> Nx.to_flat_list() |> Enum.uniq()

        # Remove doc from inverted index entries
        new_inv_idx = remove_doc_from_inverted_index(inv_idx, unique_centroids, doc_id)

        %{
          index
          | inverted_index: new_inv_idx,
            compressed_embeddings: Map.delete(index.compressed_embeddings, doc_id),
            doc_count: max(index.doc_count - 1, 0)
        }
    end
  end

  defp remove_doc_from_inverted_index(inv_idx, centroid_ids, doc_id) do
    Enum.reduce(centroid_ids, inv_idx, fn centroid_id, acc ->
      remove_doc_from_centroid(acc, centroid_id, doc_id)
    end)
  end

  defp remove_doc_from_centroid(inv_idx, centroid_id, doc_id) do
    case Map.get(inv_idx, centroid_id) do
      nil ->
        inv_idx

      doc_set ->
        new_set = MapSet.delete(doc_set, doc_id)

        if MapSet.size(new_set) == 0 do
          Map.delete(inv_idx, centroid_id)
        else
          Map.put(inv_idx, centroid_id, new_set)
        end
    end
  end

  @doc """
  Removes multiple documents from the index.

  ## Arguments
    * `index` - The compressed index struct
    * `doc_ids` - List of document IDs to remove

  ## Returns
    Updated compressed index.
  """
  @spec delete_all(t(), [doc_id()]) :: t()
  def delete_all(index, doc_ids) do
    Enum.reduce(doc_ids, index, fn doc_id, acc ->
      delete(acc, doc_id)
    end)
  end

  @doc """
  Updates a document in the index by replacing its embeddings.

  This is equivalent to deleting and re-adding the document.

  ## Arguments
    * `index` - The compressed index struct
    * `doc_id` - The document ID to update
    * `embeddings` - New embeddings tensor

  ## Returns
    Updated compressed index.
  """
  @spec update(t(), doc_id(), Nx.Tensor.t()) :: t()
  def update(index, doc_id, embeddings) do
    index
    |> delete(doc_id)
    |> add(doc_id, embeddings)
  end

  @doc """
  Checks if a document exists in the index.
  """
  @spec has_doc?(t(), doc_id()) :: boolean()
  def has_doc?(index, doc_id) do
    Map.has_key?(index.compressed_embeddings, doc_id)
  end

  @doc """
  Gets the decompressed embeddings for a document.
  """
  @spec get_embeddings(t(), doc_id()) :: Nx.Tensor.t() | nil
  def get_embeddings(index, doc_id) do
    case Map.get(index.compressed_embeddings, doc_id) do
      nil -> nil
      compressed -> Compression.decompress(index.compression, compressed)
    end
  end

  @doc """
  Gets the compressed representation for a document.
  """
  @spec get_compressed(t(), doc_id()) :: Compression.compressed_embedding() | nil
  def get_compressed(index, doc_id) do
    Map.get(index.compressed_embeddings, doc_id)
  end

  @doc """
  Returns all document IDs in the index.
  """
  @spec doc_ids(t()) :: [doc_id()]
  def doc_ids(index), do: Map.keys(index.compressed_embeddings)

  @doc """
  Returns the number of documents in the index.
  """
  @spec size(t()) :: non_neg_integer()
  def size(index), do: index.doc_count

  @doc """
  Saves the compressed index to disk.

  ## Arguments
    * `index` - The compressed index struct
    * `path` - File path to save to
  """
  @spec save(t(), Path.t()) :: :ok | {:error, term()}
  def save(index, path) do
    # Serialize compression codebook
    compression_data =
      if index.compression do
        %{
          centroids:
            {Nx.shape(index.compression.centroids), Nx.type(index.compression.centroids),
             Nx.to_binary(index.compression.centroids)},
          num_centroids: index.compression.num_centroids,
          embedding_dim: index.compression.embedding_dim,
          residual_bits: index.compression.residual_bits
        }
      else
        nil
      end

    # Serialize PLAID centroids
    plaid_centroids_data =
      if index.centroids do
        {Nx.shape(index.centroids), Nx.type(index.centroids), Nx.to_binary(index.centroids)}
      else
        nil
      end

    # Serialize compressed embeddings
    compressed_embeddings_data =
      index.compressed_embeddings
      |> Enum.map(fn {doc_id, compressed} ->
        {doc_id,
         %{
           centroid_ids:
             {Nx.shape(compressed.centroid_ids), Nx.type(compressed.centroid_ids),
              Nx.to_binary(compressed.centroid_ids)},
           residuals:
             {Nx.shape(compressed.residuals), Nx.type(compressed.residuals),
              Nx.to_binary(compressed.residuals)}
         }}
      end)
      |> Map.new()

    # Convert inverted index MapSets to lists
    inverted_index_data =
      index.inverted_index
      |> Enum.map(fn {k, v} -> {k, MapSet.to_list(v)} end)
      |> Map.new()

    data = %{
      compression: compression_data,
      plaid_centroids: plaid_centroids_data,
      inverted_index: inverted_index_data,
      compressed_embeddings: compressed_embeddings_data,
      num_centroids: index.num_centroids,
      embedding_dim: index.embedding_dim,
      doc_count: index.doc_count,
      trained: index.trained?
    }

    File.write(path, :erlang.term_to_binary(data))
  end

  @doc """
  Loads a compressed index from disk.

  ## Arguments
    * `path` - File path to load from

  ## Returns
    `{:ok, index}` or `{:error, reason}`
  """
  @spec load(Path.t()) :: {:ok, t()} | {:error, term()}
  def load(path) do
    with {:ok, binary} <- File.read(path) do
      data = :erlang.binary_to_term(binary)

      # Deserialize compression codebook
      compression =
        case data.compression do
          nil ->
            nil

          comp_data ->
            {shape, type, bin} = comp_data.centroids

            %Compression{
              centroids: bin |> Nx.from_binary(type) |> Nx.reshape(shape),
              num_centroids: comp_data.num_centroids,
              embedding_dim: comp_data.embedding_dim,
              residual_bits: comp_data.residual_bits
            }
        end

      # Deserialize PLAID centroids
      plaid_centroids =
        case data.plaid_centroids do
          nil -> nil
          {shape, type, bin} -> bin |> Nx.from_binary(type) |> Nx.reshape(shape)
        end

      # Deserialize compressed embeddings
      compressed_embeddings =
        data.compressed_embeddings
        |> Enum.map(fn {doc_id, compressed_data} ->
          {cid_shape, cid_type, cid_bin} = compressed_data.centroid_ids
          {res_shape, res_type, res_bin} = compressed_data.residuals

          {doc_id,
           %{
             centroid_ids: cid_bin |> Nx.from_binary(cid_type) |> Nx.reshape(cid_shape),
             residuals: res_bin |> Nx.from_binary(res_type) |> Nx.reshape(res_shape)
           }}
        end)
        |> Map.new()

      # Deserialize inverted index
      inverted_index =
        data.inverted_index
        |> Enum.map(fn {k, v} -> {k, MapSet.new(v)} end)
        |> Map.new()

      {:ok,
       %__MODULE__{
         compression: compression,
         centroids: plaid_centroids,
         inverted_index: inverted_index,
         compressed_embeddings: compressed_embeddings,
         num_centroids: data.num_centroids,
         embedding_dim: data.embedding_dim,
         doc_count: data.doc_count,
         trained?: data.trained
       }}
    end
  end

  @doc """
  Returns compression statistics for the index.
  """
  @spec stats(t()) :: map()
  def stats(index) do
    uncompressed_size =
      (index.doc_count * index.embedding_dim * 4 * 100)
      |> div(1)

    compressed_size =
      index.compressed_embeddings
      |> Enum.reduce(0, fn {_doc_id, compressed}, acc ->
        centroid_bytes = Nx.byte_size(compressed.centroid_ids)
        residual_bytes = Nx.byte_size(compressed.residuals)
        acc + centroid_bytes + residual_bytes
      end)

    compression_ratio =
      if compressed_size > 0 do
        Float.round(uncompressed_size / compressed_size, 2)
      else
        0.0
      end

    %{
      doc_count: index.doc_count,
      embedding_dim: index.embedding_dim,
      compression_centroids: if(index.compression, do: index.compression.num_centroids, else: 0),
      plaid_centroids: index.num_centroids,
      compressed_size_bytes: compressed_size,
      compression_ratio: compression_ratio,
      trained: index.trained?
    }
  end

  # Train PLAID centroids using K-means (cosine distance, normalized)
  defp train_plaid_centroids(embeddings, k) do
    KMeans.train(embeddings, k, distance: :cosine, normalize: true)
  end

  # Find nearest centroid for each embedding (cosine similarity)
  defp find_nearest_centroids(embeddings, centroids) do
    KMeans.find_nearest(embeddings, centroids, :cosine)
  end

  # Get candidate documents from inverted index
  defp get_candidates(query_embeddings, centroids, inv_idx, nprobe) do
    similarities = Nx.dot(query_embeddings, Nx.transpose(centroids))
    {_n_query, n_centroids} = Nx.shape(similarities)
    effective_nprobe = min(nprobe, n_centroids)

    top_centroid_indices =
      similarities
      |> Nx.argsort(axis: 1, direction: :desc)
      |> Nx.slice_along_axis(0, effective_nprobe, axis: 1)
      |> Nx.to_flat_list()
      |> Enum.uniq()

    top_centroid_indices
    |> Enum.flat_map(fn centroid_id ->
      Map.get(inv_idx, centroid_id, MapSet.new()) |> MapSet.to_list()
    end)
    |> Enum.uniq()
  end
end
