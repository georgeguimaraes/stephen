defmodule Stephen.Plaid do
  @moduledoc """
  PLAID-style indexing for efficient ColBERT retrieval.

  PLAID (Performance-optimized Late Interaction Driver) uses centroid-based
  candidate generation for faster retrieval:

  1. Cluster all document token embeddings into K centroids
  2. Build inverted lists: centroid -> [doc_ids with tokens near that centroid]
  3. At query time, find nearest centroids for query tokens
  4. Retrieve candidate docs from inverted lists
  5. Rerank candidates with full MaxSim scoring

  This achieves sub-linear search time for large collections.
  """

  alias Stephen.KMeans
  alias Stephen.Scorer

  defstruct [
    :centroids,
    :inverted_index,
    :doc_embeddings,
    :num_centroids,
    :embedding_dim,
    :doc_count
  ]

  @type t :: %__MODULE__{
          centroids: Nx.Tensor.t() | nil,
          inverted_index: %{non_neg_integer() => MapSet.t()},
          doc_embeddings: %{term() => Nx.Tensor.t()},
          num_centroids: pos_integer(),
          embedding_dim: pos_integer(),
          doc_count: non_neg_integer()
        }

  @type doc_id :: term()

  @default_num_centroids 1024
  @default_nprobe 32
  @default_kmeans_iterations 10

  @doc """
  Creates a new PLAID index.

  ## Options
    * `:embedding_dim` - Dimension of embeddings (required)
    * `:num_centroids` - Number of centroids for clustering (default: #{@default_num_centroids})

  ## Examples

      plaid = Stephen.Plaid.new(embedding_dim: 128, num_centroids: 1024)
  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    embedding_dim = Keyword.fetch!(opts, :embedding_dim)
    num_centroids = Keyword.get(opts, :num_centroids, @default_num_centroids)

    %__MODULE__{
      centroids: nil,
      inverted_index: %{},
      doc_embeddings: %{},
      num_centroids: num_centroids,
      embedding_dim: embedding_dim,
      doc_count: 0
    }
  end

  @doc """
  Indexes documents into the PLAID index.

  The first call will train centroids on the provided embeddings.
  Subsequent calls will use the existing centroids.

  ## Arguments
    * `plaid` - PLAID index
    * `documents` - List of {doc_id, embeddings} tuples
  """
  @spec index_documents(t(), [{doc_id(), Nx.Tensor.t()}]) :: t()
  def index_documents(plaid, documents) do
    # Collect all embeddings for training if needed
    all_embeddings =
      documents
      |> Enum.map(fn {_id, emb} -> emb end)
      |> Nx.concatenate(axis: 0)

    # Train centroids if not already done
    plaid =
      if plaid.centroids == nil do
        centroids = train_centroids(all_embeddings, plaid.num_centroids)
        %{plaid | centroids: centroids}
      else
        plaid
      end

    # Add each document to the index
    Enum.reduce(documents, plaid, fn {doc_id, embeddings}, acc ->
      add_document(acc, doc_id, embeddings)
    end)
  end

  @doc """
  Adds a single document to the index.
  """
  @spec add_document(t(), doc_id(), Nx.Tensor.t()) :: t()
  def add_document(plaid, doc_id, embeddings) do
    %{centroids: centroids, inverted_index: inv_idx, doc_embeddings: doc_emb} = plaid

    # Find nearest centroid for each token
    centroid_ids = find_nearest_centroids(embeddings, centroids)
    unique_centroids = centroid_ids |> Nx.to_flat_list() |> Enum.uniq()

    # Update inverted index
    new_inv_idx =
      Enum.reduce(unique_centroids, inv_idx, fn centroid_id, acc ->
        doc_set = Map.get(acc, centroid_id, MapSet.new())
        Map.put(acc, centroid_id, MapSet.put(doc_set, doc_id))
      end)

    %{
      plaid
      | inverted_index: new_inv_idx,
        doc_embeddings: Map.put(doc_emb, doc_id, embeddings),
        doc_count: plaid.doc_count + 1
    }
  end

  @doc """
  Searches the PLAID index for documents matching a query.

  ## Arguments
    * `plaid` - PLAID index
    * `query_embeddings` - Query token embeddings
    * `opts` - Search options

  ## Options
    * `:top_k` - Number of results to return (default: 10)
    * `:nprobe` - Number of centroids to probe per query token (default: #{@default_nprobe})

  ## Returns
    List of `%{doc_id: term(), score: float()}` sorted by score descending.
  """
  @spec search(t(), Nx.Tensor.t(), keyword()) :: [%{doc_id: doc_id(), score: float()}]
  def search(plaid, query_embeddings, opts \\ []) do
    top_k = Keyword.get(opts, :top_k, 10)
    nprobe = Keyword.get(opts, :nprobe, @default_nprobe)

    %{centroids: centroids, inverted_index: inv_idx, doc_embeddings: doc_emb} = plaid

    # Find candidate documents via centroid lookup
    candidates = get_candidates(query_embeddings, centroids, inv_idx, nprobe)

    # Score candidates with full MaxSim
    candidates
    |> Enum.map(fn doc_id ->
      doc_embeddings = Map.get(doc_emb, doc_id)
      score = Scorer.max_sim(query_embeddings, doc_embeddings)
      %{doc_id: doc_id, score: score}
    end)
    |> Enum.sort_by(& &1.score, :desc)
    |> Enum.take(top_k)
  end

  @doc """
  Removes a document from the index.

  ## Arguments
    * `plaid` - PLAID index
    * `doc_id` - The document ID to remove

  ## Returns
    Updated PLAID index, or the original index if doc_id not found.
  """
  @spec delete(t(), doc_id()) :: t()
  def delete(plaid, doc_id) do
    case Map.get(plaid.doc_embeddings, doc_id) do
      nil ->
        plaid

      embeddings ->
        %{centroids: centroids, inverted_index: inv_idx} = plaid

        # Find which centroids this document was indexed under
        centroid_ids = find_nearest_centroids(embeddings, centroids)
        unique_centroids = centroid_ids |> Nx.to_flat_list() |> Enum.uniq()

        # Remove doc from inverted index entries
        new_inv_idx = remove_doc_from_inverted_index(inv_idx, unique_centroids, doc_id)

        %{
          plaid
          | inverted_index: new_inv_idx,
            doc_embeddings: Map.delete(plaid.doc_embeddings, doc_id),
            doc_count: max(plaid.doc_count - 1, 0)
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
    * `plaid` - PLAID index
    * `doc_ids` - List of document IDs to remove

  ## Returns
    Updated PLAID index.
  """
  @spec delete_all(t(), [doc_id()]) :: t()
  def delete_all(plaid, doc_ids) do
    Enum.reduce(doc_ids, plaid, fn doc_id, acc ->
      delete(acc, doc_id)
    end)
  end

  @doc """
  Updates a document in the index by replacing its embeddings.

  This is equivalent to deleting and re-adding the document.

  ## Arguments
    * `plaid` - PLAID index
    * `doc_id` - The document ID to update
    * `embeddings` - New embeddings tensor

  ## Returns
    Updated PLAID index.
  """
  @spec update(t(), doc_id(), Nx.Tensor.t()) :: t()
  def update(plaid, doc_id, embeddings) do
    plaid
    |> delete(doc_id)
    |> add_document(doc_id, embeddings)
  end

  @doc """
  Checks if a document exists in the index.
  """
  @spec has_doc?(t(), doc_id()) :: boolean()
  def has_doc?(plaid, doc_id) do
    Map.has_key?(plaid.doc_embeddings, doc_id)
  end

  @doc """
  Returns the number of documents in the index.
  """
  @spec size(t()) :: non_neg_integer()
  def size(plaid), do: plaid.doc_count

  @doc """
  Returns all document IDs in the index.
  """
  @spec doc_ids(t()) :: [doc_id()]
  def doc_ids(plaid), do: Map.keys(plaid.doc_embeddings)

  @doc """
  Gets the embeddings for a document.
  """
  @spec get_embeddings(t(), doc_id()) :: Nx.Tensor.t() | nil
  def get_embeddings(plaid, doc_id), do: Map.get(plaid.doc_embeddings, doc_id)

  @doc """
  Saves the PLAID index to disk.

  ## Arguments
    * `plaid` - PLAID index to save
    * `path` - File path to save to
  """
  @spec save(t(), Path.t()) :: :ok | {:error, term()}
  def save(plaid, path) do
    # Serialize centroids
    centroids_data =
      if plaid.centroids do
        {Nx.shape(plaid.centroids), Nx.type(plaid.centroids), Nx.to_binary(plaid.centroids)}
      else
        nil
      end

    # Serialize doc embeddings
    doc_embeddings_data =
      plaid.doc_embeddings
      |> Enum.map(fn {doc_id, emb} ->
        {doc_id, {Nx.shape(emb), Nx.type(emb), Nx.to_binary(emb)}}
      end)
      |> Map.new()

    # Convert inverted index MapSets to lists for serialization
    inverted_index_data =
      plaid.inverted_index
      |> Enum.map(fn {k, v} -> {k, MapSet.to_list(v)} end)
      |> Map.new()

    data = %{
      centroids: centroids_data,
      inverted_index: inverted_index_data,
      doc_embeddings: doc_embeddings_data,
      num_centroids: plaid.num_centroids,
      embedding_dim: plaid.embedding_dim,
      doc_count: plaid.doc_count
    }

    File.write(path, :erlang.term_to_binary(data))
  end

  @doc """
  Loads a PLAID index from disk.

  ## Arguments
    * `path` - File path to load from

  ## Returns
    `{:ok, plaid}` or `{:error, reason}`
  """
  @spec load(Path.t()) :: {:ok, t()} | {:error, term()}
  def load(path) do
    with {:ok, binary} <- File.read(path) do
      data = :erlang.binary_to_term(binary)

      # Deserialize centroids
      centroids =
        case data.centroids do
          {shape, type, bin} ->
            bin |> Nx.from_binary(type) |> Nx.reshape(shape)

          nil ->
            nil
        end

      # Deserialize doc embeddings
      doc_embeddings =
        data.doc_embeddings
        |> Enum.map(fn {doc_id, {shape, type, bin}} ->
          emb = bin |> Nx.from_binary(type) |> Nx.reshape(shape)
          {doc_id, emb}
        end)
        |> Map.new()

      # Convert inverted index lists back to MapSets
      inverted_index =
        data.inverted_index
        |> Enum.map(fn {k, v} -> {k, MapSet.new(v)} end)
        |> Map.new()

      {:ok,
       %__MODULE__{
         centroids: centroids,
         inverted_index: inverted_index,
         doc_embeddings: doc_embeddings,
         num_centroids: data.num_centroids,
         embedding_dim: data.embedding_dim,
         doc_count: data.doc_count
       }}
    end
  end

  # Train centroids using K-means (cosine distance, normalized)
  defp train_centroids(embeddings, num_centroids) do
    KMeans.train(embeddings, num_centroids,
      iterations: @default_kmeans_iterations,
      distance: :cosine,
      normalize: true
    )
  end

  # Find nearest centroid for each embedding (cosine similarity)
  defp find_nearest_centroids(embeddings, centroids) do
    KMeans.find_nearest(embeddings, centroids, :cosine)
  end

  # Get candidate documents from inverted index
  defp get_candidates(query_embeddings, centroids, inv_idx, nprobe) do
    # For each query token, find top-nprobe centroids
    similarities = Nx.dot(query_embeddings, Nx.transpose(centroids))

    # Get top nprobe centroid indices for each query token
    {_n_query, n_centroids} = Nx.shape(similarities)
    effective_nprobe = min(nprobe, n_centroids)

    # Get indices of top centroids for each query token
    top_centroid_indices =
      similarities
      |> Nx.argsort(axis: 1, direction: :desc)
      |> Nx.slice_along_axis(0, effective_nprobe, axis: 1)
      |> Nx.to_flat_list()
      |> Enum.uniq()

    # Collect all doc_ids from inverted lists
    top_centroid_indices
    |> Enum.flat_map(fn centroid_id ->
      Map.get(inv_idx, centroid_id, MapSet.new()) |> MapSet.to_list()
    end)
    |> Enum.uniq()
  end
end
