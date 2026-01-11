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
          centroids: Nx.Tensor.t(),
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

  # Train centroids using K-means
  defp train_centroids(embeddings, num_centroids) do
    {n, _dim} = Nx.shape(embeddings)

    # Ensure we don't have more centroids than embeddings
    k = min(num_centroids, n)

    # Initialize centroids by random selection
    key = Nx.Random.key(42)
    {indices, _key} = Nx.Random.randint(key, 0, n, shape: {k})
    initial_centroids = Nx.take(embeddings, indices, axis: 0)

    # Run K-means iterations
    Enum.reduce(1..@default_kmeans_iterations, initial_centroids, fn _i, centroids ->
      assignments = find_nearest_centroids(embeddings, centroids)
      update_centroids(embeddings, assignments, k)
    end)
  end

  # Find nearest centroid for each embedding
  defp find_nearest_centroids(embeddings, centroids) do
    # Compute pairwise cosine similarity (embeddings should be normalized)
    similarities = Nx.dot(embeddings, Nx.transpose(centroids))
    Nx.argmax(similarities, axis: 1)
  end

  # Update centroids as mean of assigned points
  defp update_centroids(embeddings, assignments, k) do
    centroids =
      for i <- 0..(k - 1) do
        mask = Nx.equal(assignments, i)
        count = Nx.sum(mask) |> Nx.to_number()

        if count > 0 do
          mask_expanded = Nx.reshape(mask, {:auto, 1})
          masked = Nx.multiply(embeddings, mask_expanded)
          sum = Nx.sum(masked, axes: [0])
          centroid = Nx.divide(sum, count)
          # L2 normalize centroid
          norm = Nx.LinAlg.norm(centroid)
          Nx.divide(centroid, Nx.max(norm, 1.0e-9))
        else
          embeddings[0]
        end
      end

    Nx.stack(centroids)
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
