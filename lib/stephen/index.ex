defmodule Stephen.Index do
  @moduledoc """
  Manages the ColBERT document index.

  Stores per-token embeddings from documents and enables efficient
  approximate nearest neighbor search using HNSWLib.

  Each token embedding in the index maps back to its source document,
  enabling document-level retrieval through token-level search.
  """

  defstruct [
    :hnsw_index,
    :embedding_dim,
    :doc_embeddings,
    :token_to_doc,
    :doc_count,
    :token_count
  ]

  @type t :: %__MODULE__{
          hnsw_index: HNSWLib.Index.t(),
          embedding_dim: non_neg_integer(),
          doc_embeddings: %{term() => Nx.Tensor.t()},
          token_to_doc: %{non_neg_integer() => term()},
          doc_count: non_neg_integer(),
          token_count: non_neg_integer()
        }

  @type doc_id :: term()

  @doc """
  Creates a new empty index.

  ## Options
    * `:embedding_dim` - Dimension of embeddings (required)
    * `:space` - Distance space, :cosine or :l2 (default: :cosine)
    * `:max_elements` - Maximum number of token embeddings (default: 100_000)
    * `:m` - HNSW M parameter (default: 16)
    * `:ef_construction` - HNSW ef_construction parameter (default: 200)
  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    embedding_dim = Keyword.fetch!(opts, :embedding_dim)
    space = Keyword.get(opts, :space, :cosine)
    max_elements = Keyword.get(opts, :max_elements, 100_000)
    m = Keyword.get(opts, :m, 16)
    ef_construction = Keyword.get(opts, :ef_construction, 200)

    {:ok, hnsw_index} =
      HNSWLib.Index.new(space, embedding_dim, max_elements,
        m: m,
        ef_construction: ef_construction
      )

    %__MODULE__{
      hnsw_index: hnsw_index,
      embedding_dim: embedding_dim,
      doc_embeddings: %{},
      token_to_doc: %{},
      doc_count: 0,
      token_count: 0
    }
  end

  @doc """
  Adds a document's embeddings to the index.

  ## Arguments
    * `index` - The index struct
    * `doc_id` - Unique identifier for the document
    * `embeddings` - Tensor of shape {num_tokens, embedding_dim}

  ## Returns
    Updated index struct.
  """
  @spec add(t(), doc_id(), Nx.Tensor.t()) :: t()
  def add(index, doc_id, embeddings) do
    %{
      hnsw_index: hnsw_index,
      doc_embeddings: doc_embeddings,
      token_to_doc: token_to_doc,
      doc_count: doc_count,
      token_count: token_count
    } = index

    {num_tokens, _dim} = Nx.shape(embeddings)

    # Convert embeddings to list of vectors for HNSWLib
    embeddings_list =
      embeddings
      |> Nx.to_batched(1)
      |> Enum.map(&Nx.squeeze/1)

    # Add each token embedding to the HNSW index
    new_token_to_doc =
      embeddings_list
      |> Enum.with_index(token_count)
      |> Enum.reduce(token_to_doc, fn {embedding, token_idx}, acc ->
        :ok = HNSWLib.Index.add_items(hnsw_index, embedding, ids: [token_idx])
        Map.put(acc, token_idx, doc_id)
      end)

    %{
      index
      | doc_embeddings: Map.put(doc_embeddings, doc_id, embeddings),
        token_to_doc: new_token_to_doc,
        doc_count: doc_count + 1,
        token_count: token_count + num_tokens
    }
  end

  @doc """
  Adds multiple documents to the index.

  ## Arguments
    * `index` - The index struct
    * `documents` - List of {doc_id, embeddings} tuples

  ## Returns
    Updated index struct.
  """
  @spec add_all(t(), [{doc_id(), Nx.Tensor.t()}]) :: t()
  def add_all(index, documents) do
    Enum.reduce(documents, index, fn {doc_id, embeddings}, acc ->
      add(acc, doc_id, embeddings)
    end)
  end

  @doc """
  Searches for the k nearest token embeddings to the query tokens.

  Returns candidate document IDs with their matching token counts.

  ## Arguments
    * `index` - The index struct
    * `query_embeddings` - Tensor of shape {query_len, embedding_dim}
    * `k` - Number of nearest neighbors per query token (default: 10)

  ## Returns
    Map of doc_id => count of matching tokens
  """
  @spec search_tokens(t(), Nx.Tensor.t(), pos_integer()) :: %{doc_id() => pos_integer()}
  def search_tokens(index, query_embeddings, k \\ 10) do
    %{hnsw_index: hnsw_index, token_to_doc: token_to_doc, token_count: total_tokens} = index

    # Can't query for more neighbors than exist in the index
    effective_k = min(k, total_tokens)

    if effective_k == 0 do
      %{}
    else
      query_embeddings
      |> Nx.to_batched(1)
      |> Enum.map(&Nx.squeeze/1)
      |> Enum.flat_map(fn query_token ->
        {:ok, labels, _distances} =
          HNSWLib.Index.knn_query(hnsw_index, query_token, k: effective_k)

        labels
        |> Nx.to_flat_list()
        |> Enum.map(&Map.get(token_to_doc, trunc(&1)))
        |> Enum.reject(&is_nil/1)
      end)
      |> Enum.frequencies()
    end
  end

  @doc """
  Gets the stored embeddings for a document.
  """
  @spec get_embeddings(t(), doc_id()) :: Nx.Tensor.t() | nil
  def get_embeddings(index, doc_id) do
    Map.get(index.doc_embeddings, doc_id)
  end

  @doc """
  Returns all document IDs in the index.
  """
  @spec doc_ids(t()) :: [doc_id()]
  def doc_ids(index) do
    Map.keys(index.doc_embeddings)
  end

  @doc """
  Returns the number of documents in the index.
  """
  @spec size(t()) :: non_neg_integer()
  def size(index), do: index.doc_count

  @doc """
  Returns the number of token embeddings in the index.
  """
  @spec token_count(t()) :: non_neg_integer()
  def token_count(index), do: index.token_count

  @doc """
  Saves the index to disk.

  ## Arguments
    * `index` - The index struct
    * `path` - Directory path to save the index
  """
  @spec save(t(), Path.t()) :: :ok | {:error, term()}
  def save(index, path) do
    File.mkdir_p!(path)

    # Save HNSW index
    hnsw_path = Path.join(path, "hnsw.bin")
    :ok = HNSWLib.Index.save_index(index.hnsw_index, hnsw_path)

    # Save metadata and embeddings
    metadata = %{
      embedding_dim: index.embedding_dim,
      doc_embeddings: serialize_embeddings(index.doc_embeddings),
      token_to_doc: index.token_to_doc,
      doc_count: index.doc_count,
      token_count: index.token_count
    }

    metadata_path = Path.join(path, "metadata.etf")
    File.write!(metadata_path, :erlang.term_to_binary(metadata))

    :ok
  end

  @doc """
  Loads an index from disk.

  ## Arguments
    * `path` - Directory path where the index was saved
  """
  @spec load(Path.t()) :: {:ok, t()} | {:error, term()}
  def load(path) do
    metadata_path = Path.join(path, "metadata.etf")
    hnsw_path = Path.join(path, "hnsw.bin")

    with {:ok, metadata_bin} <- File.read(metadata_path),
         metadata = :erlang.binary_to_term(metadata_bin),
         {:ok, hnsw_index} <-
           HNSWLib.Index.load_index(:cosine, metadata.embedding_dim, hnsw_path) do
      {:ok,
       %__MODULE__{
         hnsw_index: hnsw_index,
         embedding_dim: metadata.embedding_dim,
         doc_embeddings: deserialize_embeddings(metadata.doc_embeddings),
         token_to_doc: metadata.token_to_doc,
         doc_count: metadata.doc_count,
         token_count: metadata.token_count
       }}
    end
  end

  # Serialize embeddings to binary format for storage
  defp serialize_embeddings(doc_embeddings) do
    Map.new(doc_embeddings, fn {doc_id, tensor} ->
      {doc_id, {Nx.shape(tensor), Nx.type(tensor), Nx.to_binary(tensor)}}
    end)
  end

  # Deserialize embeddings from binary format
  defp deserialize_embeddings(serialized) do
    Map.new(serialized, fn {doc_id, {shape, type, binary}} ->
      {doc_id, Nx.from_binary(binary, type) |> Nx.reshape(shape)}
    end)
  end
end
