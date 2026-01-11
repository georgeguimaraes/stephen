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
    :doc_to_tokens,
    :doc_count,
    :token_count,
    :deleted_token_ids
  ]

  @type t :: %__MODULE__{
          hnsw_index: HNSWLib.Index.t(),
          embedding_dim: non_neg_integer(),
          doc_embeddings: %{term() => Nx.Tensor.t()},
          token_to_doc: %{non_neg_integer() => term()},
          doc_to_tokens: %{term() => [non_neg_integer()]},
          doc_count: non_neg_integer(),
          token_count: non_neg_integer(),
          deleted_token_ids: [non_neg_integer()]
        }

  @type doc_id :: term()

  @doc """
  Creates a new empty index.

  ## Options
    * `:embedding_dim` - Dimension of embeddings (required)
    * `:space` - Distance space, :cosine or :l2 (default: :cosine)
    * `:max_tokens` - Maximum number of token embeddings (default: 100_000)
    * `:m` - HNSW M parameter (default: 16)
    * `:ef_construction` - HNSW ef_construction parameter (default: 200)
  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    embedding_dim = Keyword.fetch!(opts, :embedding_dim)
    space = Keyword.get(opts, :space, :cosine)
    max_tokens = Keyword.get(opts, :max_tokens, 100_000)
    m = Keyword.get(opts, :m, 16)
    ef_construction = Keyword.get(opts, :ef_construction, 200)

    {:ok, hnsw_index} =
      HNSWLib.Index.new(space, embedding_dim, max_tokens,
        m: m,
        ef_construction: ef_construction
      )

    %__MODULE__{
      hnsw_index: hnsw_index,
      embedding_dim: embedding_dim,
      doc_embeddings: %{},
      token_to_doc: %{},
      doc_to_tokens: %{},
      doc_count: 0,
      token_count: 0,
      deleted_token_ids: []
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
      doc_to_tokens: doc_to_tokens,
      doc_count: doc_count,
      token_count: token_count,
      deleted_token_ids: deleted_ids
    } = index

    {num_tokens, _dim} = Nx.shape(embeddings)

    # Convert embeddings to list of vectors for HNSWLib
    embeddings_list =
      embeddings
      |> Nx.to_batched(1)
      |> Enum.map(&Nx.squeeze/1)

    # Reuse deleted IDs first, then allocate new ones
    {token_ids, remaining_deleted, new_token_count} =
      allocate_token_ids(deleted_ids, num_tokens, token_count)

    # Add each token embedding to the HNSW index
    {new_token_to_doc, assigned_ids} =
      embeddings_list
      |> Enum.zip(token_ids)
      |> Enum.reduce({token_to_doc, []}, fn {embedding, token_idx}, {acc_map, acc_ids} ->
        # Unmark if this was a previously deleted ID
        if token_idx in deleted_ids do
          HNSWLib.Index.unmark_deleted(hnsw_index, token_idx)
        end

        :ok = HNSWLib.Index.add_items(hnsw_index, embedding, ids: [token_idx])
        {Map.put(acc_map, token_idx, doc_id), [token_idx | acc_ids]}
      end)

    %{
      index
      | doc_embeddings: Map.put(doc_embeddings, doc_id, embeddings),
        token_to_doc: new_token_to_doc,
        doc_to_tokens: Map.put(doc_to_tokens, doc_id, Enum.reverse(assigned_ids)),
        doc_count: doc_count + 1,
        token_count: new_token_count,
        deleted_token_ids: remaining_deleted
    }
  end

  # Allocate token IDs, reusing deleted ones first
  defp allocate_token_ids(deleted_ids, num_needed, current_count) do
    num_reused = min(length(deleted_ids), num_needed)
    {reused_ids, remaining_deleted} = Enum.split(deleted_ids, num_reused)

    num_new = num_needed - num_reused

    new_ids =
      if num_new > 0 do
        Enum.to_list(current_count..(current_count + num_new - 1))
      else
        []
      end

    all_ids = reused_ids ++ new_ids
    new_count = current_count + num_new

    {all_ids, remaining_deleted, new_count}
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
  Removes a document from the index.

  The document's token embeddings are marked as deleted in the HNSW index
  and their IDs are saved for reuse when new documents are added.

  ## Arguments
    * `index` - The index struct
    * `doc_id` - The document ID to remove

  ## Returns
    Updated index struct, or the original index if doc_id not found.
  """
  @spec delete(t(), doc_id()) :: t()
  def delete(index, doc_id) do
    case Map.get(index.doc_to_tokens, doc_id) do
      nil ->
        index

      token_ids ->
        %{
          hnsw_index: hnsw_index,
          doc_embeddings: doc_embeddings,
          token_to_doc: token_to_doc,
          doc_to_tokens: doc_to_tokens,
          doc_count: doc_count,
          deleted_token_ids: deleted_ids
        } = index

        # Mark tokens as deleted in HNSW
        Enum.each(token_ids, fn token_id ->
          HNSWLib.Index.mark_deleted(hnsw_index, token_id)
        end)

        # Remove from mappings
        new_token_to_doc = Map.drop(token_to_doc, token_ids)

        %{
          index
          | doc_embeddings: Map.delete(doc_embeddings, doc_id),
            token_to_doc: new_token_to_doc,
            doc_to_tokens: Map.delete(doc_to_tokens, doc_id),
            doc_count: max(doc_count - 1, 0),
            deleted_token_ids: token_ids ++ deleted_ids
        }
    end
  end

  @doc """
  Removes multiple documents from the index.

  ## Arguments
    * `index` - The index struct
    * `doc_ids` - List of document IDs to remove

  ## Returns
    Updated index struct.
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
    * `index` - The index struct
    * `doc_id` - The document ID to update
    * `embeddings` - New embeddings tensor

  ## Returns
    Updated index struct.
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
    Map.has_key?(index.doc_embeddings, doc_id)
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
      doc_to_tokens: index.doc_to_tokens,
      doc_count: index.doc_count,
      token_count: index.token_count,
      deleted_token_ids: index.deleted_token_ids
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
         doc_to_tokens: Map.get(metadata, :doc_to_tokens, %{}),
         doc_count: metadata.doc_count,
         token_count: metadata.token_count,
         deleted_token_ids: Map.get(metadata, :deleted_token_ids, [])
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
