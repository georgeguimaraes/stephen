defmodule Stephen.Chunker do
  @moduledoc """
  Passage chunking for long documents.

  ColBERT has a maximum document length (typically 180 tokens). For longer
  documents, we split them into overlapping chunks and track the mapping
  back to original documents.

  Stephen uses sentence-aware recursive chunking via text_chunker, which
  splits at semantic boundaries (sentences, paragraphs). Research shows
  ColBERT performs best with sentence-aware splitting.

  ## Usage

      # Split documents into chunks
      {chunks, mapping} = Stephen.Chunker.chunk_documents(documents)

      # With custom size
      {chunks, mapping} = Stephen.Chunker.chunk_documents(documents,
        chunk_size: 500,
        chunk_overlap: 100
      )

      # For markdown documents
      {chunks, mapping} = Stephen.Chunker.chunk_documents(documents,
        format: :markdown
      )

      # After retrieval, merge results back to document level
      merged_results = Stephen.Chunker.merge_results(chunk_results, mapping)
  """

  @type doc_id :: term()
  @type chunk_id :: String.t()
  @type chunk_mapping :: %{chunk_id() => %{doc_id: doc_id(), chunk_index: non_neg_integer()}}

  # ~500 chars ≈ 100-125 tokens, good for ColBERT's sweet spot
  @default_chunk_size 500
  @default_chunk_overlap 100

  @doc """
  Splits documents into overlapping chunks.

  ## Arguments
    * `documents` - List of {doc_id, text} tuples
    * `opts` - Chunking options

  ## Options
    * `:chunk_size` - Target chunk size in characters (default: #{@default_chunk_size})
    * `:chunk_overlap` - Overlap between chunks in characters (default: #{@default_chunk_overlap})
    * `:format` - Text format for separator selection (`:plaintext` or `:markdown`, default: `:plaintext`)

  ## Returns
    Tuple of `{chunks, mapping}` where:
    - `chunks` is a list of {chunk_id, text} tuples
    - `mapping` is a map from chunk_id to original doc info
  """
  @spec chunk_documents([{doc_id(), String.t()}], keyword()) ::
          {[{chunk_id(), String.t()}], chunk_mapping()}
  def chunk_documents(documents, opts \\ []) do
    {chunks, mapping} =
      documents
      |> Enum.flat_map_reduce(%{}, fn {doc_id, text}, mapping ->
        doc_chunks = chunk_document(text, doc_id, opts)

        new_mapping =
          doc_chunks
          |> Enum.with_index()
          |> Enum.reduce(mapping, fn {{chunk_id, _text}, chunk_idx}, acc ->
            Map.put(acc, chunk_id, %{doc_id: doc_id, chunk_index: chunk_idx})
          end)

        {doc_chunks, new_mapping}
      end)

    {chunks, mapping}
  end

  @doc """
  Chunks a single text into overlapping segments.

  ## Arguments
    * `text` - Text to chunk
    * `opts` - Chunking options (same as chunk_documents/2)

  ## Returns
    List of text chunks (strings)
  """
  @spec chunk_text(String.t(), keyword()) :: [String.t()]
  def chunk_text(text, opts \\ []) do
    chunk_size = Keyword.get(opts, :chunk_size, @default_chunk_size)
    chunk_overlap = Keyword.get(opts, :chunk_overlap, @default_chunk_overlap)
    format = Keyword.get(opts, :format, :plaintext)

    text
    |> TextChunker.split(
      chunk_size: chunk_size,
      chunk_overlap: chunk_overlap,
      format: format
    )
    |> Enum.map(& &1.text)
  end

  @doc """
  Merges chunk-level results back to document level.

  Takes the maximum score among all chunks of the same document.

  ## Arguments
    * `results` - List of %{doc_id: chunk_id, score: float} from search
    * `mapping` - Chunk mapping from chunk_documents/2

  ## Options
    * `:aggregation` - How to combine chunk scores (:max, :mean, :sum) (default: :max)

  ## Returns
    List of %{doc_id: original_doc_id, score: float} sorted by score descending.
  """
  @spec merge_results([%{doc_id: chunk_id(), score: float()}], chunk_mapping(), keyword()) :: [
          %{doc_id: doc_id(), score: float()}
        ]
  def merge_results(results, mapping, opts \\ []) do
    aggregation = Keyword.get(opts, :aggregation, :max)

    results
    |> Enum.map(fn %{doc_id: chunk_id, score: score} ->
      case Map.get(mapping, chunk_id) do
        %{doc_id: doc_id} -> {doc_id, score}
        nil -> {chunk_id, score}
      end
    end)
    |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))
    |> Enum.map(fn {doc_id, scores} ->
      aggregated_score =
        case aggregation do
          :max -> Enum.max(scores)
          :mean -> Enum.sum(scores) / length(scores)
          :sum -> Enum.sum(scores)
        end

      %{doc_id: doc_id, score: aggregated_score}
    end)
    |> Enum.sort_by(& &1.score, :desc)
  end

  @doc """
  Calculates how many chunks a text will produce.

  Useful for estimating index size before indexing.

  ## Arguments
    * `text` - Text to analyze
    * `opts` - Same options as chunk_text/2

  ## Returns
    Number of chunks
  """
  @spec estimate_chunks(String.t(), keyword()) :: non_neg_integer()
  def estimate_chunks(text, opts \\ []) do
    text
    |> chunk_text(opts)
    |> length()
  end

  @doc """
  Returns the original document ID for a chunk.

  ## Arguments
    * `chunk_id` - The chunk identifier
    * `mapping` - Chunk mapping from chunk_documents/2

  ## Returns
    The original document ID or nil if not found
  """
  @spec get_doc_id(chunk_id(), chunk_mapping()) :: doc_id() | nil
  def get_doc_id(chunk_id, mapping) do
    case Map.get(mapping, chunk_id) do
      %{doc_id: doc_id} -> doc_id
      nil -> nil
    end
  end

  @doc """
  Gets all chunk IDs for a document.

  ## Arguments
    * `doc_id` - The original document ID
    * `mapping` - Chunk mapping from chunk_documents/2

  ## Returns
    List of chunk IDs belonging to the document
  """
  @spec get_chunk_ids(doc_id(), chunk_mapping()) :: [chunk_id()]
  def get_chunk_ids(doc_id, mapping) do
    mapping
    |> Enum.filter(fn {_chunk_id, info} -> info.doc_id == doc_id end)
    |> Enum.map(fn {chunk_id, _info} -> chunk_id end)
  end

  # Private

  defp chunk_document(text, doc_id, opts) do
    chunks = chunk_text(text, opts)

    chunks
    |> Enum.with_index()
    |> Enum.map(fn {chunk_text, idx} ->
      chunk_id = generate_chunk_id(doc_id, idx)
      {chunk_id, chunk_text}
    end)
  end

  defp generate_chunk_id(doc_id, chunk_index) do
    "#{doc_id}__chunk_#{chunk_index}"
  end
end
