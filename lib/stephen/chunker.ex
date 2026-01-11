defmodule Stephen.Chunker do
  @moduledoc """
  Passage chunking for long documents.

  ColBERT has a maximum document length (typically 180 tokens). For longer
  documents, we split them into overlapping chunks and track the mapping
  back to original documents.

  ## Usage

      # Split documents into chunks
      {chunks, mapping} = Stephen.Chunker.chunk_documents(documents, max_length: 180, stride: 90)

      # After retrieval, merge results back to document level
      merged_results = Stephen.Chunker.merge_results(chunk_results, mapping)

  ## How it works

  Given a document with 400 tokens and max_length=180, stride=90:
  - Chunk 0: tokens 0-179
  - Chunk 1: tokens 90-269
  - Chunk 2: tokens 180-359
  - Chunk 3: tokens 270-399 (padded or truncated)

  The overlap (stride < max_length) ensures context isn't lost at boundaries.
  """

  @type doc_id :: term()
  @type chunk_id :: String.t()
  @type chunk_mapping :: %{chunk_id() => %{doc_id: doc_id(), chunk_index: non_neg_integer()}}

  @default_max_length 180
  @default_stride 90

  @doc """
  Splits documents into overlapping chunks.

  ## Arguments
    * `documents` - List of {doc_id, text} tuples
    * `opts` - Chunking options

  ## Options
    * `:max_length` - Maximum tokens per chunk (default: #{@default_max_length})
    * `:stride` - Stride between chunks in tokens (default: #{@default_stride})
    * `:tokenizer` - Optional tokenizer function (default: simple word split)

  ## Returns
    Tuple of `{chunks, mapping}` where:
    - `chunks` is a list of {chunk_id, text} tuples
    - `mapping` is a map from chunk_id to original doc info
  """
  @spec chunk_documents([{doc_id(), String.t()}], keyword()) ::
          {[{chunk_id(), String.t()}], chunk_mapping()}
  def chunk_documents(documents, opts \\ []) do
    max_length = Keyword.get(opts, :max_length, @default_max_length)
    stride = Keyword.get(opts, :stride, @default_stride)
    tokenizer = Keyword.get(opts, :tokenizer, &default_tokenize/1)

    {chunks, mapping} =
      documents
      |> Enum.flat_map_reduce(%{}, fn {doc_id, text}, mapping ->
        doc_chunks = chunk_text(text, doc_id, max_length, stride, tokenizer)

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
    max_length = Keyword.get(opts, :max_length, @default_max_length)
    stride = Keyword.get(opts, :stride, @default_stride)
    tokenizer = Keyword.get(opts, :tokenizer, &default_tokenize/1)

    tokens = tokenizer.(text)
    num_tokens = length(tokens)

    if num_tokens <= max_length do
      [text]
    else
      chunk_tokens(tokens, max_length, stride)
      |> Enum.map(&Enum.join(&1, " "))
    end
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
    max_length = Keyword.get(opts, :max_length, @default_max_length)
    stride = Keyword.get(opts, :stride, @default_stride)
    tokenizer = Keyword.get(opts, :tokenizer, &default_tokenize/1)

    num_tokens = text |> tokenizer.() |> length()

    if num_tokens <= max_length do
      1
    else
      # Calculate number of chunks with overlap
      div(num_tokens - max_length, stride) + 2
    end
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

  # Internal implementation

  defp chunk_text(text, doc_id, max_length, stride, tokenizer) do
    tokens = tokenizer.(text)
    num_tokens = length(tokens)

    if num_tokens <= max_length do
      chunk_id = generate_chunk_id(doc_id, 0)
      [{chunk_id, text}]
    else
      chunk_tokens(tokens, max_length, stride)
      |> Enum.with_index()
      |> Enum.map(fn {chunk_tokens, idx} ->
        chunk_id = generate_chunk_id(doc_id, idx)
        chunk_text = Enum.join(chunk_tokens, " ")
        {chunk_id, chunk_text}
      end)
    end
  end

  defp chunk_tokens(tokens, max_length, stride) do
    num_tokens = length(tokens)

    Stream.unfold(0, fn start ->
      if start >= num_tokens do
        nil
      else
        chunk = Enum.slice(tokens, start, max_length)
        next_start = start + stride

        if length(chunk) > 0 do
          {chunk, next_start}
        else
          nil
        end
      end
    end)
    |> Enum.to_list()
  end

  defp generate_chunk_id(doc_id, chunk_index) do
    "#{doc_id}__chunk_#{chunk_index}"
  end

  defp default_tokenize(text) do
    text
    |> String.split(~r/\s+/, trim: true)
  end
end
