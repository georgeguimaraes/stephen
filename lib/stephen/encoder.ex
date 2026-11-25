defmodule Stephen.Encoder do
  @moduledoc """
  Encodes text into per-token embeddings using BERT.

  ColBERT uses per-token embeddings rather than pooled [CLS] embeddings,
  enabling fine-grained late interaction scoring.

  ## Features

  - Query and document markers, as in reference ColBERT
  - Query augmentation: padding becomes [MASK] tokens that the model encodes in context
  - Configurable max lengths for queries and documents
  - Batch encoding for efficient processing
  - Optional linear projection to reduce embedding dimension
  """

  alias Bumblebee.HuggingFace.Hub

  import Nx.Defn

  @type encoder :: %{
          model: Axon.t(),
          params: map(),
          tokenizer: Tokenizers.Tokenizer.t(),
          embedding_dim: pos_integer(),
          output_dim: pos_integer(),
          max_query_length: pos_integer(),
          max_doc_length: pos_integer(),
          projection: Nx.Tensor.t() | nil,
          mask_token_id: non_neg_integer(),
          query_marker_id: non_neg_integer(),
          doc_marker_id: non_neg_integer(),
          skiplist: MapSet.t()
        }

  @type embeddings :: Nx.Tensor.t()

  @default_model "colbert-ir/colbertv2.0"
  @default_max_query_length 32
  @default_max_doc_length 180
  @default_projection_dim 128
  # Reference ColBERT puts these vocabulary tokens after [CLS] to mark queries and documents
  @query_marker "[unused0]"
  @doc_marker "[unused1]"

  # Reference ColBERT drops punctuation from documents (mask_punctuation: true), not from queries
  @skip_punctuation_default true

  # Known ColBERT models with trained projection weights
  @colbert_models [
    "colbert-ir/colbertv2.0",
    "colbert-ir/colbertv1.0"
  ]

  # Mapping from HuggingFace model_type to Bumblebee module
  @model_type_to_module %{
    "bert" => Bumblebee.Text.Bert,
    "roberta" => Bumblebee.Text.Roberta,
    "distilbert" => Bumblebee.Text.Distilbert,
    "albert" => Bumblebee.Text.Albert,
    "xlm-roberta" => Bumblebee.Text.Roberta
  }

  # Punctuation tokens to skip (matching Python ColBERT's skiplist)
  @punctuation_tokens [
    ".",
    ",",
    "!",
    "?",
    ";",
    ":",
    "'",
    "\"",
    "(",
    ")",
    "-",
    "#",
    "$",
    "%",
    "&",
    "*",
    "+",
    "/",
    "<",
    "=",
    ">",
    "@",
    "[",
    "\\",
    "]",
    "^",
    "_",
    "`",
    "{",
    "|",
    "}",
    "~"
  ]

  @doc """
  Loads a BERT model for encoding.

  ## Options
    * `:model` - HuggingFace model name (default: #{@default_model})
    * `:max_query_length` - Maximum query length in tokens (default: #{@default_max_query_length})
    * `:max_doc_length` - Maximum document length in tokens (default: #{@default_max_doc_length})
    * `:projection_dim` - Output dimension after projection (default: #{@default_projection_dim}, nil to disable).
      ColBERT models ignore it and use their trained projection.
    * `:base_module` - Override the Bumblebee module for ColBERT models (auto-detected from config.json)

  ## ColBERT Models

  When loading a ColBERT model (e.g., `colbert-ir/colbertv2.0`), the trained projection
  weights are automatically loaded from the model's SafeTensors file. The base model
  type is auto-detected from config.json, but can be overridden with `:base_module`.

  Supported base models (auto-detected): BERT, RoBERTa, DistilBERT, ALBERT, XLM-RoBERTa.
  Note: Only BERT has been tested with official ColBERT weights. Other architectures
  should work if the model provides compatible weights.

  ## Examples

      {:ok, encoder} = Stephen.Encoder.load()
      {:ok, encoder} = Stephen.Encoder.load(model: "bert-base-uncased", projection_dim: 128)

      # Load official ColBERT model with trained weights
      {:ok, encoder} = Stephen.Encoder.load(model: "colbert-ir/colbertv2.0")

      # Override base model type for custom ColBERT models
      {:ok, encoder} = Stephen.Encoder.load(
        model: "custom/roberta-colbert",
        base_module: Bumblebee.Text.Roberta
      )
  """
  @spec load(keyword()) :: {:ok, encoder()} | {:error, term()}
  def load(opts \\ []) do
    model_name = Keyword.get(opts, :model, @default_model)
    max_query_length = Keyword.get(opts, :max_query_length, @default_max_query_length)
    max_doc_length = Keyword.get(opts, :max_doc_length, @default_max_doc_length)
    projection_dim = Keyword.get(opts, :projection_dim, @default_projection_dim)
    base_module = Keyword.get(opts, :base_module)

    if colbert_model?(model_name) do
      load_colbert(model_name, max_query_length, max_doc_length, base_module)
    else
      load_standard(model_name, max_query_length, max_doc_length, projection_dim)
    end
  end

  # Check if model is a known ColBERT model
  defp colbert_model?(model_name) do
    model_name in @colbert_models
  end

  # Load a standard BERT model with random projection
  defp load_standard(model_name, max_query_length, max_doc_length, projection_dim) do
    with {:ok, %{model: model, params: params, spec: spec}} <-
           Bumblebee.load_model({:hf, model_name}),
         {:ok, tokenizer} <- Bumblebee.load_tokenizer({:hf, model_name}) do
      embedding_dim = spec.hidden_size

      mask_token_id = Bumblebee.Tokenizer.special_token_id(tokenizer, :mask)

      # Build skiplist of punctuation token IDs
      skiplist = build_skiplist(tokenizer)

      # Initialize projection matrix if projection_dim is set
      {projection, output_dim} =
        if projection_dim && projection_dim < embedding_dim do
          # Xavier/Glorot initialization for the projection matrix
          key = Nx.Random.key(42)
          stddev = :math.sqrt(2.0 / (embedding_dim + projection_dim))

          # Nx.Random.normal returns {tensor, new_key}
          {proj, _new_key} =
            Nx.Random.normal(key, shape: {embedding_dim, projection_dim}, type: :f32)

          # Scale by stddev
          proj = Nx.multiply(proj, stddev)

          {proj, projection_dim}
        else
          {nil, embedding_dim}
        end

      {:ok,
       %{
         model: model,
         params: params,
         tokenizer: tokenizer,
         embedding_dim: embedding_dim,
         output_dim: output_dim,
         max_query_length: max_query_length,
         max_doc_length: max_doc_length,
         projection: projection,
         mask_token_id: mask_token_id,
         query_marker_id: marker_id(tokenizer, @query_marker),
         doc_marker_id: marker_id(tokenizer, @doc_marker),
         skiplist: skiplist
       }}
    end
  end

  # Load a ColBERT model with trained projection weights
  defp load_colbert(model_name, max_query_length, max_doc_length, base_module_override) do
    # ColBERT models use HF_ColBERT architecture which Bumblebee doesn't support.
    # We detect the base model type from config.json and load it directly.

    with {:ok, base_module} <- resolve_base_module(model_name, base_module_override),
         {:ok, %{model: model, params: params, spec: spec}} <-
           Bumblebee.load_model({:hf, model_name},
             module: base_module,
             architecture: :base
           ),
         {:ok, tokenizer} <- Bumblebee.load_tokenizer({:hf, model_name}),
         {:ok, projection} <- load_colbert_projection(model_name) do
      embedding_dim = spec.hidden_size
      output_dim = Nx.axis_size(projection, 1)

      mask_token_id = Bumblebee.Tokenizer.special_token_id(tokenizer, :mask)

      # Build skiplist of punctuation token IDs
      skiplist = build_skiplist(tokenizer)

      {:ok,
       %{
         model: model,
         params: params,
         tokenizer: tokenizer,
         embedding_dim: embedding_dim,
         output_dim: output_dim,
         max_query_length: max_query_length,
         max_doc_length: max_doc_length,
         projection: projection,
         mask_token_id: mask_token_id,
         query_marker_id: marker_id(tokenizer, @query_marker),
         doc_marker_id: marker_id(tokenizer, @doc_marker),
         skiplist: skiplist
       }}
    end
  end

  # Load ColBERT projection weights from SafeTensors file
  defp load_colbert_projection(model_name) do
    # Use Bumblebee's HuggingFace Hub module for proper caching
    url = Hub.file_url(model_name, "model.safetensors", nil)

    with {:ok, path} <- Hub.cached_download(url, cache_scope: model_name) do
      tensors = Safetensors.read!(path)

      # ColBERT stores projection as "linear.weight" with shape {output_dim, input_dim}
      # We need to transpose it to {input_dim, output_dim} for our dot product
      case Map.get(tensors, "linear.weight") do
        nil ->
          {:error, "No linear.weight found in ColBERT model"}

        weight ->
          # Transpose from {128, 768} to {768, 128}
          {:ok, Nx.transpose(weight)}
      end
    end
  rescue
    e -> {:error, "Failed to read SafeTensors: #{Exception.message(e)}"}
  end

  # Resolve the Bumblebee module for a ColBERT model
  # Uses override if provided, otherwise auto-detects from config.json
  defp resolve_base_module(_model_name, module) when not is_nil(module), do: {:ok, module}

  defp resolve_base_module(model_name, nil) do
    url = Hub.file_url(model_name, "config.json", nil)

    with {:ok, path} <- Hub.cached_download(url, cache_scope: model_name),
         {:ok, content} <- File.read(path),
         {:ok, config} <- JSON.decode(content) do
      model_type = Map.get(config, "model_type", "bert")

      case Map.get(@model_type_to_module, model_type) do
        nil ->
          {:error,
           "Unsupported model_type '#{model_type}'. Use :base_module option to specify manually."}

        module ->
          {:ok, module}
      end
    end
  end

  @doc """
  Encodes a query into per-token embeddings.

  Follows reference ColBERT: the query marker goes after [CLS], and the padding up to
  `max_query_length` becomes [MASK] tokens. The model encodes the [MASK] tokens with
  the query, so they add query-specific embeddings (query augmentation).
  Returns normalized embeddings with shape {max_query_length, output_dim}.

  ## Options
    * `:pad` - Whether to add the [MASK] embeddings (default: true). Without them,
      only the real query tokens are returned.
  """
  @spec encode_query(encoder(), String.t(), keyword()) :: embeddings()
  def encode_query(encoder, text, opts \\ []) do
    [embeddings] = encode_queries(encoder, [text], opts)
    embeddings
  end

  @doc """
  Encodes multiple queries in batch.

  Returns a list of normalized embeddings, one per query. Takes the same options as
  `encode_query/3`.
  """
  @spec encode_queries(encoder(), [String.t()], keyword()) :: [embeddings()]
  def encode_queries(encoder, texts, opts \\ []) do
    keep = if Keyword.get(opts, :pad, true), do: :all, else: :real

    encoder
    |> encode_texts(texts, encoder.max_query_length, encoder.query_marker_id, keep)
    |> Enum.map(fn {embeddings, _token_ids} -> embeddings end)
  end

  @doc """
  Encodes a document into per-token embeddings.

  Puts the document marker after [CLS], as reference ColBERT does.
  Returns normalized embeddings with shape {sequence_length, output_dim}.

  ## Options
    * `:skip_punctuation?` - Whether to drop punctuation token embeddings, as reference
      ColBERT does (default: true)
    * `:deduplicate?` - Whether to remove duplicate token embeddings (default: false)
  """
  @spec encode_document(encoder(), String.t(), keyword()) :: embeddings()
  def encode_document(encoder, text, opts \\ []) do
    [embeddings] = encode_documents(encoder, [text], opts)
    embeddings
  end

  @doc """
  Encodes multiple documents in batch.

  Uses true batched inference for efficiency.
  Returns a list of normalized embeddings, one per document.

  ## Options
    * `:skip_punctuation?` - Whether to drop punctuation token embeddings, as reference
      ColBERT does (default: true)
    * `:deduplicate?` - Whether to remove duplicate token embeddings (default: false)
  """
  @spec encode_documents(encoder(), [String.t()], keyword()) :: [embeddings()]
  def encode_documents(encoder, texts, opts \\ []) do
    skip_punct? = Keyword.get(opts, :skip_punctuation?, @skip_punctuation_default)
    dedup? = Keyword.get(opts, :deduplicate?, false)

    encoder
    |> encode_texts(texts, encoder.max_doc_length, encoder.doc_marker_id, :real)
    |> Enum.map(fn {embeddings, token_ids} ->
      embeddings
      |> maybe_filter_punctuation(token_ids, encoder.skiplist, skip_punct?)
      |> maybe_deduplicate(dedup?)
    end)
  end

  @doc """
  Encodes text into per-token embeddings without markers.

  Returns normalized embeddings with shape {sequence_length, output_dim}.
  """
  @spec encode(encoder(), String.t()) :: embeddings()
  def encode(encoder, text) do
    [{embeddings, _token_ids}] = encode_texts(encoder, [text], encoder.max_doc_length, nil, :real)
    embeddings
  end

  @doc """
  Returns the output embedding dimension (after projection if enabled).
  """
  @spec embedding_dim(encoder()) :: pos_integer()
  def embedding_dim(encoder), do: encoder.output_dim

  @doc """
  Returns the raw model embedding dimension (before projection).
  """
  @spec hidden_dim(encoder()) :: pos_integer()
  def hidden_dim(encoder), do: encoder.embedding_dim

  @doc """
  Tokenizes text and returns the token strings.

  Useful for visualization and debugging. The tokens correspond to the
  embeddings returned by `encode_query/2` or `encode_document/2`.

  ## Options
    * `:type` - `:query` or `:document` (default: `:document`)
    * `:max_length` - Maximum tokens (defaults based on type)
    * `:skip_punctuation?` - For documents, whether to drop punctuation tokens, as
      `encode_document/3` does (default: true)

  ## Examples

      tokens = Stephen.Encoder.tokenize(encoder, "Stephen Colbert")
      # => ["[CLS]", "[unused1]", "stephen", "colbert", "[SEP]"]

      tokens = Stephen.Encoder.tokenize(encoder, "Conan", type: :query)
      # => ["[CLS]", "[unused0]", "conan", "[SEP]", "[MASK]", ..., "[MASK]"]
  """
  @spec tokenize(encoder(), String.t(), keyword()) :: [String.t()]
  def tokenize(encoder, text, opts \\ []) do
    {max_length, marker_id, keep, skip_punct?} =
      case Keyword.get(opts, :type, :document) do
        :query ->
          {Keyword.get(opts, :max_length, encoder.max_query_length), encoder.query_marker_id,
           :all, false}

        :document ->
          {Keyword.get(opts, :max_length, encoder.max_doc_length), encoder.doc_marker_id, :real,
           Keyword.get(opts, :skip_punctuation?, @skip_punctuation_default)}
      end

    inputs = model_inputs(encoder, [text], max_length, marker_id, keep)
    [length] = kept_lengths(inputs, keep)

    inputs["input_ids"][0]
    |> Nx.slice([0], [length])
    |> Nx.to_flat_list()
    |> Enum.reject(&(skip_punct? and MapSet.member?(encoder.skiplist, &1)))
    |> ids_to_tokens(encoder.tokenizer.native_tokenizer)
  end

  defp ids_to_tokens(ids, native_tokenizer) do
    Enum.map(ids, fn id ->
      case Tokenizers.Tokenizer.id_to_token(native_tokenizer, id) do
        token when is_binary(token) -> token
        _ -> "[UNK]"
      end
    end)
  end

  # Runs the model on a batch and returns {embeddings, token_ids} per text.
  # keep: :real drops the padding, :all keeps every position (the [MASK] padding of a query).
  defp encode_texts(encoder, texts, max_length, marker_id, keep) do
    %{model: model, params: params, projection: projection} = encoder

    inputs = model_inputs(encoder, texts, max_length, marker_id, keep)
    %{hidden_state: hidden_state} = Axon.predict(model, params, inputs)

    hidden_state =
      if projection, do: apply_projection_batched(hidden_state, projection), else: hidden_state

    hidden_state = normalize_batched(hidden_state)

    inputs
    |> kept_lengths(keep)
    |> Enum.with_index()
    |> Enum.map(fn {length, i} ->
      embeddings = Nx.slice(hidden_state[i], [0, 0], [length, encoder.output_dim])
      token_ids = inputs["input_ids"][i] |> Nx.slice([0], [length]) |> Nx.to_flat_list()
      {embeddings, token_ids}
    end)
  end

  # Tokenizes like reference ColBERT: ". " + text, then the marker replaces the "." token
  # after [CLS]. For keep: :all, the padding becomes [MASK] tokens. The attention mask
  # stays as the tokenizer made it, so real tokens do not attend to the [MASK] tokens.
  defp model_inputs(encoder, texts, max_length, marker_id, keep) do
    tokenizer = Bumblebee.configure(encoder.tokenizer, length: max_length, pad_direction: :right)
    texts = if marker_id, do: Enum.map(texts, &(". " <> &1)), else: texts

    tokenizer
    |> Bumblebee.apply_tokenizer(texts)
    |> put_marker(marker_id)
    |> augment(keep, encoder.mask_token_id)
  end

  defp put_marker(inputs, nil), do: inputs

  defp put_marker(%{"input_ids" => ids} = inputs, marker_id) do
    column = Nx.broadcast(Nx.tensor(marker_id, type: Nx.type(ids)), {Nx.axis_size(ids, 0), 1})
    %{inputs | "input_ids" => Nx.put_slice(ids, [0, 1], column)}
  end

  defp augment(inputs, :real, _mask_token_id), do: inputs

  defp augment(%{"input_ids" => ids, "attention_mask" => mask} = inputs, :all, mask_token_id) do
    %{inputs | "input_ids" => Nx.select(Nx.equal(mask, 0), mask_token_id, ids)}
  end

  defp kept_lengths(%{"attention_mask" => mask}, :real) do
    mask |> Nx.sum(axes: [1]) |> Nx.to_flat_list()
  end

  defp kept_lengths(%{"attention_mask" => mask}, :all) do
    {batch_size, length} = Nx.shape(mask)
    List.duplicate(length, batch_size)
  end

  # Filter out punctuation token embeddings
  defp maybe_filter_punctuation(embeddings, _token_ids, _skiplist, false), do: embeddings

  defp maybe_filter_punctuation(embeddings, token_ids, skiplist, true) do
    # Find indices of non-punctuation tokens
    keep_indices =
      token_ids
      |> Enum.with_index()
      |> Enum.reject(fn {id, _idx} -> MapSet.member?(skiplist, id) end)
      |> Enum.map(fn {_id, idx} -> idx end)

    if length(keep_indices) == length(token_ids) do
      # No punctuation to filter
      embeddings
    else
      # Gather embeddings at keep indices
      indices_tensor = Nx.tensor(keep_indices)
      Nx.take(embeddings, indices_tensor, axis: 0)
    end
  end

  # Remove duplicate embeddings based on cosine similarity
  defp maybe_deduplicate(embeddings, false), do: embeddings

  defp maybe_deduplicate(embeddings, true) do
    {n, _dim} = Nx.shape(embeddings)

    if n <= 1 do
      embeddings
    else
      # Compute pairwise cosine similarity (embeddings are already normalized)
      similarity = Nx.dot(embeddings, Nx.transpose(embeddings))

      # Keep track of which indices to keep
      # For each embedding, check if any previous embedding is too similar
      keep_indices = deduplicate_indices(similarity, n, 0.99)

      if length(keep_indices) == n do
        embeddings
      else
        indices_tensor = Nx.tensor(keep_indices)
        Nx.take(embeddings, indices_tensor, axis: 0)
      end
    end
  end

  # Find indices to keep after deduplication
  defp deduplicate_indices(similarity, n, threshold) do
    0..(n - 1)
    |> Enum.reduce([], fn i, acc ->
      # Check if this embedding is too similar to any kept embedding
      is_duplicate =
        Enum.any?(acc, fn kept_idx ->
          sim = similarity[[i, kept_idx]] |> Nx.to_number()
          sim > threshold
        end)

      if is_duplicate, do: acc, else: [i | acc]
    end)
    |> Enum.reverse()
  end

  # The vocabulary id of a marker token, or the unknown token for vocabularies without it
  defp marker_id(tokenizer, token) do
    Bumblebee.Tokenizer.token_to_id(tokenizer, token) ||
      Bumblebee.Tokenizer.special_token_id(tokenizer, :unk)
  end

  # Build skiplist of punctuation token IDs from tokenizer
  defp build_skiplist(tokenizer) do
    native_tokenizer = get_native_tokenizer(tokenizer)

    @punctuation_tokens
    |> Enum.reduce(MapSet.new(), fn token, acc ->
      case Tokenizers.Tokenizer.token_to_id(native_tokenizer, token) do
        id when is_integer(id) -> MapSet.put(acc, id)
        _ -> acc
      end
    end)
  end

  # Extract native tokenizer from Bumblebee wrapper
  defp get_native_tokenizer(tokenizer) do
    case tokenizer do
      %{native_tokenizer: native} -> native
      native -> native
    end
  end

  # Apply linear projection to batched embeddings
  defnp apply_projection_batched(hidden_state, projection) do
    # hidden_state: {batch, seq, hidden_dim}
    # projection: {hidden_dim, output_dim}
    # result: {batch, seq, output_dim}
    Nx.dot(hidden_state, [2], projection, [0])
  end

  # L2 normalize batched embeddings (3D tensor: {batch, seq, dim})
  defnp normalize_batched(embeddings) do
    # Manual L2 norm for 3D tensors: sqrt(sum(x^2))
    squared = Nx.pow(embeddings, 2)
    sum_squared = Nx.sum(squared, axes: [-1], keep_axes: true)
    norm = Nx.sqrt(sum_squared)
    Nx.divide(embeddings, Nx.add(norm, 1.0e-9))
  end
end
