defmodule Stephen.Encoder do
  @moduledoc """
  Encodes text into per-token embeddings using BERT.

  ColBERT uses per-token embeddings rather than pooled [CLS] embeddings,
  enabling fine-grained late interaction scoring.

  ## Features

  - Query padding with [MASK] tokens for query augmentation
  - Configurable max lengths for queries and documents
  - Batch encoding for efficient processing
  - Optional linear projection to reduce embedding dimension
  """

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
          skiplist: MapSet.t()
        }

  @type embeddings :: Nx.Tensor.t()

  @default_model "colbert-ir/colbertv2.0"
  @default_max_query_length 32
  @default_max_doc_length 180
  @default_projection_dim 128
  @query_marker "[Q]"
  @doc_marker "[D]"

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
    * `:projection_dim` - Output dimension after projection (default: #{@default_projection_dim}, nil to disable)
    * `:base_module` - Override the Bumblebee module for ColBERT models (auto-detected from config.json)

  ## ColBERT Models

  When loading a ColBERT model (e.g., `colbert-ir/colbertv2.0`), the trained projection
  weights are automatically loaded from the model's SafeTensors file. The base model
  type is auto-detected from config.json, but can be overridden with `:base_module`.

  Supported base models: BERT, RoBERTa, DistilBERT, ALBERT, XLM-RoBERTa.

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

      # Get the mask token ID from tokenizer
      mask_token_id = get_mask_token_id(tokenizer)

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

      # Get the mask token ID from tokenizer
      mask_token_id = get_mask_token_id(tokenizer)

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
         skiplist: skiplist
       }}
    end
  end

  # Load ColBERT projection weights from SafeTensors file
  defp load_colbert_projection(model_name) do
    # Use Bumblebee's HuggingFace Hub module for proper caching
    url = Bumblebee.HuggingFace.Hub.file_url(model_name, "model.safetensors", nil)

    with {:ok, path} <- Bumblebee.HuggingFace.Hub.cached_download(url, cache_scope: model_name) do
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
    url = Bumblebee.HuggingFace.Hub.file_url(model_name, "config.json", nil)

    with {:ok, path} <- Bumblebee.HuggingFace.Hub.cached_download(url, cache_scope: model_name),
         {:ok, content} <- File.read(path),
         {:ok, config} <- Jason.decode(content) do
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

  Prepends the query marker [Q] and pads with [MASK] tokens to max_query_length.
  Returns normalized embeddings with shape {max_query_length, output_dim}.

  ## Options
    * `:pad` - Whether to pad with [MASK] tokens (default: true)
  """
  @spec encode_query(encoder(), String.t(), keyword()) :: embeddings()
  def encode_query(encoder, text, opts \\ []) do
    pad = Keyword.get(opts, :pad, true)
    marked_text = @query_marker <> " " <> text

    embeddings = encode_single(encoder, marked_text, encoder.max_query_length)

    if pad do
      pad_with_mask_embeddings(encoder, embeddings, encoder.max_query_length)
    else
      embeddings
    end
  end

  @doc """
  Encodes multiple queries in batch.

  Returns a list of normalized embeddings, one per query.
  """
  @spec encode_queries(encoder(), [String.t()], keyword()) :: [embeddings()]
  def encode_queries(encoder, texts, opts \\ []) do
    pad = Keyword.get(opts, :pad, true)
    marked_texts = Enum.map(texts, &(@query_marker <> " " <> &1))

    embeddings_list = encode_batch(encoder, marked_texts, encoder.max_query_length)

    if pad do
      Enum.map(embeddings_list, &pad_with_mask_embeddings(encoder, &1, encoder.max_query_length))
    else
      embeddings_list
    end
  end

  @doc """
  Encodes a document into per-token embeddings.

  Prepends the document marker [D] before encoding.
  Returns normalized embeddings with shape {sequence_length, output_dim}.

  ## Options
    * `:skip_punctuation` - Whether to filter out punctuation token embeddings (default: false)
    * `:deduplicate` - Whether to remove duplicate token embeddings (default: false)
  """
  @spec encode_document(encoder(), String.t(), keyword()) :: embeddings()
  def encode_document(encoder, text, opts \\ []) do
    skip_punct = Keyword.get(opts, :skip_punctuation, false)
    dedup = Keyword.get(opts, :deduplicate, false)
    marked_text = @doc_marker <> " " <> text

    {embeddings, token_ids} =
      encode_single_with_ids(encoder, marked_text, encoder.max_doc_length)

    embeddings
    |> maybe_filter_punctuation(token_ids, encoder.skiplist, skip_punct)
    |> maybe_deduplicate(dedup)
  end

  @doc """
  Encodes multiple documents in batch.

  Uses true batched inference for efficiency.
  Returns a list of normalized embeddings, one per document.

  ## Options
    * `:skip_punctuation` - Whether to filter out punctuation token embeddings (default: false)
    * `:deduplicate` - Whether to remove duplicate token embeddings (default: false)
  """
  @spec encode_documents(encoder(), [String.t()], keyword()) :: [embeddings()]
  def encode_documents(encoder, texts, opts \\ []) do
    skip_punct = Keyword.get(opts, :skip_punctuation, false)
    dedup = Keyword.get(opts, :deduplicate, false)
    marked_texts = Enum.map(texts, &(@doc_marker <> " " <> &1))

    encode_batch_with_ids(encoder, marked_texts, encoder.max_doc_length)
    |> Enum.map(fn {embeddings, token_ids} ->
      embeddings
      |> maybe_filter_punctuation(token_ids, encoder.skiplist, skip_punct)
      |> maybe_deduplicate(dedup)
    end)
  end

  @doc """
  Encodes text into per-token embeddings without markers.

  Returns normalized embeddings with shape {sequence_length, output_dim}.
  """
  @spec encode(encoder(), String.t()) :: embeddings()
  def encode(encoder, text) do
    encode_single(encoder, text, encoder.max_doc_length)
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

  # Encode a single text
  defp encode_single(encoder, text, max_length) do
    %{model: model, params: params, tokenizer: tokenizer, projection: projection} = encoder

    # Configure tokenizer with max length
    configured_tokenizer =
      Bumblebee.configure(tokenizer, length: max_length, pad_direction: :right)

    inputs = Bumblebee.apply_tokenizer(configured_tokenizer, text)

    %{hidden_state: hidden_state} = Axon.predict(model, params, inputs)

    # Get attention mask to identify real tokens vs padding
    attention_mask = inputs["attention_mask"]

    # Get the sequence embeddings (remove batch dimension if present)
    embeddings =
      case Nx.shape(hidden_state) do
        {1, seq_len, dim} -> Nx.reshape(hidden_state, {seq_len, dim})
        {_seq_len, _dim} -> hidden_state
        _ -> hidden_state
      end

    # Apply projection if configured
    embeddings =
      if projection do
        apply_projection(embeddings, projection)
      else
        embeddings
      end

    # L2 normalize each token embedding
    embeddings = normalize(embeddings)

    # Mask out padding tokens (keep only real tokens based on attention mask)
    mask =
      case Nx.shape(attention_mask) do
        {1, len} -> Nx.reshape(attention_mask, {len})
        _ -> attention_mask
      end

    # Get indices of real tokens (where mask == 1)
    num_real_tokens = Nx.sum(mask) |> Nx.to_number() |> trunc()

    # Return only the real token embeddings
    Nx.slice(embeddings, [0, 0], [num_real_tokens, encoder.output_dim])
  end

  # Encode multiple texts in batch
  defp encode_batch(encoder, texts, max_length) do
    %{model: model, params: params, tokenizer: tokenizer, projection: projection} = encoder

    # Configure tokenizer with max length
    configured_tokenizer =
      Bumblebee.configure(tokenizer, length: max_length, pad_direction: :right)

    inputs = Bumblebee.apply_tokenizer(configured_tokenizer, texts)

    %{hidden_state: hidden_state} = Axon.predict(model, params, inputs)

    # hidden_state shape: {batch_size, seq_len, hidden_dim}
    attention_mask = inputs["attention_mask"]

    # Apply projection if configured
    hidden_state =
      if projection do
        apply_projection_batched(hidden_state, projection)
      else
        hidden_state
      end

    # Normalize all embeddings
    hidden_state = normalize_batched(hidden_state)

    # Split batch into individual sequences and remove padding
    batch_size = Nx.axis_size(hidden_state, 0)

    for i <- 0..(batch_size - 1) do
      embeddings = hidden_state[i]
      mask = attention_mask[i]

      # Count real tokens
      num_real_tokens = Nx.sum(mask) |> Nx.to_number() |> trunc()

      # Return only real token embeddings
      Nx.slice(embeddings, [0, 0], [num_real_tokens, encoder.output_dim])
    end
  end

  # Encode a single text and return both embeddings and token IDs
  defp encode_single_with_ids(encoder, text, max_length) do
    %{model: model, params: params, tokenizer: tokenizer, projection: projection} = encoder

    configured_tokenizer =
      Bumblebee.configure(tokenizer, length: max_length, pad_direction: :right)

    inputs = Bumblebee.apply_tokenizer(configured_tokenizer, text)

    %{hidden_state: hidden_state} = Axon.predict(model, params, inputs)

    attention_mask = inputs["attention_mask"]
    input_ids = inputs["input_ids"]

    embeddings =
      case Nx.shape(hidden_state) do
        {1, seq_len, dim} -> Nx.reshape(hidden_state, {seq_len, dim})
        {_seq_len, _dim} -> hidden_state
        _ -> hidden_state
      end

    embeddings =
      if projection do
        apply_projection(embeddings, projection)
      else
        embeddings
      end

    embeddings = normalize(embeddings)

    mask =
      case Nx.shape(attention_mask) do
        {1, len} -> Nx.reshape(attention_mask, {len})
        _ -> attention_mask
      end

    ids =
      case Nx.shape(input_ids) do
        {1, len} -> Nx.reshape(input_ids, {len})
        _ -> input_ids
      end

    num_real_tokens = Nx.sum(mask) |> Nx.to_number() |> trunc()

    real_embeddings = Nx.slice(embeddings, [0, 0], [num_real_tokens, encoder.output_dim])
    real_ids = Nx.slice(ids, [0], [num_real_tokens]) |> Nx.to_flat_list()

    {real_embeddings, real_ids}
  end

  # Encode multiple texts and return embeddings with token IDs
  defp encode_batch_with_ids(encoder, texts, max_length) do
    %{model: model, params: params, tokenizer: tokenizer, projection: projection} = encoder

    configured_tokenizer =
      Bumblebee.configure(tokenizer, length: max_length, pad_direction: :right)

    inputs = Bumblebee.apply_tokenizer(configured_tokenizer, texts)

    %{hidden_state: hidden_state} = Axon.predict(model, params, inputs)

    attention_mask = inputs["attention_mask"]
    input_ids = inputs["input_ids"]

    hidden_state =
      if projection do
        apply_projection_batched(hidden_state, projection)
      else
        hidden_state
      end

    hidden_state = normalize_batched(hidden_state)

    batch_size = Nx.axis_size(hidden_state, 0)

    for i <- 0..(batch_size - 1) do
      embeddings = hidden_state[i]
      mask = attention_mask[i]
      ids = input_ids[i]

      num_real_tokens = Nx.sum(mask) |> Nx.to_number() |> trunc()

      real_embeddings = Nx.slice(embeddings, [0, 0], [num_real_tokens, encoder.output_dim])
      real_ids = Nx.slice(ids, [0], [num_real_tokens]) |> Nx.to_flat_list()

      {real_embeddings, real_ids}
    end
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
    Enum.reduce(0..(n - 1), [], fn i, acc ->
      # Check if this embedding is too similar to any kept embedding
      is_duplicate =
        Enum.any?(acc, fn kept_idx ->
          sim = similarity[[i, kept_idx]] |> Nx.to_number()
          sim > threshold
        end)

      if is_duplicate, do: acc, else: acc ++ [i]
    end)
  end

  # Pad embeddings with [MASK] token embeddings to reach target length
  defp pad_with_mask_embeddings(encoder, embeddings, target_length) do
    {current_length, dim} = Nx.shape(embeddings)
    padding_needed = target_length - current_length

    if padding_needed <= 0 do
      embeddings
    else
      # Create [MASK] embedding by encoding a mask token
      # We use a cached approach - encode once and reuse
      mask_embedding = get_mask_embedding(encoder)

      # Repeat mask embedding for padding
      padding = Nx.broadcast(mask_embedding, {padding_needed, dim})

      # Concatenate original embeddings with padding
      Nx.concatenate([embeddings, padding], axis: 0)
    end
  end

  # Get the embedding for a [MASK] token
  defp get_mask_embedding(encoder) do
    %{model: model, params: params, tokenizer: tokenizer, projection: projection} = encoder

    # Encode just [MASK] to get its embedding
    inputs = Bumblebee.apply_tokenizer(tokenizer, "[MASK]")
    %{hidden_state: hidden_state} = Axon.predict(model, params, inputs)

    # Get the [MASK] token embedding (skip [CLS] at position 0, [MASK] is at position 1)
    mask_emb =
      case Nx.shape(hidden_state) do
        {1, _seq_len, _dim} -> hidden_state[[0, 1, ..]]
        _ -> hidden_state[1]
      end

    # Apply projection if configured
    mask_emb =
      if projection do
        Nx.dot(mask_emb, projection)
      else
        mask_emb
      end

    # Normalize
    norm = Nx.LinAlg.norm(mask_emb)
    Nx.divide(mask_emb, Nx.add(norm, 1.0e-9))
  end

  # Get mask token ID from tokenizer
  defp get_mask_token_id(tokenizer) do
    native_tokenizer = get_native_tokenizer(tokenizer)

    # Get the mask token ID (returns nil or integer)
    case Tokenizers.Tokenizer.token_to_id(native_tokenizer, "[MASK]") do
      id when is_integer(id) -> id
      _ -> 103
    end
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

  # Apply linear projection
  defnp apply_projection(embeddings, projection) do
    Nx.dot(embeddings, projection)
  end

  # Apply linear projection to batched embeddings
  defnp apply_projection_batched(hidden_state, projection) do
    # hidden_state: {batch, seq, hidden_dim}
    # projection: {hidden_dim, output_dim}
    # result: {batch, seq, output_dim}
    Nx.dot(hidden_state, [2], projection, [0])
  end

  # L2 normalize embeddings along the last axis
  defnp normalize(embeddings) do
    norm = Nx.LinAlg.norm(embeddings, axes: [-1], keep_axes: true)
    Nx.divide(embeddings, Nx.add(norm, 1.0e-9))
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
