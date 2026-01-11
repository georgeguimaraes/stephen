defmodule StephenColbert.Encoder do
  @moduledoc """
  Encodes text into per-token embeddings using BERT.

  ColBERT uses per-token embeddings rather than pooled [CLS] embeddings,
  enabling fine-grained late interaction scoring.
  """

  @type encoder :: %{
          model: Axon.t(),
          params: map(),
          tokenizer: Bumblebee.Tokenizer.t(),
          embedding_dim: pos_integer()
        }

  @type embeddings :: Nx.Tensor.t()

  @default_model "sentence-transformers/all-MiniLM-L6-v2"
  @query_marker "[Q]"
  @doc_marker "[D]"

  @doc """
  Loads a BERT model for encoding.

  ## Options
    * `:model` - HuggingFace model name (default: #{@default_model})
    * `:backend` - Nx backend to use (default: EXLA if available)

  ## Examples

      {:ok, encoder} = StephenColbert.Encoder.load()
      {:ok, encoder} = StephenColbert.Encoder.load(model: "bert-base-uncased")
  """
  @spec load(keyword()) :: {:ok, encoder()} | {:error, term()}
  def load(opts \\ []) do
    model_name = Keyword.get(opts, :model, @default_model)

    with {:ok, %{model: model, params: params, spec: spec}} <-
           Bumblebee.load_model({:hf, model_name}),
         {:ok, tokenizer} <- Bumblebee.load_tokenizer({:hf, model_name}) do
      embedding_dim = spec.hidden_size

      {:ok,
       %{
         model: model,
         params: params,
         tokenizer: tokenizer,
         embedding_dim: embedding_dim
       }}
    end
  end

  @doc """
  Encodes a query into per-token embeddings.

  Prepends the query marker [Q] before encoding.
  Returns normalized embeddings with shape {sequence_length, embedding_dim}.
  """
  @spec encode_query(encoder(), String.t()) :: embeddings()
  def encode_query(encoder, text) do
    encode(encoder, @query_marker <> " " <> text)
  end

  @doc """
  Encodes a document into per-token embeddings.

  Prepends the document marker [D] before encoding.
  Returns normalized embeddings with shape {sequence_length, embedding_dim}.
  """
  @spec encode_document(encoder(), String.t()) :: embeddings()
  def encode_document(encoder, text) do
    encode(encoder, @doc_marker <> " " <> text)
  end

  @doc """
  Encodes multiple documents in batch.

  Returns a list of normalized embeddings, one per document.
  """
  @spec encode_documents(encoder(), [String.t()]) :: [embeddings()]
  def encode_documents(encoder, texts) do
    Enum.map(texts, &encode_document(encoder, &1))
  end

  @doc """
  Encodes text into per-token embeddings without markers.

  Returns normalized embeddings with shape {sequence_length, embedding_dim}.
  """
  @spec encode(encoder(), String.t()) :: embeddings()
  def encode(encoder, text) do
    %{model: model, params: params, tokenizer: tokenizer} = encoder

    inputs = Bumblebee.apply_tokenizer(tokenizer, text)

    %{hidden_state: hidden_state} = Axon.predict(model, params, inputs)

    # Get the sequence embeddings (remove batch dimension if present)
    embeddings =
      case Nx.shape(hidden_state) do
        {1, seq_len, dim} -> Nx.reshape(hidden_state, {seq_len, dim})
        {_seq_len, _dim} -> hidden_state
        _ -> hidden_state
      end

    # L2 normalize each token embedding
    normalize(embeddings)
  end

  @doc """
  Returns the embedding dimension for the loaded model.
  """
  @spec embedding_dim(encoder()) :: pos_integer()
  def embedding_dim(encoder), do: encoder.embedding_dim

  # L2 normalize embeddings along the last axis
  defp normalize(embeddings) do
    norm = Nx.LinAlg.norm(embeddings, axes: [-1], keep_axes: true)
    # Add small epsilon to avoid division by zero
    Nx.divide(embeddings, Nx.add(norm, 1.0e-9))
  end
end
