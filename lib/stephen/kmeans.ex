defmodule Stephen.KMeans do
  @moduledoc """
  K-means clustering wrapper around Scholar.Cluster.KMeans.

  Provides a simplified interface for centroid training with support for
  both L2 (default Scholar) and cosine distance use cases.
  """

  alias Scholar.Cluster.KMeans

  @default_iterations 10
  @default_seed 42

  @doc """
  Trains K centroids using K-means clustering.

  ## Arguments
    * `embeddings` - Tensor of shape {n, dim}
    * `k` - Number of centroids to train

  ## Options
    * `:iterations` - Number of K-means iterations (default: #{@default_iterations})
    * `:distance` - Distance metric: :cosine or :l2 (default: :cosine)
    * `:normalize` - Whether to L2-normalize centroids (default: true for cosine)
    * `:seed` - Random seed for initialization (default: #{@default_seed})

  ## Returns
    Tensor of shape {k, dim} containing trained centroids.
  """
  @spec train(Nx.Tensor.t(), pos_integer(), keyword()) :: Nx.Tensor.t()
  def train(embeddings, k, opts \\ []) do
    {n, _dim} = Nx.shape(embeddings)
    k = min(k, n)

    iterations = Keyword.get(opts, :iterations, @default_iterations)
    distance = Keyword.get(opts, :distance, :cosine)
    normalize = Keyword.get(opts, :normalize, distance == :cosine)
    seed = Keyword.get(opts, :seed, @default_seed)

    # Scholar uses L2 distance internally
    model =
      KMeans.fit(embeddings,
        num_clusters: k,
        max_iterations: iterations,
        num_runs: 1,
        init: :random,
        key: Nx.Random.key(seed)
      )

    centroids = model.clusters

    if normalize do
      normalize_centroids(centroids)
    else
      centroids
    end
  end

  @doc """
  Finds the nearest centroid for each embedding.

  ## Arguments
    * `embeddings` - Tensor of shape {n, dim}
    * `centroids` - Tensor of shape {k, dim}
    * `distance` - Distance metric: :cosine or :l2

  ## Returns
    Tensor of shape {n} with centroid indices.
  """
  @spec find_nearest(Nx.Tensor.t(), Nx.Tensor.t(), :cosine | :l2) :: Nx.Tensor.t()
  def find_nearest(embeddings, centroids, :cosine) do
    # Cosine similarity: dot product (assumes normalized embeddings)
    similarities = Nx.dot(embeddings, Nx.transpose(centroids))
    Nx.argmax(similarities, axis: 1)
  end

  def find_nearest(embeddings, centroids, :l2) do
    # Use Scholar's predict which uses L2 distance
    model = %KMeans{clusters: centroids}
    KMeans.predict(model, embeddings)
  end

  # L2 normalize each centroid
  defp normalize_centroids(centroids) do
    norms = Nx.LinAlg.norm(centroids, axes: [1], keep_axes: true)
    Nx.divide(centroids, Nx.max(norms, 1.0e-9))
  end
end
