defmodule Stephen.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/georgeguimaraes/stephen"

  def project do
    [
      app: :stephen,
      version: @version,
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      docs: docs(),
      name: "Stephen",
      description: "ColBERT-style neural retrieval for Elixir",
      package: package(),
      source_url: @source_url
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:bumblebee, "~> 0.6"},
      {:nx, "~> 0.9"},
      {:axon, "~> 0.7"},
      {:scholar, "~> 0.4"},
      {:exla, "~> 0.9", optional: true},
      {:hnswlib, "~> 0.1"},
      {:ex_doc, "~> 0.34", only: :dev, runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:benchee, "~> 1.3", only: :dev},
      {:benchee_json, "~> 1.0", only: :dev}
    ]
  end

  defp docs do
    [
      main: "readme",
      extras: [
        "README.md",
        "livebook/getting_started.livemd",
        "guides/architecture.md",
        "guides/index_types.md",
        "guides/compression.md",
        "guides/chunking.md",
        "guides/configuration.md"
      ],
      groups_for_extras: [
        Livebooks: Path.wildcard("livebook/*.livemd"),
        Guides: Path.wildcard("guides/*.md")
      ],
      groups_for_modules: [
        Core: [
          Stephen,
          Stephen.Encoder,
          Stephen.Scorer,
          Stephen.Retriever
        ],
        Indexes: [
          Stephen.Index,
          Stephen.Plaid,
          Stephen.Index.Compressed
        ],
        Utilities: [
          Stephen.Compression,
          Stephen.Chunker,
          Stephen.KMeans
        ]
      ],
      source_url: @source_url,
      source_ref: "v#{@version}"
    ]
  end

  defp package do
    [
      maintainers: ["George Guimarães"],
      licenses: ["MIT"],
      links: %{"GitHub" => @source_url}
    ]
  end
end
