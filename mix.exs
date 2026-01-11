defmodule Stephen.MixProject do
  use Mix.Project

  def project do
    [
      app: :stephen,
      version: "0.1.0",
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:bumblebee, "~> 0.6"},
      {:nx, "~> 0.9"},
      {:axon, "~> 0.7"},
      {:exla, "~> 0.9", optional: true},
      {:hnswlib, "~> 0.1"}
    ]
  end
end
