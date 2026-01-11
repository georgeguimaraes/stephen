# Compare Stephen vs Python ColBERT benchmarks
#
# Usage:
#   1. Run Python benchmark first:
#      python scripts/benchmark_python.py > bench/results/python_bench.json
#
#   2. Run Elixir benchmark:
#      mix run bench/encoding_bench.exs
#
#   3. Compare results:
#      mix run scripts/compare_benchmarks.exs

defmodule BenchCompare do
  def run do
    python_path = "bench/results/python_bench.json"
    elixir_path = "bench/results/stephen_bench.json"

    python_results = load_json(python_path)
    elixir_results = load_json(elixir_path)

    if is_nil(python_results) or is_nil(elixir_results) do
      IO.puts("""
      Missing benchmark files. Run both benchmarks first:

        python scripts/benchmark_python.py > bench/results/python_bench.json
        mix run bench/encoding_bench.exs
      """)

      System.halt(1)
    end

    IO.puts("\n=== ColBERT Benchmark Comparison ===\n")

    IO.puts("Python: #{python_results["platform"]} on #{python_results["device"]}")
    IO.puts("Elixir: Stephen with #{Nx.default_backend()}")
    IO.puts("Model: #{python_results["model"]}\n")

    compare_metric(
      "Single Query Encoding",
      python_results["benchmarks"]["single_query"]["median_ms"],
      get_elixir_median(elixir_results, "single_query")
    )

    compare_metric(
      "Single Doc Encoding",
      python_results["benchmarks"]["single_doc"]["median_ms"],
      get_elixir_median(elixir_results, "single_doc")
    )

    compare_metric(
      "Query Batch (3)",
      python_results["benchmarks"]["query_batch_3"]["median_ms"],
      get_elixir_median(elixir_results, "query_batch_3")
    )

    compare_metric(
      "Doc Batch (4)",
      python_results["benchmarks"]["doc_batch_4"]["median_ms"],
      get_elixir_median(elixir_results, "doc_batch_4")
    )

    compare_metric(
      "MaxSim 3x4",
      python_results["benchmarks"]["maxsim_3x4"]["median_ms"],
      get_elixir_median(elixir_results, "maxsim_3x4")
    )

    IO.puts("\nLoad times:")
    IO.puts("  Python: #{python_results["load_time_s"]}s")
    # Elixir load time is printed during benchmark run
  end

  defp load_json(path) do
    case File.read(path) do
      {:ok, content} -> Jason.decode!(content)
      {:error, _} -> nil
    end
  end

  defp get_elixir_median(results, scenario_name) do
    scenario = Enum.find(results, fn s -> s["name"] == scenario_name end)

    if scenario do
      # Benchee stores median in microseconds, convert to ms
      scenario["statistics"]["median"] / 1000
    end
  end

  defp compare_metric(name, python_ms, elixir_ms)
       when is_number(python_ms) and is_number(elixir_ms) do
    ratio = python_ms / elixir_ms
    faster = if ratio > 1, do: "Elixir", else: "Python"
    ratio_display = if ratio > 1, do: ratio, else: 1 / ratio

    IO.puts("#{name}:")
    IO.puts("  Python: #{Float.round(python_ms, 2)}ms")
    IO.puts("  Elixir: #{Float.round(elixir_ms, 2)}ms")
    IO.puts("  #{faster} is #{Float.round(ratio_display, 2)}x faster\n")
  end

  defp compare_metric(name, python_ms, nil) do
    IO.puts("#{name}: Python #{Float.round(python_ms, 2)}ms (Elixir data missing)\n")
  end

  defp compare_metric(name, nil, elixir_ms) do
    IO.puts("#{name}: Elixir #{Float.round(elixir_ms, 2)}ms (Python data missing)\n")
  end
end

BenchCompare.run()
