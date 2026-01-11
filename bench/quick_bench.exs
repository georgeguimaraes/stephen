# Quick benchmark: Stephen vs Python comparison
#
# Run with: mix run bench/quick_bench.exs

IO.puts("Loading encoder...")
{load_time_us, {:ok, encoder}} = :timer.tc(fn -> Stephen.Encoder.load() end)
load_time_s = Float.round(load_time_us / 1_000_000, 2)
IO.puts("Encoder loaded in #{load_time_s}s")

queries = [
  "who is Stephen Colbert?",
  "best late night comedy hosts",
  "political satire on television"
]

docs = [
  "Stephen Colbert hosted The Colbert Report before taking over The Late Show from David Letterman.",
  "Conan O'Brien is known for his self-deprecating humor, red hair, and Conan Without Borders specials.",
  "Seth Meyers was head writer at SNL and now hosts Late Night with his A Closer Look segments.",
  "John Oliver hosts Last Week Tonight on HBO with in-depth investigative comedy journalism."
]

# Warm up
IO.puts("\nWarming up...")
Stephen.Encoder.encode_query(encoder, hd(queries))
Stephen.Encoder.encode_document(encoder, hd(docs))

# Helper to run timed iterations
run_bench = fn name, iterations, fun ->
  times =
    for _ <- 1..iterations do
      {time_us, _} = :timer.tc(fun)
      # convert to ms
      time_us / 1000
    end

  sorted = Enum.sort(times)
  median = Enum.at(sorted, div(iterations, 2))
  min = Enum.min(times)
  max = Enum.max(times)
  mean = Enum.sum(times) / iterations

  IO.puts(
    "#{name}: median=#{Float.round(median, 2)}ms min=#{Float.round(min, 2)}ms max=#{Float.round(max, 2)}ms"
  )

  median
end

IO.puts("\nRunning benchmarks (10 iterations each)...\n")

single_query =
  run_bench.("single_query", 10, fn ->
    Stephen.Encoder.encode_query(encoder, hd(queries))
  end)

single_doc =
  run_bench.("single_doc", 10, fn ->
    Stephen.Encoder.encode_document(encoder, hd(docs))
  end)

query_batch =
  run_bench.("query_batch_3", 10, fn ->
    Enum.map(queries, &Stephen.Encoder.encode_query(encoder, &1))
  end)

doc_batch =
  run_bench.("doc_batch_4", 10, fn ->
    Enum.map(docs, &Stephen.Encoder.encode_document(encoder, &1))
  end)

# Pre-compute for MaxSim
query_embs = Enum.map(queries, &Stephen.Encoder.encode_query(encoder, &1))
doc_embs = Enum.map(docs, &Stephen.Encoder.encode_document(encoder, &1))

maxsim =
  run_bench.("maxsim_3x4", 50, fn ->
    for q <- query_embs, d <- doc_embs do
      Stephen.Scorer.max_sim(q, d)
    end
  end)

# Compare with Python results
IO.puts("\n=== Comparison with Python ColBERT ===\n")

python_results = %{
  "single_query" => 13.83,
  "single_doc" => 13.56,
  "query_batch_3" => 21.29,
  "doc_batch_4" => 20.73,
  "maxsim_3x4" => 0.43,
  "load_time_s" => 2.65
}

compare = fn name, elixir_ms, python_ms ->
  ratio = python_ms / elixir_ms
  {faster, ratio_display} = if ratio > 1, do: {"Elixir", ratio}, else: {"Python", 1 / ratio}

  IO.puts(
    "#{name}: Elixir #{Float.round(elixir_ms, 2)}ms vs Python #{python_ms}ms (#{faster} #{Float.round(ratio_display, 2)}x faster)"
  )
end

compare.("single_query", single_query, python_results["single_query"])
compare.("single_doc", single_doc, python_results["single_doc"])
compare.("query_batch_3", query_batch, python_results["query_batch_3"])
compare.("doc_batch_4", doc_batch, python_results["doc_batch_4"])
compare.("maxsim_3x4", maxsim, python_results["maxsim_3x4"])

IO.puts("\nLoad time: Elixir #{load_time_s}s vs Python #{python_results["load_time_s"]}s")
