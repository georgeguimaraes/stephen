# Benchmark Stephen ColBERT encoding performance
#
# Run with: mix run bench/encoding_bench.exs
#
# For EXLA acceleration:
#   EXLA_TARGET=cuda mix run bench/encoding_bench.exs
#   EXLA_TARGET=rocm mix run bench/encoding_bench.exs
#   EXLA_TARGET=tpu mix run bench/encoding_bench.exs

IO.puts("Loading encoder...")
{load_time, {:ok, encoder}} = :timer.tc(fn -> Stephen.Encoder.load() end)
IO.puts("Encoder loaded in #{Float.round(load_time / 1_000_000, 2)}s")

queries = [
  "what is machine learning?",
  "how does ColBERT work?",
  "neural information retrieval"
]

docs = [
  "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
  "ColBERT uses late interaction with per-token embeddings for efficient neural retrieval.",
  "Information retrieval systems help users find relevant documents in large collections.",
  "Deep learning models have revolutionized natural language processing tasks."
]

# Pre-compute embeddings for MaxSim benchmark
query_embs = Enum.map(queries, &Stephen.Encoder.encode_query(encoder, &1))
doc_embs = Enum.map(docs, &Stephen.Encoder.encode_document(encoder, &1))

IO.puts("\nRunning benchmarks...\n")

Benchee.run(
  %{
    "single_query" => fn ->
      Stephen.Encoder.encode_query(encoder, hd(queries))
    end,
    "single_doc" => fn ->
      Stephen.Encoder.encode_document(encoder, hd(docs))
    end,
    "query_batch_3" => fn ->
      Enum.map(queries, &Stephen.Encoder.encode_query(encoder, &1))
    end,
    "doc_batch_4" => fn ->
      Enum.map(docs, &Stephen.Encoder.encode_document(encoder, &1))
    end,
    "maxsim_3x4" => fn ->
      for q <- query_embs, d <- doc_embs do
        Stephen.Scorer.max_sim(q, d)
      end
    end
  },
  warmup: 2,
  time: 10,
  memory_time: 2,
  formatters: [
    Benchee.Formatters.Console,
    {Benchee.Formatters.JSON, file: "bench/results/stephen_bench.json"}
  ]
)

IO.puts("\nResults saved to bench/results/stephen_bench.json")
