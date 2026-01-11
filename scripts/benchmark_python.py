#!/usr/bin/env python3
"""
Benchmark Python ColBERT encoding performance.

Usage:
    python scripts/benchmark_python.py
"""
import json
import time
import sys

def main():
    try:
        import torch
        from colbert.modeling.checkpoint import Checkpoint
        from colbert.infra import ColBERTConfig
    except ImportError:
        print(json.dumps({
            "error": "colbert-ai not installed. Run: pip install colbert-ai torch"
        }))
        sys.exit(1)

    # Configure ColBERT to match Stephen's defaults
    config = ColBERTConfig(
        doc_maxlen=180,
        query_maxlen=32,
        dim=128,
        checkpoint='colbert-ir/colbertv2.0'
    )

    print("Loading ColBERT checkpoint...", file=sys.stderr)
    load_start = time.perf_counter()
    checkpoint = Checkpoint(config.checkpoint, colbert_config=config)
    load_time = time.perf_counter() - load_start

    # Test data
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

    # Warm up
    print("Warming up...", file=sys.stderr)
    checkpoint.queryFromText(queries[:1], bsize=32)
    checkpoint.docFromText(docs[:1], bsize=32)

    # Benchmark query encoding
    print("Benchmarking query encoding...", file=sys.stderr)
    query_times = []
    for _ in range(10):
        start = time.perf_counter()
        Q = checkpoint.queryFromText(queries, bsize=32)
        query_times.append(time.perf_counter() - start)

    # Benchmark document encoding
    print("Benchmarking document encoding...", file=sys.stderr)
    doc_times = []
    for _ in range(10):
        start = time.perf_counter()
        D = checkpoint.docFromText(docs, bsize=32)
        doc_times.append(time.perf_counter() - start)

    # Benchmark single query encoding
    print("Benchmarking single query encoding...", file=sys.stderr)
    single_query_times = []
    for _ in range(20):
        start = time.perf_counter()
        checkpoint.queryFromText([queries[0]], bsize=1)
        single_query_times.append(time.perf_counter() - start)

    # Benchmark single doc encoding
    print("Benchmarking single document encoding...", file=sys.stderr)
    single_doc_times = []
    for _ in range(20):
        start = time.perf_counter()
        checkpoint.docFromText([docs[0]], bsize=1)
        single_doc_times.append(time.perf_counter() - start)

    # Benchmark MaxSim scoring
    print("Benchmarking MaxSim scoring...", file=sys.stderr)
    Q = checkpoint.queryFromText(queries, bsize=32)
    D_result = checkpoint.docFromText(docs, bsize=32)
    D = D_result[0] if isinstance(D_result, tuple) else D_result

    maxsim_times = []
    for _ in range(100):
        start = time.perf_counter()
        for i in range(len(queries)):
            for j in range(len(docs)):
                q = Q[i]
                d = D[j]
                sim = torch.matmul(q, d.transpose(0, 1))
                max_sims = sim.max(dim=1).values
                score = max_sims.sum().item()
        maxsim_times.append(time.perf_counter() - start)

    def stats(times):
        times = sorted(times)
        return {
            "min_ms": round(min(times) * 1000, 2),
            "max_ms": round(max(times) * 1000, 2),
            "mean_ms": round(sum(times) / len(times) * 1000, 2),
            "median_ms": round(times[len(times) // 2] * 1000, 2),
            "p95_ms": round(times[int(len(times) * 0.95)] * 1000, 2),
        }

    results = {
        "platform": "Python ColBERT",
        "device": str(next(checkpoint.bert.parameters()).device),
        "model": config.checkpoint,
        "load_time_s": round(load_time, 2),
        "benchmarks": {
            "query_batch_3": {
                "n_queries": 3,
                "iterations": 10,
                **stats(query_times)
            },
            "doc_batch_4": {
                "n_docs": 4,
                "iterations": 10,
                **stats(doc_times)
            },
            "single_query": {
                "iterations": 20,
                **stats(single_query_times)
            },
            "single_doc": {
                "iterations": 20,
                **stats(single_doc_times)
            },
            "maxsim_3x4": {
                "description": "3 queries x 4 docs MaxSim scoring",
                "iterations": 100,
                **stats(maxsim_times)
            }
        }
    }

    print(json.dumps(results, indent=2))
    print("Done!", file=sys.stderr)

if __name__ == "__main__":
    main()
