#!/usr/bin/env python3
"""
Generate reference outputs from Python ColBERT for validation.

This script produces JSON output that can be used to validate
the Stephen Elixir implementation against the reference Python ColBERT.

Usage:
    python scripts/colbert_reference.py > test/fixtures/colbert_reference.json
"""
import json
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
    checkpoint = Checkpoint(config.checkpoint, colbert_config=config)

    # Test cases - same texts we'll use in Elixir tests
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

    print("Encoding queries...", file=sys.stderr)
    Q = checkpoint.queryFromText(queries, bsize=32)
    # Q shape: [num_queries, query_maxlen, dim]

    print("Encoding documents...", file=sys.stderr)
    # docFromText returns a tuple: (tensor, lengths) or just (tensor,)
    D_result = checkpoint.docFromText(docs, bsize=32)
    D_tensor = D_result[0] if isinstance(D_result, tuple) else D_result
    # D_tensor shape: [num_docs, doc_seq_len, dim]

    # Compute MaxSim scores for all query-doc pairs
    print("Computing MaxSim scores...", file=sys.stderr)
    maxsim_scores = []
    num_queries = Q.shape[0]
    num_docs = D_tensor.shape[0]

    for i in range(num_queries):
        q = Q[i]  # [query_maxlen, dim]
        query_scores = []
        for j in range(num_docs):
            d = D_tensor[j]  # [doc_seq_len, dim]
            # MaxSim: for each query token, find max similarity with any doc token, then sum
            similarity_matrix = torch.matmul(q, d.transpose(0, 1))  # [query_tokens, doc_tokens]
            max_sims = similarity_matrix.max(dim=1).values  # [query_tokens]
            score = max_sims.sum().item()
            query_scores.append({
                "query_idx": i,
                "doc_idx": j,
                "score": round(score, 6)
            })
        maxsim_scores.append(query_scores)

    # Get tokenization info for punctuation filtering validation
    print("Getting tokenization info...", file=sys.stderr)

    # Get skiplist info (punctuation tokens that ColBERT filters out)
    skiplist = checkpoint.skiplist if hasattr(checkpoint, 'skiplist') else set()

    # Get token IDs for common punctuation using the doc tokenizer
    tokenizer = checkpoint.doc_tokenizer.tok
    punct_tokens = {}
    for p in ['.', ',', '!', '?', ';', ':', "'", '"', '(', ')', '-', '[PAD]', '[CLS]', '[SEP]', '[MASK]']:
        try:
            token_id = tokenizer.convert_tokens_to_ids(p)
            punct_tokens[p] = {
                "id": token_id,
                "in_skiplist": token_id in skiplist
            }
        except Exception as e:
            punct_tokens[p] = {"id": None, "error": str(e)}

    # Test tokenization with punctuation
    test_text = "Hello, world! This is a test. How are you?"
    test_encoding = tokenizer(test_text, return_tensors='pt')
    test_token_ids = test_encoding['input_ids'][0].tolist()
    test_tokens = tokenizer.convert_ids_to_tokens(test_token_ids)

    results = {
        "config": {
            "doc_maxlen": config.doc_maxlen,
            "query_maxlen": config.query_maxlen,
            "dim": config.dim,
            "checkpoint": config.checkpoint
        },
        "queries": queries,
        "docs": docs,
        "query_embeddings": {
            "shape": list(Q.shape),
            "dtype": str(Q.dtype),
            # Store first query's first 5 token embeddings (first 5 dims each)
            "sample_values": [[round(v, 6) for v in row] for row in Q[0, :5, :5].tolist()]
        },
        "doc_embeddings": {
            "shape": list(D_tensor.shape),
            "sample_values": [[round(v, 6) for v in row] for row in D_tensor[0, :5, :5].tolist()]
        },
        "maxsim_scores": maxsim_scores,
        "tokenization": {
            "test_text": test_text,
            "test_tokens": test_tokens,
            "test_token_ids": test_token_ids,
            "punct_tokens": punct_tokens,
            "skiplist_size": len(skiplist),
            "skiplist_sample": list(skiplist)[:20] if skiplist else []
        }
    }

    print(json.dumps(results, indent=2))
    print("Done!", file=sys.stderr)

if __name__ == "__main__":
    main()
