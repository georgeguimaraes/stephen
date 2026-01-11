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
import os
import warnings
import numpy as np

def main():
    # Suppress warnings from colbert/torch that pollute stdout
    warnings.filterwarnings("ignore")
    os.environ["COLBERT_LOAD_TORCH_EXTENSION_VERBOSE"] = "False"

    # Redirect stdout during imports to suppress colbert's print statements
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        import logging
        logging.getLogger().setLevel(logging.ERROR)

        import torch
        from colbert.modeling.checkpoint import Checkpoint
        from colbert.infra import ColBERTConfig
    except ImportError:
        sys.stdout = old_stdout
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

    # Restore stdout after all colbert initialization
    sys.stdout = old_stdout

    # Test cases - same texts we'll use in Elixir tests
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

    # Generate compression validation data
    print("Generating compression validation data...", file=sys.stderr)

    # Use first doc's embeddings for compression tests
    test_embeddings = D_tensor[0].cpu().numpy()  # shape: [seq_len, dim]

    # Generate bit packing test cases
    # 1-bit: pack 8 values per byte
    bit_pack_tests_1bit = []
    for test_vals in [[1,0,1,0,1,0,1,0], [1,1,1,1,1,1,1,1], [0,0,0,0,0,0,0,0], [1,1,0,0,1,1,0,0]]:
        packed = np.packbits(np.array(test_vals, dtype=np.uint8))
        bit_pack_tests_1bit.append({
            "input": test_vals,
            "packed": packed.tolist()
        })

    # 2-bit: pack 4 values per byte
    bit_pack_tests_2bit = []
    for test_vals in [[0,1,2,3], [3,3,3,3], [0,0,0,0], [1,2,1,2]]:
        # Pack 4 2-bit values into 1 byte: val0<<6 | val1<<4 | val2<<2 | val3
        packed = (test_vals[0] << 6) | (test_vals[1] << 4) | (test_vals[2] << 2) | test_vals[3]
        bit_pack_tests_2bit.append({
            "input": test_vals,
            "packed": [packed]
        })

    # Retrieval ranking validation
    print("Computing retrieval rankings...", file=sys.stderr)
    rankings = []
    for i in range(num_queries):
        query_scores = maxsim_scores[i]
        sorted_docs = sorted(query_scores, key=lambda x: x['score'], reverse=True)
        rankings.append({
            "query_idx": i,
            "query": queries[i],
            "ranking": [s['doc_idx'] for s in sorted_docs],
            "top_doc": docs[sorted_docs[0]['doc_idx']],
            "top_score": sorted_docs[0]['score']
        })

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
        "rankings": rankings,
        "compression_tests": {
            "bit_pack_1bit": bit_pack_tests_1bit,
            "bit_pack_2bit": bit_pack_tests_2bit,
            "compression_ratios": {
                "dim_128_1bit": round(128 * 4 / (2 + 16), 2),  # 28.44
                "dim_128_2bit": round(128 * 4 / (2 + 32), 2),  # 15.06
                "dim_128_4bit": round(128 * 4 / (2 + 64), 2),  # 7.76
                "dim_128_8bit": round(128 * 4 / (2 + 128), 2)  # 3.94
            }
        },
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
