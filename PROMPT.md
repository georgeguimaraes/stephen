# Stephen - ColBERT for Elixir (Phase 3)

Complete feature parity with Python ColBERT and validate correctness.

## Current State

The library has ColBERTv2 features:
- Per-token embeddings with [Q]/[D] markers
- Query padding with [MASK] tokens
- Batch encoding
- Linear projection (384 -> 128)
- MaxSim late interaction scoring
- Residual compression (Stephen.Compression)
- PLAID-style indexing (Stephen.Plaid)
- HNSWLib ANN search
- Index persistence (Stephen.Index)

## Features to Implement

### 1. PLAID Index Persistence (High Priority)
- Add `Stephen.Plaid.save/2` and `Stephen.Plaid.load/1`
- Save centroids, inverted index, and doc embeddings
- Use efficient binary format

### 2. Compressed Index Integration (High Priority)
- Create `Stephen.Index.Compressed` that combines compression with indexing
- Store compressed embeddings instead of full float32
- Decompress on-the-fly during scoring
- Should work with existing search API

### 3. Punctuation Filtering (Medium Priority)
- Skip embeddings for punctuation tokens: `.,!?;:'"()-`
- Add `:skip_punctuation` option to encoding (default: true)
- Reduces index size without quality loss
- Match Python ColBERT's `skiplist` behavior

### 4. Token Deduplication (Medium Priority)
- Skip duplicate token embeddings within a document
- Add `:deduplicate` option to `encode_document/3`
- Compare embeddings with cosine similarity threshold
- Reduces index size

### 5. Passage Chunking (Medium Priority)
- Add `Stephen.Chunker` module
- Split long documents into overlapping chunks
- Default: 180 tokens with 90 token stride
- Track chunk-to-document mapping
- Support `chunk_documents/2` and `merge_results/2`

### 6. Reranking API (Low Priority)
- Add `Stephen.Retriever.rerank/4`
- Takes query + candidate doc_ids, returns reranked results
- Useful for two-stage retrieval pipelines

## Python ColBERT Validation

### Setup Python Environment
```bash
# Create venv and install colbert
python3 -m venv .venv
source .venv/bin/activate
pip install colbert-ai torch
```

### Validation Tests to Create

Create `test/python_colbert_validation_test.exs` that:

1. **Encoding Validation**
   - Encode same text with both Python ColBERT and Stephen
   - Compare embedding shapes and rough similarity
   - Test query encoding with [MASK] padding
   - Test document encoding

2. **MaxSim Scoring Validation**
   - Compute MaxSim scores for same query-doc pairs
   - Scores should match within tolerance (< 0.01 difference)

3. **Punctuation Filtering Validation**
   - Verify same tokens are filtered
   - Compare resulting embedding counts

4. **Create Python Helper Script**
   - `scripts/colbert_reference.py` that outputs JSON
   - Encode queries and documents
   - Compute MaxSim scores
   - List filtered tokens

### Python Reference Script Template
```python
#!/usr/bin/env python3
"""Generate reference outputs from Python ColBERT for validation."""
import json
import torch
from colbert.modeling.checkpoint import Checkpoint
from colbert.infra import ColBERTConfig

def main():
    config = ColBERTConfig(
        doc_maxlen=180,
        query_maxlen=32,
        dim=128,
    )
    checkpoint = Checkpoint("colbert-ir/colbertv2.0", colbert_config=config)

    # Test cases
    queries = ["what is machine learning?", "how does ColBERT work?"]
    docs = ["Machine learning is a subset of artificial intelligence.",
            "ColBERT uses late interaction for efficient retrieval."]

    # Encode
    Q = checkpoint.queryFromText(queries)
    D = checkpoint.docFromText(docs)

    # Output shapes and sample values
    results = {
        "query_shape": list(Q.shape),
        "doc_shapes": [list(d.shape) for d in D],
        "query_sample": Q[0, 0, :5].tolist(),
        "doc_sample": D[0][0, :5].tolist(),
    }

    # MaxSim scores
    scores = []
    for q in Q:
        for d in D:
            score = (q @ d.T).max(dim=1).values.sum().item()
            scores.append(score)
    results["maxsim_scores"] = scores

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
```

## Implementation Order

1. First: Python reference script + validation test structure
2. Second: Punctuation filtering (easy to validate)
3. Third: Token deduplication
4. Fourth: PLAID persistence
5. Fifth: Compressed index integration
6. Sixth: Passage chunking
7. Seventh: Reranking API
8. Throughout: Run validation tests against Python

## Requirements

- All existing tests must continue to pass
- New features must have tests
- Python validation tests should pass (within tolerance)
- Maintain backward compatibility

## Success Criteria

- `mix test` passes (all tests including new ones)
- `mix test --only python_validation` passes
- Punctuation filtering matches Python ColBERT
- MaxSim scores within 0.01 of Python ColBERT
- All new modules have documentation

Output <promise>PHASE 3 COMPLETE</promise> when all features are implemented, tests pass, and Python validation confirms correctness.
