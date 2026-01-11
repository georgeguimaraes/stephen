# Stephen - ColBERT for Elixir (Phase 2)

Enhance the existing Stephen library to match Python ColBERT features.

## Current State

The library has basic ColBERT functionality:
- Per-token embeddings with [Q]/[D] markers
- MaxSim late interaction scoring
- HNSWLib ANN search
- Index persistence

## Features to Implement

### 1. Query Padding with [MASK] Tokens (High Priority)
- Pad all queries to a fixed length (default: 32 tokens)
- Use [MASK] tokens for padding after the query text
- This enables query augmentation - the model learns to use [MASK] for soft matching
- Update `Encoder.encode_query/2` to support this

### 2. Batch Encoding (High Priority)
- Implement true batched inference in `Encoder.encode_documents/2`
- Process multiple texts in a single forward pass instead of sequential map
- Add `Encoder.encode_queries/2` for batch query encoding
- Use Bumblebee's batching capabilities

### 3. Configurable Max Lengths (High Priority)
- Add `:max_query_length` option (default: 32)
- Add `:max_doc_length` option (default: 180)
- Truncate inputs that exceed these lengths
- Store config in encoder struct

### 4. Linear Projection Layer (Medium Priority)
- Add optional linear projection to reduce embedding dimension
- Default projection: hidden_size -> 128 dimensions
- Smaller embeddings = smaller index = faster search
- Make projection configurable via `:projection_dim` option
- Initialize projection weights (can be random or loaded)

### 5. Residual Compression for ColBERTv2 (Medium Priority)
- Implement centroid-based compression in `Stephen.Compression` module
- Train K centroids on token embeddings (K=2^16 typical)
- Store centroid ID (2 bytes) + residual (quantized) per token
- Add `Stephen.Index.Compressed` for compressed index variant
- Achieves ~6x compression ratio

### 6. PLAID-style Indexing (Medium Priority)
- Implement centroid-based candidate generation
- For each query token, find nearest centroids
- Use centroid inverted lists to get candidate docs
- Then score candidates with full MaxSim
- Faster than pure ANN for large collections

### 7. Token Deduplication (Low Priority)
- Skip duplicate token embeddings within a document
- Reduces index size without quality loss
- Add `:deduplicate` option to indexing

### 8. Passage Chunking (Low Priority)
- Add `Stephen.Chunker` module for splitting long documents
- Support overlapping chunks (e.g., stride of 90 tokens)
- Track chunk-to-document mapping in index

## Implementation Order

1. First: Query padding + max lengths (encoder changes)
2. Second: Batch encoding (performance)
3. Third: Linear projection (index size)
4. Fourth: Compression module (ColBERTv2)
5. Fifth: PLAID indexing (search speed)
6. Sixth: Deduplication + chunking (polish)

## Requirements

- All existing tests must continue to pass
- Add new tests for each feature
- Update documentation
- Maintain backward compatibility (new features opt-in)

## Success Criteria

- mix compile succeeds
- mix test passes (including new tests)
- Query padding works with [MASK] tokens
- Batch encoding processes multiple docs in one forward pass
- Linear projection reduces embedding dimension
- All features are configurable and documented

Output <promise>PHASE 2 COMPLETE</promise> when all high and medium priority features are implemented and tests pass.
