# StephenColbert - ColBERT for Elixir

Build a ColBERT-style neural retrieval library for Elixir.

## Background

ColBERT (Contextualized Late Interaction over BERT) is a neural retrieval model that:
- Encodes queries and documents into per-token embeddings (not just [CLS])
- Uses "late interaction" - MaxSim scoring between query and doc token embeddings
- Enables efficient retrieval via ANN search over token embeddings

## Requirements

### Core Dependencies (mix.exs)
- bumblebee ~> 0.6 for BERT model loading
- nx ~> 0.9 for tensor operations
- axon ~> 0.7 for neural network
- exla ~> 0.9 as Nx backend (optional dep)
- hnswlib ~> 0.1 for approximate nearest neighbor search

### Modules to Build

1. **StephenColbert.Encoder**
   - Load BERT model via Bumblebee
   - Encode text to per-token embeddings (not pooled)
   - Add [Q] marker for queries, [D] marker for documents
   - Normalize embeddings

2. **StephenColbert.Scorer**
   - Implement MaxSim: for each query token, find max similarity to any doc token
   - Sum the max similarities for final score
   - Use Nx for efficient tensor operations

3. **StephenColbert.Index**
   - Store document embeddings with doc_id mapping
   - Build HNSWLib index over token embeddings
   - Support adding documents incrementally
   - Persist/load index to disk

4. **StephenColbert.Retriever**
   - Query the index: encode query, search ANN, aggregate by doc_id
   - Return top-k documents with scores
   - Support reranking with full MaxSim after ANN retrieval

5. **StephenColbert** (main module)
   - High-level API: index/search/rerank
   - Configuration options (model name, embedding dim, etc.)

### Tests
- Test encoding produces correct tensor shapes
- Test MaxSim scoring logic
- Test index add/search roundtrip
- Test full retrieval pipeline

### Code Style
- Use Elixir best practices
- Add @moduledoc and @doc
- Use typespecs

## Success Criteria
- mix compile succeeds
- mix test passes with meaningful tests
- Can index a few sample documents and retrieve them

Output <promise>LIBRARY COMPLETE</promise> when tests pass and the library is functional.
