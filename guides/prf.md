# Query Expansion (PRF)

Pseudo-relevance feedback (PRF) expands your query with terms from top-ranked documents, improving recall for ambiguous or underspecified queries.

## How PRF Works

1. Run initial search with the original query
2. Take top-k documents as "feedback" (assumed relevant)
3. Extract representative token embeddings from feedback docs
4. Combine original query with expansion embeddings
5. Re-run search with the expanded query

This helps find documents that are relevant but don't match the exact query terms.

## Basic Usage

```elixir
{:ok, encoder} = Stephen.load_encoder()
index = Stephen.load_index("my_index")

# Search with PRF
results = Stephen.search_with_prf(encoder, index, "late night hosts")
```

## When PRF Helps

PRF is most useful when:

- **Short queries**: "comedy" expands to include "humor", "jokes", "satirical"
- **Ambiguous terms**: "apple" in a tech corpus expands toward technology terms
- **Vocabulary mismatch**: query says "car", documents say "automobile"

Example:

```elixir
# Without PRF: might miss documents about "automobiles"
Stephen.search(encoder, index, "car safety")

# With PRF: top results about "cars" help find "automobile" docs too
Stephen.search_with_prf(encoder, index, "car safety")
```

## Tuning Parameters

```elixir
results = Stephen.search_with_prf(encoder, index, query,
  feedback_docs: 5,        # docs to use for expansion (default: 3)
  expansion_tokens: 15,    # tokens to add from feedback (default: 10)
  expansion_weight: 0.3    # weight vs original query (default: 0.5)
)
```

### feedback_docs

Number of top documents to use as feedback. More docs = more diverse expansion, but may introduce noise.

- `3` (default): Conservative, high precision
- `5-10`: Broader expansion, better recall
- `>10`: Risky, may drift from original intent

### expansion_tokens

Number of token embeddings to add from feedback documents. These are selected to be relevant but not redundant with the original query.

- `10` (default): Modest expansion
- `15-20`: More aggressive expansion
- `5`: Minimal expansion, mostly preserves original query

### expansion_weight

How much weight expansion tokens get relative to original query tokens.

- `0.5` (default): Equal influence
- `0.3`: Original query dominates
- `0.7`: Heavy expansion influence (use cautiously)

## How Tokens Are Selected

Not all tokens from feedback docs are useful. Stephen selects expansion tokens by scoring:

```
expansion_score = relevance * novelty
```

Where:
- **relevance** = max similarity to any query token (must be somewhat related)
- **novelty** = 1 - relevance (shouldn't duplicate query terms)

This favors tokens that are related to the query but add new information.

## Example: Understanding Expansion

```elixir
# Original query
query = "late night comedy"

# PRF might effectively expand this to include terms like:
# "talk show", "host", "satirical", "monologue", "interview"
# from the top-ranked documents

results = Stephen.search_with_prf(encoder, index, query,
  feedback_docs: 3,
  expansion_tokens: 10
)
```

## Comparison: With and Without PRF

```elixir
# Standard search
standard = Stephen.search(encoder, index, "jazz musicians")

# PRF search
expanded = Stephen.search_with_prf(encoder, index, "jazz musicians")

# PRF might surface docs about:
# - "bebop artists" (related term from feedback)
# - "saxophone players" (specific instrument)
# - "Miles Davis" (specific person mentioned in feedback)
```

## Performance Considerations

PRF runs two searches:
1. Initial search for feedback docs
2. Final search with expanded query

This roughly doubles search time. Consider:

- Caching encoder between searches (already done internally)
- Using smaller `feedback_docs` for faster expansion
- Only using PRF for queries that need it (short, ambiguous)

## When to Avoid PRF

- **Precise queries**: "Stephen Colbert Late Show" doesn't need expansion
- **Real-time search**: Added latency may not be acceptable
- **High-precision needs**: PRF may introduce false positives

## Advanced: Custom Expansion

For more control, access the lower-level functions:

```elixir
alias Stephen.{Encoder, Retriever}

# Encode query
query_emb = Encoder.encode_query(encoder, query)

# Get feedback docs
feedback = Retriever.search_with_embeddings(query_emb, index, top_k: 3)

# Extract expansion embeddings
expansion_emb = Retriever.extract_expansion_embeddings(
  index, feedback, query_emb, 10
)

# expansion_emb is now a tensor you can inspect or modify
```
