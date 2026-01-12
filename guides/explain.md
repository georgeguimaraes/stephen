# Debugging Scores

Stephen provides tools to understand why documents scored the way they did. This is useful for debugging retrieval quality and understanding the MaxSim scoring mechanism.

## The explain Function

`Stephen.explain/3` shows which query tokens matched which document tokens:

```elixir
{:ok, encoder} = Stephen.load_encoder()

explanation = Stephen.explain(
  encoder,
  "satirical comedy",
  "Colbert is a satirical talk show host"
)

# Pretty print
explanation |> Stephen.Scorer.format_explanation() |> IO.puts()
```

Output:

```
Score: 15.20

Query Token          -> Doc Token            Similarity
------------------------------------------------------------
satirical            -> satirical            0.9512
comedy               -> show                 0.4823
##ical               -> satirical            0.8234
...
```

## Understanding the Output

Each row shows:

- **Query Token**: A token from your query
- **Doc Token**: The document token it matched best with
- **Similarity**: Cosine similarity (contribution to final score)

The total score is the sum of all similarities.

## What to Look For

### Good matches

High similarity (>0.8) between semantically related tokens:

```
satirical            -> satirical            0.95   # exact match
comedy               -> humor                0.82   # semantic match
```

### Weak matches

Low similarity indicates poor alignment:

```
python               -> javascript           0.31   # different concepts
```

### Unexpected matches

Sometimes tokens match unexpectedly:

```
comedy               -> Colbert              0.45   # name matched "comedy"?
```

This can reveal why irrelevant documents scored high.

## Filtering the Output

The `format_explanation/2` function accepts options:

```elixir
# Show only top 5 matches by similarity
Stephen.Scorer.format_explanation(explanation, top_k: 5)

# Skip special tokens ([CLS], [SEP], [MASK], etc.)
Stephen.Scorer.format_explanation(explanation, skip_special: true)  # default

# Show all tokens including special ones
Stephen.Scorer.format_explanation(explanation, skip_special: false)

# Only show matches above a threshold
Stephen.Scorer.format_explanation(explanation, min_similarity: 0.5)
```

## Working with Raw Explanation Data

The explanation map contains structured data:

```elixir
%{
  score: 15.2,
  matches: [
    %{
      query_token: "satirical",
      query_index: 0,
      doc_token: "satirical",
      doc_index: 3,
      similarity: 0.9512
    },
    # ...
  ]
}
```

Use this for programmatic analysis:

```elixir
# Find the weakest match
weakest = Enum.min_by(explanation.matches, & &1.similarity)

# Find matches below threshold
weak_matches = Enum.filter(explanation.matches, & &1.similarity < 0.5)

# Average similarity
avg = Enum.sum(Enum.map(explanation.matches, & &1.similarity)) / length(explanation.matches)
```

## Similarity Matrix

For deeper analysis, compute the full similarity matrix:

```elixir
query_emb = Stephen.Encoder.encode_query(encoder, "satirical comedy")
doc_emb = Stephen.Encoder.encode_document(encoder, "Colbert is satirical")

# Full similarity matrix: {query_len, doc_len}
sim_matrix = Stephen.Scorer.similarity_matrix(query_emb, doc_emb)

# Each cell [i, j] is cosine similarity between query token i and doc token j
```

## Score Normalization

Raw MaxSim scores depend on query length. Normalize for comparison:

```elixir
results = Stephen.search(encoder, index, "short query")

# Normalize to [0, 1] based on query length
normalized = Stephen.Scorer.normalize_results(results, 32)

# Or normalize within result set (highest = 1.0, lowest = 0.0)
normalized = Stephen.Scorer.normalize_minmax(results)
```

## Debugging Common Issues

### Document scores too low

Check if tokens are matching at all:

```elixir
explanation = Stephen.explain(encoder, query, doc_text)
IO.inspect(Enum.map(explanation.matches, & &1.similarity))
# All low? Query and doc may be semantically unrelated
```

### Wrong document ranks highest

Compare explanations:

```elixir
exp1 = Stephen.explain(encoder, query, good_doc)
exp2 = Stephen.explain(encoder, query, bad_doc)

IO.puts("Good doc score: #{exp1.score}")
IO.puts("Bad doc score: #{exp2.score}")

# Look for unexpected high-similarity matches in bad_doc
```

### Query too short

Short queries have fewer tokens to match:

```elixir
# "AI" might only have 2-3 tokens after encoding
exp = Stephen.explain(encoder, "AI", doc)
IO.puts("Query tokens: #{length(exp.matches)}")
# Consider using longer, more specific queries
```

## Comparing Documents

Score multiple documents against the same query:

```elixir
query = "political satire"
docs = [
  {"colbert", "Stephen Colbert does political comedy"},
  {"conan", "Conan O'Brien does absurdist humor"},
  {"seth", "Seth Meyers covers political news"}
]

for {id, text} <- docs do
  exp = Stephen.explain(encoder, query, text)
  IO.puts("#{id}: #{Float.round(exp.score, 2)}")
end
```
