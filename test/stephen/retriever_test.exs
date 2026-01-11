defmodule Stephen.RetrieverTest do
  use ExUnit.Case, async: true

  alias Stephen.{Index, Plaid, Retriever}

  @embedding_dim 4

  describe "search_with_embeddings/3 with Index" do
    test "searches using pre-computed embeddings" do
      index = Index.new(embedding_dim: @embedding_dim)

      # Add documents with distinct embeddings
      emb1 = Nx.tensor([[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]])
      emb2 = Nx.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0]])
      emb3 = Nx.tensor([[0.0, 0.0, 1.0, 0.0]])

      index =
        index
        |> Index.add("doc1", emb1)
        |> Index.add("doc2", emb2)
        |> Index.add("doc3", emb3)

      # Query similar to doc1
      query_emb = Nx.tensor([[0.95, 0.05, 0.0, 0.0]])
      results = Retriever.search_with_embeddings(query_emb, index, top_k: 2)

      assert length(results) == 2
      assert is_float(hd(results).score)
      # doc1 should be in results since query is similar
      doc_ids = Enum.map(results, & &1.doc_id)
      assert "doc1" in doc_ids
    end

    test "respects top_k option" do
      index = Index.new(embedding_dim: @embedding_dim)

      emb1 = Nx.tensor([[1.0, 0.0, 0.0, 0.0]])
      emb2 = Nx.tensor([[0.0, 1.0, 0.0, 0.0]])
      emb3 = Nx.tensor([[0.0, 0.0, 1.0, 0.0]])

      index =
        index
        |> Index.add("doc1", emb1)
        |> Index.add("doc2", emb2)
        |> Index.add("doc3", emb3)

      query_emb = Nx.tensor([[0.5, 0.5, 0.0, 0.0]])
      results = Retriever.search_with_embeddings(query_emb, index, top_k: 1)

      assert length(results) == 1
    end

    test "works without reranking" do
      index = Index.new(embedding_dim: @embedding_dim)

      emb1 = Nx.tensor([[1.0, 0.0, 0.0, 0.0]])
      emb2 = Nx.tensor([[0.0, 1.0, 0.0, 0.0]])

      index =
        index
        |> Index.add("doc1", emb1)
        |> Index.add("doc2", emb2)

      query_emb = Nx.tensor([[1.0, 0.0, 0.0, 0.0]])
      results = Retriever.search_with_embeddings(query_emb, index, top_k: 2, rerank: false)

      assert length(results) <= 2
      # Without rerank, score is based on token match count
    end
  end

  describe "search_with_embeddings/3 with Plaid" do
    test "searches using pre-computed embeddings" do
      # Create enough documents for centroid training
      documents =
        for i <- 1..20 do
          # Create embeddings pointing in different directions
          angle = i * :math.pi() / 10

          emb =
            Nx.tensor([
              [Float.round(:math.cos(angle), 4), Float.round(:math.sin(angle), 4), 0.0, 0.0]
            ])

          {"doc#{i}", emb}
        end

      plaid = Plaid.new(embedding_dim: @embedding_dim, num_centroids: 4)
      plaid = Plaid.index_documents(plaid, documents)

      # Query at a specific angle
      query_emb = Nx.tensor([[1.0, 0.0, 0.0, 0.0]])
      results = Retriever.search_with_embeddings(query_emb, plaid, top_k: 3)

      assert length(results) <= 3
      assert is_float(hd(results).score)
    end
  end

  describe "rerank_with_embeddings/4" do
    test "reranks documents with pre-computed embeddings" do
      index = Index.new(embedding_dim: @embedding_dim)

      emb1 = Nx.tensor([[1.0, 0.0, 0.0, 0.0]])
      emb2 = Nx.tensor([[0.0, 1.0, 0.0, 0.0]])
      emb3 = Nx.tensor([[0.5, 0.5, 0.0, 0.0]])

      index =
        index
        |> Index.add("doc1", emb1)
        |> Index.add("doc2", emb2)
        |> Index.add("doc3", emb3)

      query_emb = Nx.tensor([[1.0, 0.0, 0.0, 0.0]])
      results = Retriever.rerank_with_embeddings(query_emb, index, ["doc1", "doc2", "doc3"])

      assert length(results) == 3
      # doc1 should rank highest (perfect match)
      assert hd(results).doc_id == "doc1"
    end

    test "respects top_k option" do
      index = Index.new(embedding_dim: @embedding_dim)

      emb1 = Nx.tensor([[1.0, 0.0, 0.0, 0.0]])
      emb2 = Nx.tensor([[0.0, 1.0, 0.0, 0.0]])
      emb3 = Nx.tensor([[0.0, 0.0, 1.0, 0.0]])

      index =
        index
        |> Index.add("doc1", emb1)
        |> Index.add("doc2", emb2)
        |> Index.add("doc3", emb3)

      query_emb = Nx.tensor([[1.0, 0.0, 0.0, 0.0]])

      results =
        Retriever.rerank_with_embeddings(query_emb, index, ["doc1", "doc2", "doc3"], top_k: 1)

      assert length(results) == 1
      assert hd(results).doc_id == "doc1"
    end
  end

  describe "rerank_texts/4" do
    @tag :slow
    test "reranks raw text documents without an index" do
      {:ok, encoder} = Stephen.Encoder.load()

      documents = [
        {"doc1", "Elixir is a dynamic, functional programming language"},
        {"doc2", "Python is used for machine learning and data science"},
        {"doc3", "JavaScript runs in web browsers"}
      ]

      results = Retriever.rerank_texts(encoder, "functional programming language", documents)

      assert length(results) == 3
      # doc1 should rank highest (mentions functional programming language)
      assert hd(results).doc_id == "doc1"
      assert is_float(hd(results).score)
    end

    @tag :slow
    test "respects top_k option" do
      {:ok, encoder} = Stephen.Encoder.load()

      documents = [
        {"doc1", "The quick brown fox"},
        {"doc2", "Jumps over the lazy dog"},
        {"doc3", "Hello world example"}
      ]

      results = Retriever.rerank_texts(encoder, "fox jumping", documents, top_k: 1)

      assert length(results) == 1
    end
  end
end
