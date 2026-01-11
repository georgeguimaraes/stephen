defmodule StephenColbert.IndexTest do
  use ExUnit.Case, async: true

  alias StephenColbert.Index

  @embedding_dim 4

  describe "new/1" do
    test "creates an empty index" do
      index = Index.new(embedding_dim: @embedding_dim)

      assert Index.size(index) == 0
      assert Index.token_count(index) == 0
      assert Index.doc_ids(index) == []
    end
  end

  describe "add/3" do
    test "adds document embeddings to index" do
      index = Index.new(embedding_dim: @embedding_dim)

      embeddings =
        Nx.tensor([
          [1.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0]
        ])

      index = Index.add(index, "doc1", embeddings)

      assert Index.size(index) == 1
      assert Index.token_count(index) == 2
      assert "doc1" in Index.doc_ids(index)
    end

    test "adds multiple documents" do
      index = Index.new(embedding_dim: @embedding_dim)

      emb1 = Nx.tensor([[1.0, 0.0, 0.0, 0.0]])
      emb2 = Nx.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

      index =
        index
        |> Index.add("doc1", emb1)
        |> Index.add("doc2", emb2)

      assert Index.size(index) == 2
      assert Index.token_count(index) == 3
    end
  end

  describe "get_embeddings/2" do
    test "retrieves stored embeddings" do
      index = Index.new(embedding_dim: @embedding_dim)

      embeddings =
        Nx.tensor([
          [1.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0]
        ])

      index = Index.add(index, "doc1", embeddings)
      retrieved = Index.get_embeddings(index, "doc1")

      assert Nx.shape(retrieved) == {2, 4}
    end

    test "returns nil for missing document" do
      index = Index.new(embedding_dim: @embedding_dim)

      assert Index.get_embeddings(index, "missing") == nil
    end
  end

  describe "search_tokens/3" do
    test "finds candidate documents" do
      index = Index.new(embedding_dim: @embedding_dim)

      # Add two documents with different embeddings
      emb1 = Nx.tensor([[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]])
      emb2 = Nx.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0]])

      index =
        index
        |> Index.add("doc1", emb1)
        |> Index.add("doc2", emb2)

      # Query similar to doc1
      query = Nx.tensor([[0.95, 0.05, 0.0, 0.0]])
      candidates = Index.search_tokens(index, query, 5)

      assert is_map(candidates)
      assert Map.has_key?(candidates, "doc1")
    end
  end

  describe "add_all/2" do
    test "adds multiple documents at once" do
      index = Index.new(embedding_dim: @embedding_dim)

      documents = [
        {"doc1", Nx.tensor([[1.0, 0.0, 0.0, 0.0]])},
        {"doc2", Nx.tensor([[0.0, 1.0, 0.0, 0.0]])},
        {"doc3", Nx.tensor([[0.0, 0.0, 1.0, 0.0]])}
      ]

      index = Index.add_all(index, documents)

      assert Index.size(index) == 3
    end
  end

  describe "save/2 and load/1" do
    @tag :tmp_dir
    test "persists and restores index", %{tmp_dir: tmp_dir} do
      index = Index.new(embedding_dim: @embedding_dim)

      embeddings =
        Nx.tensor([
          [1.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0]
        ])

      index = Index.add(index, "doc1", embeddings)
      path = Path.join(tmp_dir, "test_index")

      assert :ok = Index.save(index, path)
      assert {:ok, loaded} = Index.load(path)

      assert Index.size(loaded) == 1
      assert Index.token_count(loaded) == 2
      assert "doc1" in Index.doc_ids(loaded)

      # Verify embeddings are preserved
      retrieved = Index.get_embeddings(loaded, "doc1")
      assert Nx.shape(retrieved) == {2, 4}
    end
  end
end
