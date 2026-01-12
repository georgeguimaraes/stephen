defmodule Stephen.IndexTest do
  use ExUnit.Case, async: true

  alias Stephen.Index

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

  describe "delete/2" do
    test "removes document from index" do
      index = Index.new(embedding_dim: @embedding_dim)

      emb1 = Nx.tensor([[1.0, 0.0, 0.0, 0.0]])
      emb2 = Nx.tensor([[0.0, 1.0, 0.0, 0.0]])

      index =
        index
        |> Index.add("doc1", emb1)
        |> Index.add("doc2", emb2)

      assert Index.size(index) == 2
      assert Index.has_doc?(index, "doc1")

      index = Index.delete(index, "doc1")

      assert Index.size(index) == 1
      refute Index.has_doc?(index, "doc1")
      assert Index.has_doc?(index, "doc2")
      assert Index.get_embeddings(index, "doc1") == nil
    end

    test "returns unchanged index when doc not found" do
      index = Index.new(embedding_dim: @embedding_dim)
      emb1 = Nx.tensor([[1.0, 0.0, 0.0, 0.0]])
      index = Index.add(index, "doc1", emb1)

      index2 = Index.delete(index, "nonexistent")

      assert Index.size(index2) == 1
      assert Index.has_doc?(index2, "doc1")
    end

    test "deleted documents are not returned in search" do
      index = Index.new(embedding_dim: @embedding_dim)

      # Add more documents to avoid HNSWLib issues with very small indices
      emb1 = Nx.tensor([[1.0, 0.0, 0.0, 0.0]])
      emb2 = Nx.tensor([[0.0, 1.0, 0.0, 0.0]])
      emb3 = Nx.tensor([[0.0, 0.0, 1.0, 0.0]])
      emb4 = Nx.tensor([[0.0, 0.0, 0.0, 1.0]])

      index =
        index
        |> Index.add("doc1", emb1)
        |> Index.add("doc2", emb2)
        |> Index.add("doc3", emb3)
        |> Index.add("doc4", emb4)

      # Delete doc1
      index = Index.delete(index, "doc1")

      # Search for doc1's embedding - should find doc2/3/4 but not doc1
      query = Nx.tensor([[1.0, 0.0, 0.0, 0.0]])
      candidates = Index.search_tokens(index, query, 3)

      refute Map.has_key?(candidates, "doc1")
      assert Index.size(index) == 3
    end
  end

  describe "delete_all/2" do
    test "removes multiple documents" do
      index = Index.new(embedding_dim: @embedding_dim)

      documents = [
        {"doc1", Nx.tensor([[1.0, 0.0, 0.0, 0.0]])},
        {"doc2", Nx.tensor([[0.0, 1.0, 0.0, 0.0]])},
        {"doc3", Nx.tensor([[0.0, 0.0, 1.0, 0.0]])}
      ]

      index = Index.add_all(index, documents)
      assert Index.size(index) == 3

      index = Index.delete_all(index, ["doc1", "doc3"])

      assert Index.size(index) == 1
      refute Index.has_doc?(index, "doc1")
      assert Index.has_doc?(index, "doc2")
      refute Index.has_doc?(index, "doc3")
    end
  end

  describe "update/3" do
    test "replaces document embeddings" do
      index = Index.new(embedding_dim: @embedding_dim)

      emb1 = Nx.tensor([[1.0, 0.0, 0.0, 0.0]])
      index = Index.add(index, "doc1", emb1)

      # Update with different embeddings
      emb2 = Nx.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
      index = Index.update(index, "doc1", emb2)

      assert Index.size(index) == 1
      assert Index.has_doc?(index, "doc1")

      retrieved = Index.get_embeddings(index, "doc1")
      assert Nx.shape(retrieved) == {2, 4}
    end
  end

  describe "token ID reuse" do
    test "reuses deleted token IDs for new documents" do
      index = Index.new(embedding_dim: @embedding_dim)

      # Add doc with 2 tokens
      emb1 = Nx.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
      index = Index.add(index, "doc1", emb1)

      initial_token_count = index.token_count

      # Delete doc1 - tokens go to deleted pool
      index = Index.delete(index, "doc1")
      assert length(index.deleted_token_ids) == 2

      # Add new doc - should reuse deleted IDs
      emb2 = Nx.tensor([[0.0, 0.0, 1.0, 0.0]])
      index = Index.add(index, "doc2", emb2)

      # Token count shouldn't grow since we reused a deleted ID
      assert index.token_count == initial_token_count
      assert length(index.deleted_token_ids) == 1
    end
  end

  describe "has_doc?/2" do
    test "returns true for existing document" do
      index = Index.new(embedding_dim: @embedding_dim)
      emb = Nx.tensor([[1.0, 0.0, 0.0, 0.0]])
      index = Index.add(index, "doc1", emb)

      assert Index.has_doc?(index, "doc1")
    end

    test "returns false for missing document" do
      index = Index.new(embedding_dim: @embedding_dim)

      refute Index.has_doc?(index, "nonexistent")
    end
  end

  describe "save/load with deletions" do
    @tag :tmp_dir
    test "preserves deleted token IDs after load", %{tmp_dir: tmp_dir} do
      index = Index.new(embedding_dim: @embedding_dim)

      emb1 = Nx.tensor([[1.0, 0.0, 0.0, 0.0]])
      emb2 = Nx.tensor([[0.0, 1.0, 0.0, 0.0]])

      index =
        index
        |> Index.add("doc1", emb1)
        |> Index.add("doc2", emb2)
        |> Index.delete("doc1")

      path = Path.join(tmp_dir, "test_index_deleted")

      assert :ok = Index.save(index, path)
      assert {:ok, loaded} = Index.load(path)

      assert Index.size(loaded) == 1
      refute Index.has_doc?(loaded, "doc1")
      assert Index.has_doc?(loaded, "doc2")
      assert length(loaded.deleted_token_ids) == 1
    end
  end
end
