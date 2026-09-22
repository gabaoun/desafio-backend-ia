from pathlib import Path

from questao_3.semantic_search import (
    LocalVectorEmbeddings,
    SemanticSearchEngine,
    default_corpus,
)


def test_embedding_dimensions_and_norm():
    embedder = LocalVectorEmbeddings(dimension=256)
    vector = embedder.embed_query("Query de teste de embedding")
    assert len(vector) == 256
    assert all(isinstance(val, float) for val in vector)


def test_search_retrieval_ranking():
    docs = default_corpus()
    engine = SemanticSearchEngine(embeddings=LocalVectorEmbeddings(dimension=512))
    engine.build_index(docs)

    results = engine.search("FastAPI APIs OpenAPI", top_k=1)
    assert len(results) == 1
    doc, score = results[0]
    assert doc.metadata["title"] == "APIs REST Modernas com FastAPI"
    assert isinstance(score, float)


def test_index_persistence(tmp_path: Path):
    docs = default_corpus()
    embedder = LocalVectorEmbeddings(dimension=512)
    engine = SemanticSearchEngine(embeddings=embedder)
    engine.build_index(docs)

    save_dir = str(tmp_path / "faiss_test")
    engine.save(save_dir)

    reloaded_engine = SemanticSearchEngine(embeddings=embedder)
    reloaded_engine.load(save_dir)

    results = reloaded_engine.search("AsyncIO assincrono", top_k=1)
    assert len(results) == 1
    assert "AsyncIO" in results[0][0].metadata["title"]
