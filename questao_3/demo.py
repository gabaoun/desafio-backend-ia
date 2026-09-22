from questao_3.semantic_search import (
    LocalVectorEmbeddings,
    SemanticSearchEngine,
    default_corpus,
)


def run_demo() -> None:
    print("=" * 60)
    print("Execucao da Questao 3 - Busca Semantica (Embeddings + FAISS)")
    print("=" * 60)

    docs = default_corpus()
    print(f"\n[1] Carregando {len(docs)} documentos:")
    for d in docs:
        print(f"  - [{d.metadata['category']}] {d.metadata['title']}")

    engine = SemanticSearchEngine(embeddings=LocalVectorEmbeddings())
    engine.build_index(docs)
    print("\n[2] Embeddings gerados e FAISS indexado com sucesso.")

    queries = [
        "Como construir APIs modernas com validacao de schema?",
        "Qual mecanismo utilizar para busca vetorial de documentos?",
        "Como gerenciar conexoes assincronas e concorrencia?",
    ]

    print("\n[3] Executando consultas semanticas:")
    for query in queries:
        print("\n" + "-" * 50)
        print(f'Consulta: "{query}"')
        results = engine.search(query, top_k=2)

        for rank, (doc, score) in enumerate(results, 1):
            print(f"  #{rank} [Distancia L2: {score:.4f}] {doc.metadata['title']}")
            print(f"     Trecho: {doc.page_content}")


if __name__ == "__main__":
    run_demo()
