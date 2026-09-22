import os
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


class LocalVectorEmbeddings(Embeddings):
    """
    Embedder determinístico baseado em n-grams e frequência de termos com projeção normalizada.
    Executa 100% offline, sem dependência de download de pesos externos, produzindo representações
    vetoriais densas adequadas para busca por produto interno / distância L2.
    """

    def __init__(self, dimension: int = 512) -> None:
        self.dimension = dimension

    def _tokenize(self, text: str) -> list[str]:
        words = re.findall(r"\w+", text.lower())
        tokens = list(words)
        # Adiciona 3-grams de caracteres para cobrir morfologia e variações
        for w in words:
            if len(w) >= 3:
                tokens.extend([w[i : i + 3] for i in range(len(w) - 2)])
        return tokens

    def _embed(self, text: str) -> list[float]:
        tokens = self._tokenize(text)
        vec = np.zeros(self.dimension, dtype=np.float32)
        if not tokens:
            return vec.tolist()

        counts = Counter(tokens)
        for token, count in counts.items():
            # Hash estável determinístico (FNV-1a 64-bit)
            h = 14695981039346656037
            for byte in token.encode("utf-8"):
                h = ((h ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
            idx = h % self.dimension
            weight = (1.0 + np.log(count)) * (1.0 + len(token) / 10.0)
            vec[idx] += weight

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def default_corpus() -> list[Document]:
    articles = [
        {
            "title": "Concorrencia e Assincronia com AsyncIO",
            "category": "Python",
            "content": "AsyncIO viabiliza operacoes de I/O nao bloqueantes em Python atraves de async e await para sistemas assincronos.",
        },
        {
            "title": "APIs REST Modernas com FastAPI",
            "category": "Backend",
            "content": "FastAPI oferece alta performance, validacao automatica com Pydantic e geracao de documentacao OpenAPI e Swagger.",
        },
        {
            "title": "Bancos Vetoriais e Busca Semantica com FAISS",
            "category": "AI / Search",
            "content": "FAISS possibilita indexacao e recuperacao vetorial eficiente de embeddings para sistemas RAG e busca semantica.",
        },
        {
            "title": "Persistencia Relacional com SQLAlchemy ORM",
            "category": "Database",
            "content": "SQLAlchemy 2.0 fornece mapeamento objeto-relacional tipado e gerenciamento transacional para PostgreSQL, MySQL e SQLite.",
        },
        {
            "title": "Pipelines RAG com LangChain e LLMs",
            "category": "AI / LLM",
            "content": "Arquiteturas RAG conectam vector stores a modelos de linguagem para fundamentar respostas com base em documentos privados.",
        },
    ]

    return [
        Document(
            page_content=item["content"],
            metadata={"title": item["title"], "category": item["category"]},
        )
        for item in articles
    ]


class SemanticSearchEngine:
    def __init__(self, embeddings: Embeddings | None = None) -> None:
        if embeddings is not None:
            self.embeddings = embeddings
        elif os.getenv("OPENAI_API_KEY"):
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        else:
            self.embeddings = LocalVectorEmbeddings()

        self.index: FAISS | None = None

    def build_index(self, documents: Sequence[Document]) -> None:
        self.index = FAISS.from_documents(
            documents=list(documents),
            embedding=self.embeddings,
        )

    def search(self, query: str, top_k: int = 2) -> list[tuple[Document, float]]:
        if self.index is None:
            raise ValueError("Indice nao inicializado. Chame build_index primeiro.")
        results = self.index.similarity_search_with_score(query, k=top_k)
        return [(doc, float(score)) for doc, score in results]

    def save(self, directory_path: str) -> None:
        if self.index:
            self.index.save_local(directory_path)

    def load(self, directory_path: str) -> None:
        self.index = FAISS.load_local(
            directory_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
