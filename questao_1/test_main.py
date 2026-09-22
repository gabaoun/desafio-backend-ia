import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from .database import Base, get_db
from .main import app

engine_test = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine_test)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_and_teardown_db():
    Base.metadata.create_all(bind=engine_test)
    yield
    Base.metadata.drop_all(bind=engine_test)


def test_criar_livro_com_sucesso():
    payload = {
        "titulo": "Designing Data-Intensive Applications",
        "autor": "Martin Kleppmann",
        "data_publicacao": "2017-03-16",
        "resumo": "Análise aprofundada dos princípios e arquiteturas por trás de sistemas confiáveis e escaláveis.",
    }
    response = client.post("/livros/", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["id"] == 1
    assert data["titulo"] == payload["titulo"]
    assert data["autor"] == payload["autor"]
    assert data["data_publicacao"] == payload["data_publicacao"]
    assert data["resumo"] == payload["resumo"]


@pytest.mark.parametrize(
    "invalid_payload",
    [
        {"titulo": "", "autor": "Autor", "data_publicacao": "2020-01-01", "resumo": "Valido"},
        {"titulo": "   ", "autor": "Autor", "data_publicacao": "2020-01-01", "resumo": "Valido"},
        {"titulo": "Titulo", "autor": "", "data_publicacao": "2020-01-01", "resumo": "Valido"},
        {
            "titulo": "Titulo",
            "autor": "Autor",
            "data_publicacao": "invalid-date",
            "resumo": "Valido",
        },
        {"titulo": "Titulo", "autor": "Autor", "data_publicacao": "2020-01-01", "resumo": "123"},
    ],
)
def test_validacao_campos_invalidos(invalid_payload):
    response = client.post("/livros/", json=invalid_payload)
    assert response.status_code == 422


def test_consulta_livros_filtros():
    livros = [
        {
            "titulo": "Fluent Python",
            "autor": "Luciano Ramalho",
            "data_publicacao": "2022-05-01",
            "resumo": "Guia avançado sobre a linguagem Python.",
        },
        {
            "titulo": "Clean Architecture",
            "autor": "Robert C. Martin",
            "data_publicacao": "2017-09-20",
            "resumo": "Estrutura e design de software.",
        },
        {
            "titulo": "Clean Code",
            "autor": "Robert C. Martin",
            "data_publicacao": "2008-08-01",
            "resumo": "Manual de estilo de desenvolvimento ágil.",
        },
    ]
    for livro in livros:
        res = client.post("/livros/", json=livro)
        assert res.status_code == 201

    # Filtro por autor
    res_autor = client.get("/livros/?autor=Robert")
    assert res_autor.status_code == 200
    assert len(res_autor.json()) == 2

    # Filtro por título
    res_titulo = client.get("/livros/?titulo=Fluent")
    assert res_titulo.status_code == 200
    assert len(res_titulo.json()) == 1
    assert res_titulo.json()[0]["titulo"] == "Fluent Python"

    # Filtro q (busca em ambos)
    res_q = client.get("/livros/?q=Ramalho")
    assert res_q.status_code == 200
    assert len(res_q.json()) == 1


def test_paginacao_livros():
    for i in range(15):
        client.post(
            "/livros/",
            json={
                "titulo": f"Livro {i:02d}",
                "autor": "Autor Teste",
                "data_publicacao": "2024-01-01",
                "resumo": f"Resumo descritivo do livro {i}.",
            },
        )

    res_page_1 = client.get("/livros/?skip=0&limit=5")
    assert res_page_1.status_code == 200
    assert len(res_page_1.json()) == 5

    res_page_2 = client.get("/livros/?skip=5&limit=5")
    assert res_page_2.status_code == 200
    assert len(res_page_2.json()) == 5
    assert res_page_1.json()[0]["id"] != res_page_2.json()[0]["id"]


def test_obter_livro_inexistente_retorna_404():
    response = client.get("/livros/404")
    assert response.status_code == 404
    assert response.json()["detail"] == "Livro com ID 404 não encontrado."
