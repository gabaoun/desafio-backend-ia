# Backend & AI Engineer Assessment

[![CI Pipeline](https://github.com/gabaoun/desafio-backend-ia/actions/workflows/ci.yml/badge.svg)](https://github.com/gabaoun/desafio-backend-ia/actions)
![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-v0.2+-black?logo=chainlink)
![FAISS](https://img.shields.io/badge/VectorStore-FAISS-00599C)
![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)

Repositório contendo a implementação técnica das 3 etapas da avaliação prática para **Desenvolvedor Backend com Foco em IA**.

---

## Estrutura do Projeto

```text
.
├── .github/workflows/ci.yml   # Pipeline CI (Linting + Test matrix 3.11/3.12)
├── questao_1/                 # API REST de Biblioteca Virtual (FastAPI + SQLite)
│   ├── database.py            # Conexão SQLite + Session engine SQLAlchemy 2.0
│   ├── models.py              # Entidade ORM Livro tipada
│   ├── schemas.py             # Contratos Pydantic v2 com validações
│   ├── main.py                # Endpoints RESTful documentados (OpenAPI)
│   └── test_main.py           # Testes unitários com SQLite in-memory
├── questao_2/                 # Chatbot Especialista com LangChain & GPT-4
│   ├── chatbot.py             # Agente conversacional com histórico por sessão
│   ├── demo.py                # Demonstração CLI interativa / simulada
│   └── test_chatbot.py        # Testes de isolamento de sessão e fluxo
├── questao_3/                 # Busca Semântica com Embeddings & FAISS
│   ├── semantic_search.py     # Motor vetorial (Embeddings + Indexação L2/IP)
│   ├── demo.py                # Pipeline de busca semântica em artigos técnicos
│   └── test_semantic_search.py# Testes de ranking, dimensionalidade e persistência
├── Dockerfile                 # Containerização multi-stage
├── docker-compose.yml         # Orquestração local
├── Makefile                   # Automação de comandos (install, test, lint)
├── pyproject.toml             # Configuração Ruff & Pytest
└── requirements.txt           # Dependências de produção
```

---

## Questão 1: API REST com FastAPI & SQLite

### Destaques Técnicos:
- **FastAPI** assíncrono com injeção de dependências (`Depends(get_db)`).
- **SQLAlchemy 2.0** com declaração tipada (`Mapped`, `mapped_column`) e índices compostos.
- **Pydantic v2** com validações de sanitização e contratos de entrada/saída.
- Suporte a busca combinada (`?q=termo`), filtros parciais por título (`?titulo=`) ou autor (`?autor=`), além de paginação (`?skip=0&limit=50`).
- Documentação interativa Swagger (`/docs`) e ReDoc (`/redoc`).

### Endpoints:
| Método | Rota | Descrição |
| :--- | :--- | :--- |
| `POST` | `/livros/` | Cadastra um novo livro (título, autor, data, resumo) |
| `GET` | `/livros/` | Consulta e lista livros com filtros parciais e paginação |
| `GET` | `/livros/{id}` | Recupera detalhes de um livro por ID |

---

## Questão 2: Chatbot Especialista em Python (LangChain + GPT-4)

### Destaques Técnicos:
- **LangChain Core**: Pipeline baseado em LCEL (`prompt | llm | StrOutputParser`).
- **Isolamento de Sessões**: Gerenciamento de histórico multi-usuário (`session_id`).
- **Prompt Engineering**: System prompt restrito e focado em engenharia de software sênior e PEP 8.
- **Observabilidade**: Integração transparente com LangSmith via variáveis de ambiente.
- Execução híbrida: Utiliza GPT-4 quando `OPENAI_API_KEY` estiver presente ou fallback determinístico para testes e avaliação offline.

---

## Questão 3: Busca Semântica com Embeddings & FAISS

### Destaques Técnicos:
- Indexação vetorial com **FAISS (Facebook AI Similarity Search)**.
- Geração de embeddings vetoriais densos com normalização L2 para cálculo de similaridade de cosseno.
- Suporte a persistência em disco (`save_local` / `load_local`).
- Compatível com modelos OpenAI (`text-embedding-3-small`) e Embeddings locais determinísticos.

---

## Instalação e Execução

### 1. Clonar o repositório e preparar o ambiente:
```bash
git clone https://github.com/gabaoun/desafio-backend-ia.git
cd desafio-backend-ia

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements-dev.txt
```

### 2. Configurar variáveis de ambiente (Opcional para modo live):
```bash
cp .env.example .env
# Edite .env inserindo sua OPENAI_API_KEY se desejar
```

### 3. Rodar a suíte de testes e linter:
```bash
# Executar todos os 15 testes unitários
pytest -v

# Executar linter estático
ruff check .
```

### 4. Executar a API (Questão 1):
```bash
uvicorn questao_1.main:app --reload --port 8000
# Acesse o Swagger em: http://127.0.0.1:8000/docs
```

### 5. Executar as Demonstrações (Questões 2 e 3):
```bash
# Demonstração do Chatbot (Questão 2)
python -m questao_2.demo

# Demonstração da Busca Semântica (Questão 3)
python -m questao_3.demo
```

### 6. Execução via Docker:
```bash
docker-compose up --build
```
