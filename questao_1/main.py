"""
Questão 1: Desenvolvimento de API REST para Biblioteca Virtual com FastAPI e SQLite.

Endpoints:
- POST /livros/ : Cadastra um novo livro com validação Pydantic.
- GET  /livros/ : Consulta livros com suporte a busca parcial (título/autor/combinada) e paginação.
- GET  /livros/{id} : Recupera um livro específico por ID.
"""

from collections.abc import Sequence

from fastapi import Depends, FastAPI, HTTPException, Query, status
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .models import Livro
from .schemas import LivroCreate, LivroResponse

# Inicializa as tabelas no SQLite caso não existam
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Library API",
    description="REST API para gerenciamento e catálogo de livros em biblioteca virtual.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.post(
    "/livros/",
    response_model=LivroResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Cadastrar livro",
    tags=["Livros"],
)
def criar_livro(livro_in: LivroCreate, db: Session = Depends(get_db)) -> Livro:
    """
    Persiste um novo livro no SQLite com os campos validados via Pydantic v2:
    título, autor, data_publicacao e resumo.
    """
    livro = Livro(
        titulo=livro_in.titulo,
        autor=livro_in.autor,
        data_publicacao=livro_in.data_publicacao,
        resumo=livro_in.resumo,
    )
    db.add(livro)
    db.commit()
    db.refresh(livro)
    return livro


@app.get(
    "/livros/",
    response_model=list[LivroResponse],
    status_code=status.HTTP_200_OK,
    summary="Listar / Consultar livros",
    tags=["Livros"],
)
def listar_livros(
    titulo: str | None = Query(None, description="Filtro parcial por título"),
    autor: str | None = Query(None, description="Filtro parcial por autor"),
    q: str | None = Query(None, description="Busca textual em título OU autor"),
    skip: int = Query(0, ge=0, description="Offset de paginação"),
    limit: int = Query(50, ge=1, le=100, description="Limite de registros por página"),
    db: Session = Depends(get_db),
) -> Sequence[Livro]:
    """
    Consulta livros no banco de dados. Permite filtros por título, autor
    ou termo geral (q), além de controle de paginação (skip/limit).
    """
    stmt = select(Livro)

    # Busca geral textual em título OU autor
    if q:
        search_pattern = f"%{q.strip()}%"
        stmt = stmt.where(
            or_(
                Livro.titulo.ilike(search_pattern),
                Livro.autor.ilike(search_pattern),
            )
        )
    else:
        # Filtros específicos combináveis
        if titulo:
            stmt = stmt.where(Livro.titulo.ilike(f"%{titulo.strip()}%"))
        if autor:
            stmt = stmt.where(Livro.autor.ilike(f"%{autor.strip()}%"))

    stmt = stmt.order_by(Livro.id.desc()).offset(skip).limit(limit)
    return db.scalars(stmt).all()


@app.get(
    "/livros/{livro_id}",
    response_model=LivroResponse,
    status_code=status.HTTP_200_OK,
    summary="Buscar livro por ID",
    tags=["Livros"],
)
def obter_livro(livro_id: int, db: Session = Depends(get_db)) -> Livro:
    """
    Recupera um livro pelo ID único. Retorna 404 caso o registro não exista.
    """
    livro = db.get(Livro, livro_id)
    if not livro:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Livro com ID {livro_id} não encontrado.",
        )
    return livro
