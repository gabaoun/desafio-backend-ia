import datetime

from sqlalchemy import Date, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class Livro(Base):
    __tablename__ = "livros"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, index=True)
    titulo: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    autor: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    data_publicacao: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    resumo: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (Index("idx_livro_titulo_autor", "titulo", "autor"),)

    def __repr__(self) -> str:
        return f"<Livro id={self.id} titulo='{self.titulo}' autor='{self.autor}'>"
