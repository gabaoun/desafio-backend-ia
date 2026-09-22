from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LivroBase(BaseModel):
    titulo: str = Field(..., min_length=1, max_length=255, description="Título do livro")
    autor: str = Field(..., min_length=1, max_length=255, description="Nome do autor")
    data_publicacao: date = Field(..., description="Data de publicação (YYYY-MM-DD)")
    resumo: str = Field(..., min_length=5, description="Sinopse ou resumo do livro")

    @field_validator("titulo", "autor", "resumo")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        v_stripped = v.strip()
        if not v_stripped:
            raise ValueError("O campo não pode conter apenas espaços em branco.")
        return v_stripped


class LivroCreate(LivroBase):
    pass


class LivroResponse(LivroBase):
    id: int

    model_config = ConfigDict(from_attributes=True)
