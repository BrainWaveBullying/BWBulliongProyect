from typing import Optional
from pydantic import BaseModel
from typing import List, Dict

class AulaData(BaseModel):
    aulas: Dict[str, int]


class CursoData(BaseModel):
    cursos: Dict[str, AulaData]


class NivelEducativoData(BaseModel):
    niveles_educativos: Dict[str, CursoData]

