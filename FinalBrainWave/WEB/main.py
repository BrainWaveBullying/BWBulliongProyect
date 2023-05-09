from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI()

# Configuración de CORS
origins = [
    "http://localhost:8000",
    "http://localhost",
    "null",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Item(BaseModel):
    desc_school: str
    cif: str
    phone: str
    email: str
    password: str
    country_id: str
    city: str
    type_id: str


@app.post("/items/")
async def create_item(item: Item):
    return {"item": item.dict()}

# ----------------------------------------------------------------------------------------------------------------------
class LoginReg(BaseModel):
    username: str
    password: str


@app.post("/login/")
async def create_item(item: LoginReg):
    return True

@app.post("/login_student/")
async def create_item(item: LoginReg):
    return True
# ----------------------------------------------------------------------------------------------------------------------


class AulaData(BaseModel):
    aulas: Dict[str, int]


class CursoData(BaseModel):
    cursos: Dict[str, AulaData]


class NivelEducativoData(BaseModel):
    niveles_educativos: Dict[str, CursoData]


@app.post("/applicate_id/")
async def receive_data(data: NivelEducativoData):

    return True

# ----------------------------------------------------------------------------------------------------------------------
questions = {
    "q1": ["r1", "r2", "r3"],
    "q2": ["r1", "r2", "r3", "r4"],
    "q3": ["r1", "r2", "r3", "r4", "r5"]
}


@app.get("/questions")
async def get_questions():
    return questions


class Answer(BaseModel):
    question: str
    answer: str


@app.post("/submit-answers")
async def submit_answers(answers: List[Answer]):
    print(answers)
    # Aquí puedes procesar las respuestas, por ejemplo, guardarlas en una base de datos
    # ...
    
    # Simulamos una respuesta exitosa
    success = True

    # Retorna un objeto JSON con el resultado
    return True