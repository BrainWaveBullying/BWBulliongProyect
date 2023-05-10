from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from project.models.school_model import School
from project.config import Config

from project.endpoints.school_endpoint import router as schools_router
from project.endpoints.survey_endpoint import router as survey_router

app = FastAPI(title='API predicción encuesta',
              description='API que realiza la conexión con el servidor donde esta alojado el modelo de predicción')

#http://localhost:8000/

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(schools_router)
app.include_router(survey_router)
