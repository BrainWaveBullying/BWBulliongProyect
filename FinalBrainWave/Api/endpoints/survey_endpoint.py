from fastapi import APIRouter
from fastapi import Depends
from fastapi import status
from fastapi import Body
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from typing import List, Dict

from project.endpoints.school_endpoint import oauth2_scheme

from project.models.school_model import School as SchoolModel
from project.schemas.survey_questions_schema import SurveyQuestions, SurveyAnswers

from project.controllers.survey_controllers import get_questions_from_db
from project.controllers.student_controllers import create_item_master, save_survey_answers
from project.service.predict import PreprocesingAndPredict
from project.db import get_db, db 
from project.config import Config

"""settings = Config()

router = APIRouter(
    prefix= '/survey',
    tags= ['survey']
)

@router.post(
    "/{student_id}",
    response_model = Dict,
    summary = "permite acceder a la encuesta a traves del id del alumno, verificando que exista el id registrado y devuelve un json con las preguntas del test",
    dependencies = [Depends(get_db)])
def get_cuestions( ques : SurveyQuestions= Body(...)):    
    questions = get_cuestions_to_db(ques.student_id)
    return {"questions": questions}
        
@router.post(
    "/submit",
    response_model= Dict,
    summary = "permite almacenar la encuesta a traves del id del alumno en la base de datos",
    dependencies = [Depends(get_db)])

def save_survey_and_predict(answers : SurveyAnswers = Body(...)): 
    student_id= answers.student_id,
    student_answers = answers.answers
    model_id = settings.model_id
    master=create_item_master(#crea la entrada en la tabla master
        student_id= student_id,
        model_id = model_id
         )
    save_answers_students(
        student_id = student_id, 
        model_id = model_id, 
        answers = student_answers)

    #predict = PreprocesingAndPredict()
    return 
"""
settings = Config()

router = APIRouter(
    prefix= '/survey',
    tags= ['survey']
)

@router.post(
    "/questions/{student_id}",
    response_model = Dict,
    summary = "Obtener preguntas de encuesta para un estudiante",
    dependencies = [Depends(get_db)])
def get_survey_questions(student_id: str):
    questions = get_questions_from_db(student_id)
    return {"questions": questions}

@router.post(
    "/submit",
    response_model= Dict,
    summary = "Guardar las respuestas de la encuesta de un estudiante en la base de datos",
    dependencies = [Depends(get_db)])
def submit_survey_answers(answers : SurveyAnswers = Body(...)):
    print(answers)
    # predict = PreprocesingAndPredict()
    student_id= answers.student_id,
    student_answers = answers.answers
    model_id = settings.model_id
    master = create_item_master(student_id=student_id, model_id=model_id)
    entry_id = master.entry_id
    print(entry_id)
    save_survey_answers(entry_id=entry_id, model_id=model_id, answers=student_answers, student_id=student_id)
    return {'master':True}