from project.models.student_model import Student
from project.models.survey_model import Survey_Questions,Response_Types
from fastapi import HTTPException,Depends
from project.db import db


def get_questions_from_db(student_id):
    """ devuelve las preguntas del cuestionario si existe el id de estudiante """
    student = Student.select().where(Student.student_id == student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Estudiante no encontrado para el colegio proporcionado")
    questions = (Survey_Questions
                 .select(Survey_Questions.quest, Response_Types.answer)
                 .join(Response_Types)
                 .order_by(Survey_Questions.answer_id)
                 .dicts())
    return list(questions)



    
