import datetime
import json

from fastapi import HTTPException
from project.models.master_model import Master
from project.models.student_model import Student
from project.models.answers_model import Answers
from project.controllers.survey_controllers import get_questions_from_db

def create_item_master(student_id: str, model_id : int):
    """comprueba si existe el student_id pasado por
    parámetro y si es true crea un registro en la tabla master"""
    student = Student.select().where(Student.student_id == student_id).get()
    if not student:
        raise HTTPException(status_code=404, detail="Estudiante no encontrado para el colegio proporcionado")
    master = Master(
        school_id = student.school_id,
        student_id = student.student_id,
        prediction = None,
        prob_prediction = None,
        model_id = model_id,
        )
    master.save()
    return Master(
        entry_id = master.entry_id,
        school_id= master.school_id,
        student_id = master.student_id,
        prediction= master.prediction,
        prob_prediction = master.prob_prediction,
        comments = master.comments
        )

    # Funcion que realiza el guardado en BBDD de las respuestas generadas por los alumnos
def save_survey_answers(entry_id: str, model_id: int, answers: dict, student_id: str):

    questions = get_questions_from_db(student_id)

    # -- Paso 1: Transformamos en listas las posibles respustas (vienen '["r1", "r2"...]')
    for q in range(len(questions)):
        questions[q]['answer'] = json.loads(questions[q]['answer'])

    # -- Paso 2: Definimos la funcion que va a retornar el indice de la respuesta
    def retIndex(lista: list, elem: str):
        for ix in range(len(lista)):
            if elem == lista[ix]:
                return str(ix)

    # -- Paso 3: Sacamos una lista con las distintas questions y apendeamos el indice de la question en la lista
    start_question_test_list: list = [z["quest"] for z in questions]
    question_text_list: list = []

    for key in range(len(questions)):
        if start_question_test_list[key] in [x for x in answers.keys()]:
            question_text_list.append(key)

    # -- Paso 4: Aplicamos la funcion retindex y modificamos los values de answers
    for i in range(len(question_text_list)):
        # -- Primero hay que tener en cuenta las preguntas que tienen la lista vacía para no aplicar
        if len(questions[question_text_list[i]]["answer"]) != 0:
            answers[questions[question_text_list[i]]["quest"]] = retIndex(questions[question_text_list[i]]["answer"],
                                                                          answers[questions[question_text_list[i]]["quest"]])

    # -- Paso 5: Cambiamos las claves del diccionario answers por el numero de la pregunta (q1, q2....)
    for index in question_text_list:
        answers[f"q{index}"] = answers[questions[index]["quest"]]
        del answers[questions[index]["quest"]]
    answers_register = Answers(entry_id = entry_id, model_id = model_id, answer_dict= str(answers))
    answers_register.save()
    print(answers_register)
    return Answers(entry_id = answers_register.entry_id, 
                   answer_dict= answers_register.answer_dict)

