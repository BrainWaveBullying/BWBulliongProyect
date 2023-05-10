import uuid

from datetime import datetime
from peewee import DoesNotExist, JOIN, fn, Case
from fastapi import HTTPException, Depends

from typing import List
from project.config import Config
from project.models.student_model import Student
from project.models.school_model import School as SchoolModel
from project.models.master_model import Master
from project.controllers.auth_school import get_user_disabled_current
from typing import List, Dict

settings = Config()

def create_students_with_owner(
            db,
            application: dict,
            current_user: SchoolModel) -> Dict:
      students = []
      results = {}
      year = datetime.now().year
      month = datetime.now().month

      for level, courses in application.items():
            results[level] = {}
            for course, classrooms in courses.items():
                  results[level][course] = {}
                  for classroom, count in classrooms.items():
                        results[level][course][classroom] = []
                        student_ids = []
                        for i in range(count):
                              student_id = f"{year}{month:02d}{current_user.school_id:03d}{level[0:3]}{course[-1]}Au{classroom[-1]}{i + 1:03d}"
                              student_ids.append(student_id)
                              student = Student(student_id=student_id, school_id=current_user.school_id,
                                                date_insert=datetime.utcnow(), credits=0, times_done=0)
                              students.append(student)
                        results[level][course][classroom] = student_ids
      try:
            with db.atomic():
                  # con bulk_create hacemos múltiples registros al mismo tiempo en la base de datos
                  Student.bulk_create(students)
      except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
      return results


def survey_date_query(school_id: int):
      """comprueba las diferentes fechas en las que se solicitaron los id para realizar los test para un scholl_id dado"""
      query = Student.select(Student.dt_insert.distinct()).where(Student.school_id == school_id)
      # query = Student.select(fn.DATE_TRUNC('month', Student.dt_insert).alias('month_year')).distinct().where(Student.school_id == school_id)
      return query


def get_student_info(student_id):
      """metodo que extrae informacion del student_id"""
      levels = {
            'PRI': 'PRIMARIA',
            'SEC': 'SECUNDARIA',
            'BAC': 'BACHILLERATO'
      }

      initial_level = student_id[9:12]
      level = levels[initial_level]
      course = student_id[12:13]
      room = student_id[15:16]
      return {
            'level': level,
            'course': f'Curso {course}',
            'room': f'Aula {room}'
      }


def read_predict(school_id: int,  order_by: str,date: str):
      """Método que realiza la consulta sobre las predicciones a la base de datos por fecha"""
      # primero transformamos el parámetro date a datetime
      date_obj = datetime.strptime(date, "%B - %Y")
      
      order_by_map = {  # Creamos un diccionario con las posibles agrupaciones
            'student_id': Master.student_id,
            'prediction': Master.prediction,
      }
      # Construyendo la consulta
      students = Student.alias()
      masters = Master.alias()

      query = (
            students
                  .select(
                  students.student_id,
                  Case(None, (
                (masters.prediction.is_null(), None),
            ), masters.prediction).alias("prediction"),
                  Case(None,
                  (
                        (masters.prob_prediction >= 0.8, 'A'),
                        (masters.prob_prediction >= 0.6, 'B'),
                        (masters.prob_prediction >= 0.4, 'C')
                  ),
                  'D'
                  ).alias("prob_category")
            )
                  .join(masters, JOIN.LEFT_OUTER, on=(masters.student_id == students.student_id))
                  .where(
                  (students.school_id == school_id) &
                  (students.dt_insert.month == date_obj.month) &
                  (students.dt_insert.year == date_obj.year)
            )
      )

      # Creamos el formato de la respuesta
      results = {}
      
      for record in query:
            
            student_id = record.student_id
            prob_category = record.prob_category
            student_info = get_student_info(student_id)
            # comprobamos que exista la prediccion, de no ser asi, devolvera None
            #try:
            #      prediction = record.prediction
            #      print(prediction)
            #except AttributeError:
            #      prediction = None
            prediction = record.prediction
           
            level = student_info['level']
            course = student_info['course']
            room = student_info['room']

            if level not in results:
                  results[level] = {}
            if course not in results[level]:
                  results[level][course] = {}
            if room not in results[level][course]:
                  results[level][course][room] = {}
            if student_id not in results[level][course][room]:
                  results[level][course][room][student_id] = {}
            
            if prediction is not None:
                  prediction_label = settings.LABELS[int(prediction)]                  
            else:
                  prediction_label = None
            
            results[level][course][room][student_id] = {
                  'prob_prediction': prediction_label,
                  'prob_category': prob_category
            }
      
      
      #return sort_nested_dict(results, order_by)
      return results


def sort_nested_dict(nested_dict, sort_by="key"):
      sorted_dict = {}


      for level1_key, level1_value in nested_dict.items():
            sorted_dict[level1_key] = {}

            for level2_key, level2_value in level1_value.items():
                  sorted_dict[level1_key][level2_key] = {}

                  for level3_key, level3_value in level2_value.items():
                        sorted_items = []
                        for aula in nested_dict[level1_key][level2_key].values():
                        
                              for ids in aula.keys():
                                    if aula[ids]['prob_prediction'] is None:
                                          aula[ids]['prob_prediction'] = "NA"
                                          aula[ids]['prob_category'] = "NA"

                  '''sorted(nested_dict[level1_key][level2_key][room],
                  key=lambda x: (x['prob_prediction'] is None, x['prob_prediction']))'''

                  if sort_by == "student_id":
                        sorted_items = sorted(level3_value.items(), key=lambda x: x[0])
                  elif sort_by == "prediction":
                        sorted_items = sorted(level3_value.items(), key=lambda x: x[1].get("prob_prediction", 0))

                  sorted_dict[level1_key][level2_key][level3_key] = dict(sorted_items)

      
      return sorted_dict



'''# Ordenar estudiantes por id
    if order_by == 'student_id':
        # Ordenar por student_id
        sorted_results = {
            level: {
                course: {
                    room: sorted(results[level][course][room], reverse=True)
                    for room in sorted(results[level][course])
                }
                for course in sorted(results[level])
            }
            for level in sorted(results)
        }
    elif order_by == 'prediction':
        # Ordenar por prediction
        sorted_results = {
            level: {
                course: {
                    room: sorted(results[level][course][room],
                                 key=lambda x: (x['prob_prediction'] is None, x['prob_prediction']))
                    for room in sorted(results[level][course])
                }
                for course in sorted(results[level])
            }
            for level in sorted(results)
        }'''







