from datetime import datetime

import peewee
from peewee import DateTimeField

from project.db import db

from project.models.school_model import School


class Student(peewee.Model):
    student_id = peewee.TextField(primary_key=True) 
    school_id = peewee.ForeignKeyField(model = School, backref = "students" )
    comments = peewee.TextField(unique=False, index=True)
    dt_insert = DateTimeField(default=datetime.utcnow)
    dt_update = DateTimeField(unique=False, index=True)
    credits = peewee.IntegerField(unique=False, index=True)
    times_done = peewee.IntegerField(unique=False, index=True)
    
    
    class Meta:
        indexes = (
            # Creamos un índice único para evitar que se repita el valor de "order" para cada "model_id"
            (('student_id', 'school_id'), True),
        )
        database = db

