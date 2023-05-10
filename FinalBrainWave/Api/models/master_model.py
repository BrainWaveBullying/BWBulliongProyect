from datetime import datetime

import peewee
from peewee import DateTimeField

from project.db import db

from project.models.school_model import School
from project.models.student_model import Student
from project.models.survey_model import SurveyModels

class Master(peewee.Model):
    """modelo tabla Master"""
    entry_id = peewee.AutoField(primary_key=True) 
    school_id = peewee.ForeignKeyField(model = School, backref='answers')
    student_id = peewee.ForeignKeyField(model= Student, backref='students')
    model_id = peewee.ForeignKeyField(model= SurveyModels, backref='surveymodels')
    prediction = peewee.IntegerField(unique=False, index=True)
    prob_prediction = peewee.IntegerField(unique=False, index=True)
    dt_insert = DateTimeField(default=datetime.utcnow)
    dt_update = DateTimeField(unique=False, index=True)
    comments = peewee.TextField(unique=False, index=True)
    target = peewee.FloatField()

    class Meta:
        database = db