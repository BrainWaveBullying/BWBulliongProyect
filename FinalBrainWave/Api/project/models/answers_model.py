import peewee
from project.models.master_model import Master
from project.models.survey_model import SurveyModels

from project.db import db

class Answers(peewee.Model):
    """modelo de la tabla answers"""
    student_answer_id = peewee.AutoField(primary_key=True) 
    entry_id = peewee.ForeignKeyField(model= Master, backref='masters')
    model_id = peewee.ForeignKeyField(model= SurveyModels, backref='masters')
    answer_dict = peewee.TextField()

    class Meta:
        database = db