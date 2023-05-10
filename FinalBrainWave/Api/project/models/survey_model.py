from datetime import datetime

import peewee
from peewee import DateTimeField,CompositeKey,IntegerField,BooleanField

from project.db import db

class SurveyModels(peewee.Model):
    """modelo de la tabla survey_models"""
    model_id = peewee.AutoField(primary_key=True) 
    desc_survey = peewee.CharField(unique=True, index=True)
    comments = peewee.TextField(unique=False, index=True)
    date_insert = DateTimeField(default=datetime.utcnow)
    date_update = DateTimeField(unique=False, index=True)
    
    class Meta:
        database = db

class Response_Types(peewee.Model):
    answer_id = peewee.AutoField(primary_key=True) 
    answer = peewee.CharField( index=True)


class Survey_Questions(peewee.Model):
    """modelo de la tabla survey_questions"""
    model_id = peewee.ForeignKeyField(model=SurveyModels, ) 
    order_num = peewee.IntegerField() 
    quest = peewee.TextField(unique=False, index=True)
    answer_id = peewee.ForeignKeyField(model=Response_Types, )

    class Meta:
        primary_key = CompositeKey('model_id', 'order_num')
        indexes = (
            # Creamos un índice único para evitar que se repita el valor de "order" para cada "model_id"
            (('model_id', 'order_num'), True),
        )
        database = db


