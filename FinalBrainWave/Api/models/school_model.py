from datetime import datetime

import peewee
from peewee import DateTimeField

from project.db import db

from project.models.satellite_tables_model import School_types, Countries

class School(peewee.Model):
    """modelo de la tabla school"""
    school_id = peewee.AutoField(primary_key=True) 
    desc_school = peewee.CharField(unique=False, index=True)
    cif = peewee.CharField(unique=True, index=True)
    phone = peewee.CharField(unique=False, index=True)
    zip_code = peewee.CharField(unique=False, index=True)
    email = peewee.CharField(unique=False, index=True)
    country_id = peewee.ForeignKeyField(model= Countries, field='country_id', backref="schools")
    city = peewee.CharField(unique=False, index=True)    
    password = peewee.CharField(unique=False, index=True)
    credits = peewee.IntegerField(unique=False, index=True)
    type_id = peewee.ForeignKeyField(model= School_types, field='type_id', backref="schools")
    dt_insert = DateTimeField(default=datetime.utcnow)
    dt_update = DateTimeField(unique=False, index=True)
    comments = peewee.TextField(unique=False, index=True)
    disable = peewee.BooleanField(default=False)
    

    class Meta:
        database = db


