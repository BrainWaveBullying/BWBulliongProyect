from datetime import datetime

import peewee
from peewee import DateTimeField

from project.db import db

class School_types(peewee.Model):
    """modelo de la tabla school_types"""
    type_id = peewee.AutoField(primary_key=True) 
    desc_type = peewee.CharField(unique=True, index=True)
    dt_insert = DateTimeField(default=datetime.utcnow)
    dt_update = DateTimeField(unique=False, index=True)

    class Meta:
        database = db

class Countries(peewee.Model):
    """modelo de la tabla countries"""
    country_id = peewee.AutoField(primary_key=True) 
    desc_country = peewee.CharField(unique=True, index=True)
    dt_insert = DateTimeField(default=datetime.utcnow)
    dt_update = DateTimeField(unique=False, index=True)

    class Meta:
        database = db




