
from dotenv import load_dotenv

load_dotenv()

import os
import tensorflow as tf

load_dotenv()

class Config:
    #variables de acceso a cloud
    GOOGLE_APPLICATION_CREDENTIALS = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    GOOGLE_PROJECT_ID = os.environ.get('GOOGLE_PROJECT_ID')
    #variables para la conexion a sql
    DB_HOST = os.environ.get('DB_IP_PUBLIC')
    DB_USER = os.environ.get('DB_USER')
    DB_PASS = os.environ.get('DB_PASSWORD')
    DB_NAME = os.environ.get('DB_NAME')
    DB_PORT = os.environ.get('DB_PORT')
    DB_SCHEMA = os.environ.get('DB_SCHEMA')
    #claves para la autentifcacion
    SECRET_KEY= os.environ.get('SECRET_KEY')
    ACCESS_TOKEN_EXPIRE_MINUTES = os.environ.get('ACCESS_TOKEN_EXPIRE_MINUTES')
    ALGORITHM = os.environ.get('ALGORITHM')
    #modelo de encuesta
    model_id = os.environ.get('model_id')
    #storage models
    BUCKET_NAME = os.environ.get('BUCKET_NAME')
    MODEL_FILE = os.environ.get('MODEL_FILE')
    SCALER_FILE = os.environ.get('SCALER_FILE')
    MODEL_NAME = os.environ.get('MODEL_NAME')
    SCALER_NAME = os.environ.get('SKALER_NAME')
    #Labels
    LABELS = {
    1: 'Bullying',
    0: 'No Bullying'
}




    

    