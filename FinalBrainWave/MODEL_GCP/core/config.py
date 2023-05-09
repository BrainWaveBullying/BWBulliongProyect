import os
from dotenv import load_dotenv

import tensorflow as tf

load_dotenv()

class Config:
    # Configuración para pruebas
    PROJECT_ID = os.environ.get('PROJECT_ID')
    BUCKET_NAME = os.environ.get('BUCKET_NAME')

    OUTPUT_DIR = os.environ.get('OUTPUT_DIR')
    TEMP_DIR = os.environ.get('TEMP_DIR')

    # Opciones de beam para CGP
    BEAM_OPTIONS = os.environ.get('BEAM_OPTIONS')

    work_dir = os.environ.get('work_dir')
    input = os.environ.get('input')
    output = os.environ.get('output')
    
    #variables que almacenan los nombre de los modelos y scalados
    SCALER_NAME= os.environ.get('SCALER_NAME')
    SCALER_FILE = os.environ.get('SCALER_FILE')
    MODEL_NAME = 'model'

    #Variables del modelo
    EVAL_PERCENT = os.environ.get('EVAL_PERCENT')
    BATCH_SIZE = 1024
    EPOCHS = 8
    STEPS = 1000

    LABELS = ['Trgt Physically attacked']
    input_feature_spec = {
        'Trgt Physically attacked': tf.io.FixedLenFeature([], dtype=tf.float32), 
        'q1 Custom Age': tf.io.FixedLenFeature([], dtype=tf.float32), 
        'q2 Sex': tf.io.FixedLenFeature([], dtype=tf.float32),
        'q3 In what grade are you': tf.io.FixedLenFeature([], dtype=tf.float32), 
        'q4 How often went hungry': tf.io.FixedLenFeature([], dtype=tf.float32),
        'q5 Fast food eating ': tf.io.FixedLenFeature([], dtype=tf.float32), 
        'q6 Physical fighting': tf.io.FixedLenFeature([], dtype=tf.float32),
        'q7 Seriously injured': tf.io.FixedLenFeature([], dtype=tf.float32), 
        'q8 Serious injury type ': tf.io.FixedLenFeature([], dtype=tf.float32),
        'q9 Serious injury cause': tf.io.FixedLenFeature([], dtype=tf.float32), 
        'q10 Felt lonely': tf.io.FixedLenFeature([], dtype=tf.float32), 
        'q11 Could not sleep': tf.io.FixedLenFeature([], dtype=tf.float32),
        'q12 Considered suicide': tf.io.FixedLenFeature([], dtype=tf.float32), 
        'q13 Made a suicide plan': tf.io.FixedLenFeature([], dtype=tf.float32),
        'q14 Attempted suicide': tf.io.FixedLenFeature([], dtype=tf.float32),
        'q15 Close friends': tf.io.FixedLenFeature([], dtype=tf.float32),
        'q16 Initiation of cigarette use': tf.io.FixedLenFeature([], dtype=tf.float32),
        'q17 Current cigarette use': tf.io.FixedLenFeature([], dtype=tf.float32),
        'q18 Initiation of alcohol use': tf.io.FixedLenFeature([], dtype=tf.float32),
        'q19 Current alcohol use': tf.io.FixedLenFeature([], dtype=tf.float32),
        'q20 Drank 2+ drinks ': tf.io.FixedLenFeature([], dtype=tf.float32),
        'q21 Source of alchohol': tf.io.FixedLenFeature([], dtype=tf.float32), 
        'q22 Really drunk': tf.io.FixedLenFeature([], dtype=tf.float32),
        'q23 Trouble from drinking': tf.io.FixedLenFeature([], dtype=tf.float32), 
        'q24 Initiation of drug use': tf.io.FixedLenFeature([], dtype=tf.float32),
        'q25 Ever marijuana use': tf.io.FixedLenFeature([], dtype=tf.float32), 
        'q26 Current marijuana use ': tf.io.FixedLenFeature([], dtype=tf.float32),
        'q27 Amphethamine or methamphetamine use': tf.io.FixedLenFeature([], dtype=tf.float32),
        'q28 Ever sexual intercourse': tf.io.FixedLenFeature([], dtype=tf.float32),
        'q29 Age first had sex': tf.io.FixedLenFeature([], dtype=tf.float32),
        'q30 Number of sex partners': tf.io.FixedLenFeature([], dtype=tf.float32),
        'q31 Condom use': tf.io.FixedLenFeature([], dtype=tf.float32),
        'q32 Birth control used ': tf.io.FixedLenFeature([], dtype=tf.float32), 
        'q33 Physical activity past 7 days': tf.io.FixedLenFeature([], dtype=tf.float32),
        'q34 Walk or bike to school': tf.io.FixedLenFeature([], dtype=tf.float32),
        'q35 PE attendance': tf.io.FixedLenFeature([], dtype=tf.float32),
        'q36 Sitting activities': tf.io.FixedLenFeature([], dtype=tf.float32),
        'q37 Miss school no permission': tf.io.FixedLenFeature([], dtype=tf.float32),
        'q38 Other students kind and helpful ': tf.io.FixedLenFeature([], dtype=tf.float32),
        'q39 Parents check homework ': tf.io.FixedLenFeature([], dtype=tf.float32),
        'q40 Parents understand problems': tf.io.FixedLenFeature([], dtype=tf.float32), 
        'q41 Parents know about free time': tf.io.FixedLenFeature([], dtype=tf.float32),
        'q42 Parents go through their things': tf.io.FixedLenFeature([], dtype=tf.float32)
    }
        
