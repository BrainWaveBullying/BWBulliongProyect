from __future__ import absolute_import

import tempfile
import pickle
import os
#import tensorflow as tf
from datetime import datetime
import numpy as np
from joblib import load
from tensorflow.keras.models import load_model
from project.models.master_model import Master
from project.gcs import GCStorage
from project.config import Config

settings = Config()

class PreprocesingAndPredict:
    """clase que al crear la instancia realiza la descarga de los modelos para """

    def __init__(self):
        self.model_filename = f'{settings.MODEL_FILE}{settings.MODEL_NAME}.h5'
        self.scaler_filename = f'{settings.SCALER_FILE}{settings.SCALER_NAME}.pkl'
        self.bucket_storage = GCStorage(settings.GOOGLE_APPLICATION_CREDENTIALS, settings.BUCKET_NAME)
     # Crear carpeta temporal
        self.temp_dir = tempfile.TemporaryDirectory()

        # Descarga del modelo y scaler del bucket a la carpeta temporal
        self.bucket_storage.download_blob(self.model_filename, f'{self.temp_dir.name}/model.h5')
        self.bucket_storage.download_blob(self.scaler_filename, f'{self.temp_dir.name}/scaler.pkl')

        # Carga del modelo y scaler desde ficheros locales
        self.model = load_model(f'{self.temp_dir.name}/model.h5')
        print(type(self.model))
        with open(f'{self.temp_dir.name}/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        print(type(self.scaler))

    def predict(self, data, entry_id):
        # Preprocesamiento de los datos  
        # Obtener los valores del diccionario como una lista
        values_list = list(data.values())

        # Convertir la lista en un array NumPy
        data_array = np.array(values_list)

        # Redimensionar el array para que tenga una forma de (1, 46)
        data_array = data_array.reshape(1, -1)

        # Predicción
        predict = self.model.predict(data_array)
        indice_clase = np.argmax(predict)
        probabilidad = predict[0][indice_clase]
    
        self.save_predict_from_db(indice_clase,probabilidad,entry_id )
        return indice_clase,probabilidad
    
    @staticmethod
    def save_predict_from_db(indice_clase,probabilidad, entry_id):
        prediction = indice_clase
        prob_prediction = np.float64(probabilidad)
        print(prediction)
        print(prob_prediction)
        registro = Master.get_by_id(entry_id)
        registro.prediction = prediction
        registro.prob_prediction = prob_prediction
        registro.dt_update = datetime.utcnow()
        registro.save()









        

  