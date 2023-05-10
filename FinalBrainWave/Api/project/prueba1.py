import os
import joblib
import tempfile
import numpy as np
import pandas as pd
from google.cloud import storage
from config import Config

settings = Config()

class GCStorage:
    def __init__(self, config):
        GOOGLE_APPLICATION_CREDENTIALS = config
        self.storage_client = storage.Client()
        self.bucket_name = 'model_api_storage'
        print('storage_client instance created successfully')

    def download_blob(self, source_blob_name):
        """descarga de ficheros del bucket."""
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(source_blob_name)

        _, temp_local_filename = tempfile.mkstemp()

        blob.download_to_filename(temp_local_filename)

        return temp_local_filename

# Crear una instancia de la clase GCStorage
gc_storage = GCStorage(settings.GOOGLE_APPLICATION_CREDENTIALS)

model_filename = f'{settings.MODEL_FILE}{settings.MODEL_NAME}.h5'


scaler_filename = f'{settings.SCALER_FILE}{settings.SCALER_NAME}.pkl'

# Descargar un archivo del bucket y comprobar si se ha descargado correctamente
try:
    temp_local_filename = gc_storage.download_blob(model_filename)
    temp_local_filename = gc_storage.download_blob(scaler_filename)
    print(f'Archivo descargado correctamente: {temp_local_filename}')
except Exception as e:
    print(f'Error al descargar el archivo: {e}')
input = [[ 3.  ,  2.  ,  2.  ,  1.62, 52.8 ,  1.  ,  1.  ,  1.  ,  1.  ,
         1.  ,  1.  ,  1.  ,  1.  ,  2.  ,  2.  ,  1.  ,  4.  ,  1.  ,
         1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,
         1.  ,  1.  ,  2.  ,  1.  ,  1.  ,  1.  ,  1.  ,  6.  ,  3.  ,
         1.  ,  2.  ,  1.  ,  3.  ,  3.  ,  5.  ,  5.  ,  1.  ,  2.  ,
         1.8 ,  2.  ]]
input_data = np.array(input)
scaler = joblib.load(open(temp_local_filename, 'rb'))
scaled_input_data = scaler.transform(input_data)


print(f'scaled_adta: {scaled_input_data}')