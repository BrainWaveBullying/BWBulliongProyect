from __future__ import absolute_import

import json
import argparse
import multiprocessing as mp
import logging
import tempfile
import os
import pickle

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from datetime import datetime

from core.config import Config

settings = Config()


def train_and_evaluate(
    work_dir, train_df, eval_df, batch_size=1024, epochs=8, steps=1000
    ):
    # Obtenemos la fecha actual en formato AAAA-MM-DD
    today = datetime.today().strftime('%Y-%m-%d')

    # Creamos el directorio del modelo en una carpeta con la fecha actual en el nombre
    model_dir = os.path.join(work_dir, f"data/model_{today}")
    
    # Comprobamos si ya existe un modelo
    if tf.io.gfile.exists(model_dir):
        tf.io.gfile.rmtree(model_dir) #si existe lo eliminamos
    tf.io.gfile.mkdir(model_dir) #creamos un directorio de modelo

    # Configuramos donde guardar el modelo
    run_config = tf.estimator.RunConfig() 
    run_config = run_config.replace(model_dir=model_dir)

    # Nos permite seguir el entrenamiento cada 10 steps
    run_config = run_config.replace(save_summary_steps=10)

    #Configuramos el modelo 
    model = Sequential()

    with tempfile.NamedTemporaryFile(suffix=".h5") as local_file:
        with tf.io.gfile.GFile(
            os.path.join(model_dir, settings.MODEL_NAME), mode="wb" #guardamos el archivo
        ) as gcs_file:
            model.save(local_file.name)
            gcs_file.write(local_file.read())
    
if __name__ == "__main__":

    """Main function called by AI Platform."""

    logging.getLogger().setLevel(logging.INFO) # Activamos el logger

    parser = argparse.ArgumentParser( #Implementamos el parseador de argumentos. Ver script de preprocesamiento para más detalle
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--job-dir",
        help="Directory for staging trainer files. "
        "This can be a Google Cloud Storage path.",
    )

    parser.add_argument(
        "--work-dir",
        required=True,
        help="Directory for staging and working files. "
        "This can be a Google Cloud Storage path.",
    )

   

    args = parser.parse_args() #cargamos todos los argumentos antes mencionados

    train_data_files = tf.io.gfile.glob(
        os.path.join(args.work_dir, "data/output/train/part-*") # el * permite que lea todas las particiones
    ) #archivos de entrenamiento
    
    eval_data_files = tf.io.gfile.glob(
        os.path.join(args.work_dir, "data/output/eval/part-*")
    ) #archivos de test

    #tf_dtypes = {tf.float32: np.float32}
    #dtype = {k: 'float32'  for k in settings.input_feature_spec.keys()}
    df = pd.DataFrame(columns=list(settings.input_feature_spec.keys()))
    
    # iterar sobre cada archivo y cargar en DataFrame individual
    dfs_train = []
    dfs_eval = []

    # Iterar sobre cada archivo y cargar en DataFrame individual
    for file_path in train_data_files:
        data = pd.read_csv(file_path, header=None)
        data[data.columns[0]] = data[data.columns[0]].str.replace('[','', regex=True).astype("float64")
        data[data.columns[-1]] = data[data.columns[-1]].str.replace(']','', regex=True).astype("float64")
        dfs_train.append(data)
    # Concatenar los DataFrames horizontalemente
    train_data = pd.concat(dfs_train, ignore_index=True).dropna()
    


    # Iterar sobre cada archivo y cargar en DataFrame individual
    for file_path in eval_data_files:
        data = pd.read_csv(file_path, header=None)
        data[data.columns[0]] = data[data.columns[0]].str.replace('[','', regex=True).astype("float64")
        data[data.columns[-1]] = data[data.columns[-1]].str.replace(']','', regex=True).astype("float64")
        dfs_eval.append(data)
    # Concatenar los DataFrames horizontalemente
    eval_data = pd.concat(dfs_eval, ignore_index=True).dropna()
    
    train_and_evaluate(
        args.work_dir,
        train_df=train_data,
        eval_df=eval_data,
        batch_size=settings.BATCH_SIZE,
        epochs=settings.EPOCHS,
        steps=settings.STEPS,
    )

