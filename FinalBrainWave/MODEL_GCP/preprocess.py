from __future__ import absolute_import
import joblib
import argparse
import logging
import os
import csv
import random
import pickle
import numpy as np
from datetime import datetime

import tensorflow as tf
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io import ReadFromText #fichero de texto a colección
from apache_beam.io import WriteToText #colección a fichero de texto

from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions, DirectOptions

from apache_beam.transforms import DoFn
from apache_beam.transforms import ParDo

import tensorflow_transform.beam.impl as beam_impl
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_transform.beam.tft_beam_io import transform_fn_io


from core.config import Config
from core.utils import normalize_inputs_train, normalize_inputs_test

settings = Config()

class CheckNoneValues(beam.DoFn):
    def process(self, element):
        if all(value not in (None, '') for value in element.values()):
            yield element
        else:
            logging.warning(f"Element with None values: {element}")

class RemoveNulls(beam.DoFn):
    def process(self, element):
        values = element.split(",")
        if all(value is not None and value != "" for value in values):
            yield element

class ValidateInputData(beam.DoFn):
  """This DoFn validates that every element matches the metadata given."""
  def __init__(self, feature_spec):
    super(ValidateInputData, self).__init__()
    self.feature_names = set(feature_spec.keys())

  def process(self, elem):
    if not isinstance(elem, dict):
      raise ValueError(
          'Element must be a dict(str, value). '
          'Given: {} {}'.format(elem, type(elem)))
    elem_features = set(elem.keys())
    if not self.feature_names.issubset(elem_features):
      raise ValueError(
          "Element features are missing from feature_spec keys. "
          'Given: {}; Features: {}'.format(
              list(elem_features), list(self.feature_names)))
    yield elem

def format_row(row_dict):
    return list(row_dict.values())

#######################################################################################################################      
######################################Configuramos y creamos el Pipeline###############################################
#######################################################################################################################

def run (arg=None, save_main_session= True):
    """método de entrada al pipeline, con save_main_session= True indica que se deben guardar los objetos del espacio 
    de nombres global de la función principal (__main__) en una caché, para que estén disponibles para su reutilización
    en los procesos secundarios que se ejecuten en la pipeline"""


    parser = argparse.ArgumentParser() #Creamos una instacia al parseador de argumentos

    parser.add_argument( #añado argumento y lo guardo en la variable work_dir. Directorio de trabajo
        "--work-dir", dest="work_dir", required=True, help="Working directory",
  )

    parser.add_argument( # añado argumento. Fichero de entrada
        "--input", dest="input", required=True, help="Input dataset in work dir",
    )
    parser.add_argument( # añado argumento. Fichero de salida
        "--output",
        dest="output",
        required=True,
        help="Output path to store transformed data in work dir",
    )
    parser.add_argument( # añado argumento. Indica si estamos entrenando o evaluando
        "--mode",
        dest="mode",
        required=True,
        choices=["train", "test"],
        help="Type of output to store transformed data",
    )

    known_args, pipeline_args = parser.parse_known_args(arg) #recogemos los argumentos enviados por consola y los almacenamos

    # Añadimos la configuración de la pipeline
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session #permite usar dependencias externas como nltk
    pipeline_options.view_as(DirectOptions).direct_num_workers = 0 #número de workers. 0 para que coja el número máximo que tiene la máquina
    
    #creamos el directorio para almacenar el trasnform con el scalado
    tft_temp_dir = os.path.join(settings.work_dir, 'tft-temp')
    transform_fn_dir = os.path.join(settings.work_dir, transform_fn_io.TRANSFORM_FN_DIR)
    if tf.io.gfile.exists(transform_fn_dir):
        tf.io.gfile.rmtree(transform_fn_dir)
    
    #Construimos la pipeline
    with beam.Pipeline(options=pipeline_options) as p, beam_impl.Context(temp_dir=os.path.join(settings.work_dir, 'tft-temp')):
        #Leemos el csv con los datos a traves de una pCollection y realizamos su transformación a float
        input_metadata = dataset_metadata.DatasetMetadata(
        schema_utils.schema_from_feature_spec(settings.input_feature_spec))
        data = (
                p 
                | "ReadCSV" >> ReadFromText(known_args.input, skip_header_lines=1)
                | "DecodeCsvToDict" >> beam.Map(lambda line: dict(zip(settings.input_feature_spec.keys(), line.split(",")))) # Conversion a clave-valor
                | 'Validate inputs' >> beam.ParDo(ValidateInputData(
                    settings.input_feature_spec))  #Valida que los datos de entrada se ajusten al squema indicado
                | 'CheckNonevalues' >> beam.ParDo(CheckNoneValues()) 
                | 'ConvertToFloat' >> beam.Map(lambda row: {k: float(v) if isinstance(v, str) else v for k, v in row.items()})
                                   )     
    
        input_metadata = dataset_metadata.DatasetMetadata(
        schema_utils.schema_from_feature_spec(settings.input_feature_spec))

        if known_args.mode == 'train':
                   
            dataset_and_metadata, transform_fn = (#Normalizamos el conjunto con minmaxscaler
                    (data, input_metadata)
                    | 'FeatureTrainScaling' >> beam_impl.AnalyzeAndTransformDataset(
            normalize_inputs_train))
                                
            dataset_norm, _ = dataset_and_metadata

            #Realizamos el split del conjunto
            assert 0 < int(settings.EVAL_PERCENT) < 100, 'eval_percent must in the range (0-100)'
            train_dataset, eval_dataset = (
            dataset_norm
            | 'Split dataset' >> beam.Partition(
                lambda elem, _: int(random.uniform(0, 100) < int(settings.EVAL_PERCENT) ), 2))


            train_dataset | "FormatRowsTrain" >> beam.Map(format_row) | "TrainWriteToCSV" >> WriteToText( # escribimos el conjunto de train en un csv
                    os.path.join(known_args.output, "train", "part"), #part genera las distintas particiones para cada fichero. Es el prefijo del fichero.
                    file_name_suffix=".csv",  
                                                       
                )
            eval_dataset | "FormatRowsEval" >> beam.Map(format_row) | "EvalWriteToCSV" >> WriteToText( # escribimos el conjunto de eval en un csv
               os.path.join(known_args.output, "eval", "part")) #part genera las distintas particiones para cada fichero. Es el prefijo del fichero.
            
            _ = (#Guardamos el trasnform
            transform_fn
            |   'Write transformFn' >> transform_fn_io.WriteTransformFn(transform_fn_dir))
        
        else: # known_args.mode == "test"
           
           dataset_and_metadata, transform_fn = (#Normalizamos el conjunto con minmaxscaler
                    (data, input_metadata)
                    | 'FeatureTrainScaling' >> beam_impl.AnalyzeAndTransformDataset(
           normalize_inputs_test))
           
           test_dataset, _ = dataset_and_metadata
           
           test_dataset | "EvalWriteToCSV" >> WriteToText( # escribimos el conjunto de eval en un csv
               os.path.join(known_args.output, "test", "part")) #part genera las distintas particiones para cada fichero. Es el prefijo del fichero.
        


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()







