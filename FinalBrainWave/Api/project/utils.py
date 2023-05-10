import os
from core.config import Config
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from google.cloud import storage

settings = Config()

def download_transform_from_bucket(transform_filename, local_dir):
    """
    Downloads a file from a Google Cloud Storage bucket to a local directory.

    Args:
        bucket_name (str): The name of the bucket containing the file.
        transform_filename (str): The name of the file to download.
        local_dir (str): The local directory to which the file will be downloaded.
    """
    client = storage.Client()
    bucket = client.get_bucket(settings.BUCKET_NAME)
    blob = bucket.blob(transform_filename)
    blob.download_to_filename(os.path.join(local_dir, transform_filename))

def normalize_inputs_test(inputs):
    transform_fn_dir = os.path.join(settings.work_dir, transform_fn_io.TRANSFORM_FN_DIR)
    inputs_copy = inputs.copy()
    transform_fn = tft.TFTransformOutput(transform_output_dir = transform_fn_dir)
    normalized_inputs = {'Bullying': inputs_copy['Bullying']}
    raw_features = {}

    # Copiar valores de entrada a raw_features
    for key, value in inputs_copy.items():
        if key == 'Bullying':
            continue
        raw_features[key] = value

    # Aplicar transformación a raw_features
    transformed_features = transform_fn.transform_raw_features(raw_features)
    for key, value in transformed_features.items():
        normalized_inputs[f'norm_{key}'] = value

    return normalized_inputs

