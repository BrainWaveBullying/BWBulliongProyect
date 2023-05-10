import os
from google.cloud import storage


class GCStorage:

    def __init__(self, config, bucket_name):
        
        GOOGLE_APPLICATION_CREDENTIALS = config
        self.storage_client = storage.Client()
        self.bucket_name = bucket_name

    def download_blob(self, source_blob_name, destination_file_name):
        """descarga de ficheros del bucket."""
        bucket = self.storage_client.get_bucket(self.bucket_name)
        blob = bucket.blob(source_blob_name)

        blob.download_to_filename(destination_file_name)


