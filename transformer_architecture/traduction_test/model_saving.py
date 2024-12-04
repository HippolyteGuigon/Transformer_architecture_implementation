from google.cloud import storage
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "french-english-traduction.json"


def upload_to_gcp_bucket(
    local_file_path: str, bucket_name: str, destination_blob_name: str
)->None:
    """
    Upload a file to a GCP bucket.

    Arguments:
        - local_file_path: str: The path to the local file to upload.
        - bucket_name: str: The name of the GCP bucket.
        - destination_blob_name: str: The name of the
        destination file in the bucket.
    Returns:
        -None
    """

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(local_file_path, timeout=900)

    print(
        f"{local_file_path} uploaded to {bucket_name}/{destination_blob_name}"
    )


def download_file_from_gcs(bucket_name, blob_name, local_file_path):
    """Télécharge un fichier depuis un bucket Google Cloud Storage"""
    
    # Crée un client de stockage
    storage_client = storage.Client()

    # Récupère le bucket
    bucket = storage_client.get_bucket(bucket_name)
    
    # Récupère le blob (fichier) dans le bucket
    blob = bucket.blob(blob_name)
    
    # Télécharge le fichier vers le chemin local
    blob.download_to_filename(local_file_path)

    print(f"Le fichier {blob_name} a été téléchargé dans {local_file_path}")