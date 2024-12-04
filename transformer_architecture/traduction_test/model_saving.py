from google.cloud import storage
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "french-english-traduction.json"


def upload_to_gcp_bucket(
    local_file_path: str, bucket_name: str, destination_blob_name: str
) -> None:
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


def download_file_from_gcs(
    bucket_name: str, blob_name: str, local_file_path: str
) -> None:
    """
    Downloads a file from a Google Cloud Storage bucket to a local path.

    Arguments:
        - bucket_name: str: The name of the Google
        Cloud Storage bucket.
        - blob_name: str: The name of the blob (file)
        in the bucket.
        - local_file_path: str: The local path where the
        file will be downloaded.

    Returns:
        - None
    """

    storage_client = storage.Client()

    bucket = storage_client.get_bucket(bucket_name)

    blob = bucket.blob(blob_name)

    blob.download_to_filename(local_file_path)

    print(f"The file '{blob_name}' has been downloaded to '{local_file_path}'")
