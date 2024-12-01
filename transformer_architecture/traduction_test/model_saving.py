from google.cloud import storage
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "french-english-traduction.json"


def upload_to_gcp_bucket(
    local_file_path: str, bucket_name: str, destination_blob_name: str
):
    """
    Upload a file to a GCP bucket.

    Arguments:
        - local_file_path: str: The path to the local file to upload.
        - bucket_name: str: The name of the GCP bucket.
        - destination_blob_name: str: The name of the
        destination file in the bucket.
    """

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(local_file_path, timeout=900)

    print(
        f"{local_file_path} uploaded to {bucket_name}/{destination_blob_name}"
    )


if __name__ == "__main__":
    upload_to_gcp_bucket(
        "models/bonjour.txt", "french-english-raw-data", "bonjour.txt"
    )
