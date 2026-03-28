from minio import Minio
import os
import uuid

# Create client ONCE
client = Minio(
    os.getenv("MINIO_ENDPOINT", "localhost:9000"),
    access_key=os.getenv("MINIO_USER"),
    secret_key=os.getenv("MINIO_PASSWORD"),
    secure=False
)

BUCKET_NAME = "images"


def ensure_bucket():
    """Create bucket if it does not exist"""
    try:
        if not client.bucket_exists(BUCKET_NAME):
            client.make_bucket(BUCKET_NAME)
    except Exception as e:
        print("Bucket error:", e)


def upload_image(pil_image):
    """Upload image to MinIO and return filename"""
    ensure_bucket()

    filename = f"{uuid.uuid4()}.jpg"
    temp_path = f"/tmp/{filename}"

    pil_image.save(temp_path)

    client.fput_object(
        BUCKET_NAME,
        filename,
        temp_path
    )

    return filename


def get_image_url(filename):
    """Get presigned URL"""
    try:
        return client.presigned_get_object(BUCKET_NAME, filename)
    except Exception as e:
        print("URL error:", e)
        return None