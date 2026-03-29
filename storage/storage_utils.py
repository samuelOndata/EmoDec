import os
import io
import uuid
import tempfile
import numpy as np
from PIL import Image
from minio import Minio

from dotenv import load_dotenv
load_dotenv()

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


def upload_image(image_input):
    ensure_bucket()
    
    # Conversion en objet PIL si nécessaire
    if isinstance(image_input, np.ndarray):
        pil_image = Image.fromarray(image_input)
    else:
        pil_image = image_input

    # Création d'un buffer en mémoire (pas de fichier sur le disque)
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0) # Revenir au début du flux

    filename = f"{uuid.uuid4()}.jpg"

    # Utilisation de put_object (pour les flux de données) au lieu de fput_object
    client.put_object(
        BUCKET_NAME,
        filename,
        img_byte_arr,
        length=img_byte_arr.getbuffer().nbytes,
        content_type='image/jpeg'
    )

    return filename



def get_image_url(filename):
    """Get presigned URL"""
    try:
        return client.presigned_get_object(BUCKET_NAME, filename)
    except Exception as e:
        print("URL error:", e)
        return None