_ = """ import os
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
    try:
        return client.presigned_get_object(BUCKET_NAME, filename)
    except Exception as e:
        print("URL error:", e)
        return None

"""

##########################

import os
import io
import uuid
import numpy as np
from PIL import Image
from supabase import create_client

from dotenv import load_dotenv
load_dotenv()

# Initialisation du client Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

BUCKET_NAME = "images"

def upload_image(image_input):
    """Upload vers Supabase Storage et retourne l'URL publique directe"""
    
    # 1. Conversion en objet PIL si c'est du Numpy (OpenCV)
    if isinstance(image_input, np.ndarray):
        pil_image = Image.fromarray(image_input)
    else:
        pil_image = image_input

    # 2. Préparation du flux mémoire (JPEG)
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()

    filename = f"{uuid.uuid4()}.jpg"

    try:
        # 3. Upload vers le bucket Supabase
        # Note: Le bucket doit être créé manuellement dans l'UI Supabase
        supabase.storage.from_(BUCKET_NAME).upload(
            path=filename,
            file=img_bytes,
            file_options={"content-type": "image/jpeg"}
        )

        # 4. Récupérer l'URL publique
        # Important: Le bucket "images" doit être configuré en "Public" sur Supabase
        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(filename)
        return public_url
        
    except Exception as e:
        print(f"Supabase Storage error: {e}")
        return None

def get_image_url(filename_or_url):
    """Retourne l'URL telle quelle (déjà générée lors de l'upload)"""
    return filename_or_url
