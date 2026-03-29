import os
import numpy as np
import cv2
import hashlib
import streamlit as st
import tensorflow as tf
import keras
from PIL import Image

# Imports specific for Keras 3
from keras import layers, models
from keras.models import load_model
from keras.utils import register_keras_serializable
from model.model_utils import predict_emotion

from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="EmoDec", layout="wide")
st.markdown("""
    <style>
    /* disable mirror effect */
    div[data-testid="stCameraInput"] video, 
    div[data-testid="stCameraInput"] img {
        transform: scaleX(-1);
    }
    </style>
""", unsafe_allow_html=True)

st.title("EmoDec - Facial Expression Recognition Model 😀 😞 😡 🫣")

# Create 2 columns
col1, col2 = st.columns(2)

MODEL_PATH = "model/best_model_20260327-122803.keras"
CASCADE_PATH = "model/haarcascade_frontalface_default.xml"
IMG_SIZE = (224, 224)

label_to_int = {
    "Anger": 0,
    "Contempt": 1,
    "Disgust": 2,
    "Fear": 3,
    "Happy": 4,
    "Neutral": 5,
    "Sad": 6,
    "Suprise": 7
}

int_to_label = {v: k for k, v in label_to_int.items()}


# CBAM (Convolutional Block Attention Module) blocks
# Channel Attention Block
@register_keras_serializable(package="Custom")
class ChannelAttention(layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.shared_dense_1 = None
        self.shared_dense_2 = None

    def build(self, input_shape):
        channel_dim = int(input_shape[-1])

        reduced_dim = max(channel_dim // self.ratio, 1)

        self.shared_dense_1 = layers.Dense(
            units=reduced_dim,
            activation="relu",
            kernel_initializer="he_normal",
            use_bias=True,
            name="ca_dense_1",
        )
        self.shared_dense_2 = layers.Dense(
            units=channel_dim,
            activation=None,
            kernel_initializer="he_normal",
            use_bias=True,
            name="ca_dense_2",
        )

        super().build(input_shape)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)

        avg_out = self.shared_dense_2(self.shared_dense_1(avg_pool))
        max_out = self.shared_dense_2(self.shared_dense_1(max_pool))

        attention = tf.nn.sigmoid(avg_out + max_out)
        return inputs * attention

    def get_config(self):
        config = super().get_config()
        config.update({
            "ratio": self.ratio,
        })
        return config


@register_keras_serializable(package="Custom")
class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = None

    def build(self, input_shape):
        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            strides=1,
            padding="same",
            activation="sigmoid",
            kernel_initializer="he_normal",
            use_bias=False,
            name="sa_conv",
        )
        super().build(input_shape)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)

        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.conv(concat)
        return inputs * attention

    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
        })
        return config


@register_keras_serializable(package="Custom")
class CBAMBlock(layers.Layer):
    def __init__(self, ratio=8, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.channel_attention = ChannelAttention(ratio=ratio, name="channel_attention")
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size, name="spatial_attention")

    def call(self, inputs):
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "ratio": self.ratio,
            "kernel_size": self.kernel_size,
        })
        return config


@st.cache_resource
def load_emotion_model():
    return load_model(
        MODEL_PATH,
        custom_objects={
            "ChannelAttention": ChannelAttention,
            "SpatialAttention": SpatialAttention,
            "CBAMBlock": CBAMBlock,
        },
        compile=False,
    )


@st.cache_resource
def load_face_cascade():
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        raise ValueError("Failed to load Haar cascade classifier.")
    return face_cascade


# backend availability ----

# Check if MinIO and Postgres are available
_ = """ st.cache_resource()
def is_backend_available():
    backend_available = False
    try:
        # MinIO
        from storage.storage_utils import client
        client.list_buckets()

        # Postgres
        import psycopg2
        from db.db_utils import get_connection
        conn = get_connection()
        conn.close()

        backend_available = True
    except Exception as e:
        backend_available = False

    return backend_available """

@st.cache_resource()
def is_backend_available():
    backend_available = False
    try:
        # 1. Test Supabase Storage (Remplace MinIO)
        from storage.storage_utils import supabase, BUCKET_NAME
        # On vérifie juste si on peut accéder au bucket
        supabase.storage.get_bucket(BUCKET_NAME)

        # 2. Test Postgres (Supabase DB)
        from db.db_utils import get_connection
        conn = get_connection()
        conn.close()

        backend_available = True
    except Exception as e:
        # Optionnel : décommenter pour voir l'erreur exacte dans la console
        # st.error(f"Détail erreur backend: {e}") 
        backend_available = False

    return backend_available

USE_BACKEND = is_backend_available()

if USE_BACKEND:
    from storage.storage_utils import upload_image, get_image_url
    from db.db_utils import save_prediction
else:
    st.write("database state NOT seen")
    
    

def safe_upload_image(pil_image):
    if not USE_BACKEND or pil_image is None:
        return None
    try:
        return upload_image(pil_image)
    except Exception as e:
        print("MinIO upload error:", e)
        return None


def render_feedback(image_url, predicted_label, confidence):
    st.subheader("Feedback")

    feedback = st.radio(
        "Is this prediction correct?",
        ["Yes ✅", "No ❌"],
        key="feedback_choice"
    )

    correct_label = predicted_label

    if feedback == "No ❌":
        correct_label = st.selectbox(
            "What emotion were you expressing?",
            list(int_to_label.values()),
            key="correct_label"
        )

    if st.button("Submit Feedback"):

        save_prediction(
            image_url=image_url,
            predicted=predicted_label,
            correct=correct_label,
            confidence=confidence
        )

        st.success("Feedback saved to database 🚀")

# -----------------------
model = load_emotion_model()
face_cascade = load_face_cascade()

# ---------------- LEFT SIDE ----------------
with col1:
    st.write("Upload a face image or take a picture to predict an emotion.")

    option = st.radio("Choose input method:", ["Upload Image", "Use Camera"])

    # variables initialized
    image_array = None
    pil_image = None
    uploaded_file = None
    camera_image = None

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            pil_image = Image.open(uploaded_file).convert("RGB")
            image_array = np.array(pil_image)

            st.subheader("Uploaded Image")
            st.image(image_array, width='stretch')

    elif option == "Use Camera":
        camera_image = st.camera_input("Take a picture")

        if camera_image is not None:
            pil_image = Image.open(camera_image).convert("RGB")
            raw_array = np.array(pil_image)
            image_array = np.fliplr(raw_array)

            st.subheader("Captured Image")
            st.image(image_array, width='stretch')


# ---------------- RIGHT SIDE ----------------
with col2:
    st.header("Results")

    if uploaded_file is not None or camera_image is not None:
        if uploaded_file:
            current_file_id = uploaded_file.name
        else:
            # Create a unique ID based on the pixel data of the camera shot
            # This ensures that even if it's the "camera", a new face = a new ID
            img_hash = hashlib.md5(image_array.tobytes()).hexdigest()
            current_file_id = f"camera_{img_hash}"
        
        predicted_label, confidence, preds, cropped_face_rgb, face_found = predict_emotion(
            model=model,
            image_array=image_array,
            face_cascade=face_cascade,
        )

        if not face_found:
            st.warning("No face detected. Please upload an image with a human face.")
            # Clear previous session data if no face is found
            st.session_state.current_image_url = None
            st.session_state.last_uploaded_file = None

        else:
            st.subheader("Processed Face")
            col_left, col_center, col_right = st.columns([1,2,1])

            with col_left:
                st.image(cropped_face_rgb, width=200)

            st.success("Face detected successfully ✅")
            
            # --- SINGLE UPLOAD LOGIC ---
            # Only upload if the file ID has changed or if we don't have a URL yet
            if ("last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != current_file_id):
                image_filename = safe_upload_image(image_array)
                if image_filename:
                    st.session_state.current_image_url = get_image_url(image_filename)
                    st.session_state.last_uploaded_file = current_file_id
            
            # Retrieve the URL from session state for the rest of the logic
            image_url = st.session_state.get("current_image_url")

            #if image_url:
             #   st.write(f"Stored original image URL: {image_url}")

            st.write(f"Predicted emotion: **{predicted_label}**")
            st.write(f"Confidence: **{confidence:.2%}**")

            st.subheader("Class Probabilities")

            for idx, prob in enumerate(preds):
                label = int_to_label[idx]

                col_label, col_bar = st.columns([1, 3])

                with col_label:
                    st.write(f"{label} ({prob:.2%})")

                with col_bar:
                    st.progress(float(prob))
            
            if USE_BACKEND:
                render_feedback(image_url, predicted_label, confidence)