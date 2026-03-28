import os
import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
import keras
from PIL import Image

# Imports specific for Keras 3
from keras import layers, models
from keras.models import load_model
from keras.utils import register_keras_serializable
from keras.applications.resnet50 import ResNet50, preprocess_input

st.set_page_config(page_title="EmoDec", layout="wide")
st.markdown("""
    <style>
    /* Inverse le flux vidéo en direct ET l'image capturée */
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


def detect_and_crop_face_from_array(image_array, face_cascade, target_size=(224, 224)):
    img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4
    )

    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        face_bgr = img_bgr[y:y+h, x:x+w]
        face_found = True
    else:
        face_bgr = img_bgr
        face_found = False

    face_bgr = cv2.resize(face_bgr, target_size)
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

    return face_rgb.astype(np.float32), face_found


def prepare_image_for_model(image_array, face_cascade):
    cropped_face_rgb, face_found = detect_and_crop_face_from_array(
        image_array=image_array,
        face_cascade=face_cascade,
        target_size=IMG_SIZE,
    )

    model_input = preprocess_input(cropped_face_rgb.copy())
    model_input = np.expand_dims(model_input, axis=0)

    return model_input, cropped_face_rgb.astype(np.uint8), face_found


def predict_emotion(model, image_array, face_cascade):
    model_input, cropped_face_rgb, face_found = prepare_image_for_model(
        image_array=image_array,
        face_cascade=face_cascade,
    )

    preds = model.predict(model_input, verbose=0)[0]
    predicted_index = int(np.argmax(preds))
    confidence = float(np.max(preds))
    predicted_label = int_to_label[predicted_index]

    return predicted_label, confidence, preds, cropped_face_rgb, face_found



model = load_emotion_model()
face_cascade = load_face_cascade()

# ---------------- LEFT SIDE ----------------
with col1:
    st.write("Upload a face image or take a picture to predict emotion.")

    option = st.radio("Choose input method:", ["Upload Image", "Use Camera"])

    image_array = None  # important
    uploaded_file = None
    camera_image = None

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            pil_image = Image.open(uploaded_file).convert("RGB")
            image_array = np.array(pil_image)

            st.subheader("Uploaded Image")
            st.image(image_array, use_container_width=True)

    elif option == "Use Camera":
        camera_image = st.camera_input("Take a picture")

        if camera_image is not None:
            pil_image = Image.open(camera_image).convert("RGB")
            raw_array = np.array(pil_image)
            image_array = np.fliplr(raw_array)

            st.subheader("Captured Image")
            st.image(image_array, use_container_width=True)


# ---------------- RIGHT SIDE ----------------
with col2:
    st.header("Results")

    if uploaded_file is not None or camera_image is not None:
        predicted_label, confidence, preds, cropped_face_rgb, face_found = predict_emotion(
            model=model,
            image_array=image_array,
            face_cascade=face_cascade,
        )

        if not face_found:
            st.warning("No face detected. Please upload an image with a human face.")
        else:
            st.subheader("Processed Face")
            col_left, col_center, col_right = st.columns([1,2,1])

            with col_left:
                st.image(cropped_face_rgb, width=200)

            st.success("Face detected successfully ✅")

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
            
            _ = """
            feedback = st.radio(
                "Is this prediction correct?",
                ["Yes ✅", "No ❌"],
                key="feedback_choice"
            )

            correct_label = None

            if feedback == "No ❌":
                correct_label = st.selectbox(
                    "What emotion were you expressing?",
                    list(int_to_label.values()),
                    key="correct_label"
                )

            if st.button("Submit Feedback"):
                if feedback == "Yes ✅":
                    st.success("Thanks for confirming! 🙌")
                else:
                    st.success("Thanks! This helps improve the model 🚀") """