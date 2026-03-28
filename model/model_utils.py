import numpy as np
import cv2
import keras

# Imports specific for Keras 3
from keras.applications.resnet50 import preprocess_input

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