import streamlit as st
import numpy as np
import pandas as pd
import json
from tensorflow.keras.models import load_model
from PIL import Image
import os

# -------- PATH --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "crop_disease_model.h5")
CLASS_PATH = os.path.join(BASE_DIR, "model", "class_indices.json")
CSV_PATH = os.path.join(BASE_DIR, "fertilizer_data", "fertilizer_pesticide.csv")

# -------- LOAD MODEL --------
@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH, compile=False)

model = load_my_model()

# -------- LOAD CLASSES --------
with open(CLASS_PATH) as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

# -------- LOAD CSV --------
df = pd.read_csv(CSV_PATH)
df = df.applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)

# -------- UI --------
st.title("🌿 Crop Disease Detection System")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

# -------- FUNCTION --------
def predict(img):
    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)
    return model.predict(img)

# -------- MAIN --------
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    pred = predict(img)

    class_idx = np.argmax(pred)
    confidence = round(np.max(pred)*100, 2)

    label = idx_to_class[class_idx]

    if "___" in label:
        crop, disease = label.split("___")
    else:
        crop, disease = label, "unknown"

    crop = crop.lower()
    disease = disease.lower()

    st.success(f"Prediction: {crop.upper()} - {disease.upper()}")
    st.info(f"Confidence: {confidence}%")

    result = df[(df['crop']==crop) & (df['disease']==disease)]

    if not result.empty:
        st.warning(f"Pesticide: {result.iloc[0]['pesticide']}")
        st.write(f"Dosage: {result.iloc[0]['dosage_ml_per_l']}")
        st.write(f"Precautions: {result.iloc[0]['precautions']}")
    else:
        st.error("No recommendation found")