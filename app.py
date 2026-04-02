import streamlit as st
import numpy as np
import pandas as pd
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
model = load_model("model/crop_disease_model.h5", compile=False)

# Load classes
with open("model/class_indices.json") as f:
    class_indices = json.load(f)

idx_to_class = {v:k for k,v in class_indices.items()}

# Load fertilizer/pesticide data
df = pd.read_csv("fertilizer_data/fertilizer_pesticide.csv")
df = df.applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)

# UI
st.set_page_config(page_title="Crop Disease Detector", layout="centered")

st.title("🌿 Crop Disease Detection System")
st.write("Upload a leaf image to detect disease and get recommendations")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    confidence = round(np.max(pred)*100, 2)

    label = idx_to_class[class_idx]

    # Split crop & disease
    crop, disease = label.split("___")
    crop = crop.lower()
    disease = disease.lower()

    st.success(f"Prediction: {crop.upper()} - {disease.upper()}")
    st.info(f"Confidence: {confidence}%")

    # Lookup recommendation
    result = df[(df['crop']==crop) & (df['disease']==disease)]

    if not result.empty:
        pesticide = result.iloc[0]['pesticide']
        dosage = result.iloc[0]['dosage_ml_per_l']
        precaution = result.iloc[0]['precautions']

        st.warning(f"🧪 Pesticide: {pesticide}")
        st.write(f"💧 Dosage: {dosage}")
        st.write(f"⚠️ Precautions: {precaution}")
    else:
        st.error("No recommendation found")