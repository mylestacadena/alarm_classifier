import streamlit as st
import librosa
import numpy as np
import joblib
import os

# Load model and label encoder
model = joblib.load("decision_tree_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Audio feature extraction
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossings = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    duration = librosa.get_duration(y=y, sr=sr)

    features = {
        "duration": duration,
        "zcr": np.mean(zero_crossings),
        "rms": np.mean(rms),
        "centroid": np.mean(spectral_centroid),
        "rolloff": np.mean(spectral_rolloff),
    }

    for i, coeff in enumerate(mfcc):
        features[f"mfcc_{i+1}"] = np.mean(coeff)

    return np.array(list(features.values())).reshape(1, -1)

# Streamlit UI
st.title("Real-Time Alarm Sound Classifier")
uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    try:
        features = extract_features("temp.wav")
        prediction = model.predict(features)
        label = label_encoder.inverse_transform(prediction)[0]
        st.success(f"Predicted Sound: **{label}**")
    except Exception as e:
        st.error(f"Error: {e}")

