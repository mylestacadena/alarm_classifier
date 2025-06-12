import streamlit as st
import numpy as np
import librosa
import joblib
import os
import soundfile as sf
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import tempfile

# === Custom UI Styling ===
st.set_page_config(page_title="Alarm Sound Classifier", layout="centered")

st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: Arial, sans-serif;
        background-color: #F5F5DC;
    }

    .stApp {
        background-color: #F5F5DC;
    }

    .app-title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: #1f3c88;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 18px;
        font-weight: bold;
        color: #1f3c88;
        font-family: Arial, sans-serif;
    }

    .stButton > button {
        font-size: 16px;
        background-color: #DFD0B8;
        color: white;
        border-radius: 10px;
        padding: 0.5em 1.5em;
        border: none;
    }

    .stButton > button:hover {
        background-color: #948979;
    }
    </style>

    <div class="app-title">Alarm Sound Classifier</div>
""", unsafe_allow_html=True)

# === Load Model & Label Encoder ===
model = joblib.load("decision_tree_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# === Feature Extraction ===
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

# === UI Title ===

tab1, tab2 = st.tabs(["📁 Upload Audio File", "🎤 Use Microphone"])

# === File Upload Tab ===
with tab1:
    uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

    if uploaded_file:
        st.audio(uploaded_file, format="audio/wav")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        try:
            with st.spinner("Analyzing audio..."):
                features = extract_features(temp_path)
                prediction = model.predict(features)
                label = label_encoder.inverse_transform(prediction)[0]
                st.success(f"🎯 Predicted Sound: **{label}**")

                # Optional waveform
                y, sr = librosa.load(temp_path, sr=16000)
                fig, ax = plt.subplots()
                ax.plot(y)
                ax.set_title("Waveform")
                ax.set_xlabel("Sample")
                ax.set_ylabel("Amplitude")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# === Microphone Input Tab ===
with tab2:
    st.write("🎤 Record audio from your microphone (requires permission)")

    class AudioProcessor(AudioProcessorBase):
        def __init__(self):
            self.frames = []

        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            audio = frame.to_ndarray().flatten()
            self.frames.append(audio)
            return frame

    ctx = webrtc_streamer(
        key="mic",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=512,
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        audio_processor_factory=AudioProcessor,
    )

    if ctx and ctx.state.playing and hasattr(ctx.state, "audio_processor") and ctx.state.audio_processor:
        st.write(f"🎧 Captured frames: {len(ctx.state.audio_processor.frames)}")

        if st.button("🔍 Predict from Microphone Audio"):
            audio_data = np.concatenate(ctx.state.audio_processor.frames)

            if len(audio_data) < 16000:
                st.warning("Captured audio is too short. Please try again.")
            else:
                temp_path = "mic_input.wav"
                sf.write(temp_path, audio_data, 16000)

                try:
                    with st.spinner("🔍 Analyzing audio..."):
                        features = extract_features(temp_path)
                        prediction = model.predict(features)
                        label = label_encoder.inverse_transform(prediction)[0]
                        st.success(f"Predicted Sound: **{label}**")
                except Exception as e:
                    st.error(f"Error processing mic input: {e}")
                finally:
                    ctx.state.audio_processor.frames.clear()

