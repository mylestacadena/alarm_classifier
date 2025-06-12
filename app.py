import streamlit as st
import numpy as np
import librosa
import joblib
import os
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import tempfile

# Load model and label encoder
model = joblib.load("decision_tree_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Feature extraction function
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

# === Streamlit UI ===
st.title("🔊 Real-Time Alarm Sound Classifier")

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
            features = extract_features(temp_path)
            prediction = model.predict(features)
            label = label_encoder.inverse_transform(prediction)[0]
            st.success(f"Predicted Sound: **{label}**")
        except Exception as e:
            st.error(f"Error processing file: {e}")

# === Microphone Tab ===
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
        mode="SENDONLY",
        audio_receiver_size=512,
        media_stream_constraints={"audio": True, "video": False},  # ✅ No client_settings
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        audio_processor_factory=AudioProcessor,
    )

    if ctx and ctx.state.playing:
        if hasattr(ctx.state, "audio_processor") and ctx.state.audio_processor:
            if st.button("🔍 Predict from Microphone Audio"):
                audio_data = np.concatenate(ctx.state.audio_processor.frames)
                if len(audio_data) == 0:
                    st.warning("No audio captured. Please speak into the mic.")
                else:
                    temp_path = "mic_input.wav"
                    librosa.output.write_wav(temp_path, audio_data, sr=16000)

                    try:
                        features = extract_features(temp_path)
                        prediction = model.predict(features)
                        label = label_encoder.inverse_transform(prediction)[0]
                        st.success(f"Predicted Sound: **{label}**")
                    except Exception as e:
                        st.error(f"Error processing mic input: {e}")
        else:
            st.info("🔄 Initializing microphone... please wait.")
