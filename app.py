import streamlit as st
import base64
import numpy as np
import librosa
import joblib
import os
import soundfile as sf
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import tempfile
import plotly.graph_objs as go
from streamlit_option_menu import option_menu

# === Page Setup ===
st.set_page_config(page_title="Alarm Sound Classifier", layout="wide")

# === Background Image ===
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

base64_image = get_base64_image("bg.png")
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{base64_image}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
""", unsafe_allow_html=True)

# === Sidebar Navigation ===
with st.sidebar:
    selected_page = option_menu(
        menu_title="Alarm Sound Classifier",
        options=[
            "Home",
            "Upload Audio File",
            "Use Microphone",
        ],
        icons=[
            "house", "file-earmark-arrow-down", "mic"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {
                "padding": "20px",
                "background-color": "#6C63FF",
                "border-radius": "20px",
            },
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {
                "color": "#6C63FF",
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px 0",
                "border-radius": "10px",
                "padding": "10px 15px"
            },
            "nav-link-selected": {
                "background-color": "white",
                "color": "#6C63FF",
                "font-weight": "bold"
            },
            "menu-title": {
                "color": "white",
                "font-size": "24px",
                "font-weight": "bold",
                "padding-bottom": "20px"
            }
        }
    )

# === Global Styling ===
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: Arial, sans-serif;
        background-color: #fefae0;
    }
    .app-title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: #5e503f;
        margin-top: 10px;
        margin-bottom: 20px;
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
    .result-box {
        background-color: #e6f4e6;
        padding: 15px 25px;
        border-radius: 10px;
        font-size: 18px;
        margin-top: 10px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)


# === Load Model & Encoder ===
model = joblib.load("decision_tree_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# === Feature Extraction ===
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)  # Load at original sample rate
    if sr != 16000:
        st.warning("Resampling to 16kHz for processing.")
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
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

# === Page Logic ===
if selected_page == "Home":
    st.markdown('<div class="app-title">Alarm Sound Classifier</div>', unsafe_allow_html=True)
    st.markdown("Welcome to the alarm sound classifier. Choose a mode on the left.")

elif selected_page == "Upload Audio File":
    st.markdown('<div class="app-title">Upload Audio File</div>', unsafe_allow_html=True)
    st.markdown("_Upload a .wav file and see its predicted alarm type with visual analysis._")

    uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
    if uploaded_file:
        st.audio(uploaded_file, format="audio/wav")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        try:
            duration = librosa.get_duration(filename=temp_path)
            st.write(f"Audio Duration: {round(duration, 2)} seconds")
            with st.spinner("Analyzing audio..."):
                features = extract_features(temp_path)
                prediction = model.predict(features)
                label = label_encoder.inverse_transform(prediction)[0]

                st.markdown('<div class="result-box">')
                st.success(f"Predicted Sound: **{label}**")
                st.markdown('</div>', unsafe_allow_html=True)

                y, sr = librosa.load(temp_path, sr=16000)
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                fig = go.Figure(data=go.Heatmap(z=D, colorscale='Viridis'))
                fig.update_layout(title="Spectrogram View", xaxis_title="Time", yaxis_title="Frequency (Hz)")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error processing file: {e}")

    st.markdown("### 🔁 Compare with Sample Alarms")
    if st.checkbox("Enable Sample Comparison"):
        sample_choice = st.selectbox("Choose a sample sound", ["Fire Alarm", "Car Horn", "Dog Bark"])
        sample_path = f"samples/{sample_choice}.wav"
        if os.path.exists(sample_path):
            st.audio(sample_path, format="audio/wav")
        else:
            st.warning("Sample file not found.")

elif selected_page == "Use Microphone":
    st.markdown('<div class="app-title">Use Microphone</div>', unsafe_allow_html=True)
    st.markdown("_Use your microphone to record and classify sounds in real-time._")

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

    if ctx and ctx.state.playing and hasattr(ctx.state, "audio_processor"):
        st.write(f"Captured frames: {len(ctx.state.audio_processor.frames)}")
        if st.button("Predict from Microphone Audio"):
            audio_data = np.concatenate(ctx.state.audio_processor.frames)
            if len(audio_data) < 16000:
                st.warning("Captured audio is too short.")
            else:
                temp_path = "mic_input.wav"
                sf.write(temp_path, audio_data, 16000)
                try:
                    duration = librosa.get_duration(filename=temp_path)
                    st.write(f"Audio Duration: {round(duration, 2)} seconds")
                    with st.spinner("Analyzing audio..."):
                        features = extract_features(temp_path)
                        prediction = model.predict(features)
                        label = label_encoder.inverse_transform(prediction)[0]
                        st.success(f"Predicted Sound: **{label}**")
                except Exception as e:
                    st.error(f"Error processing mic input: {e}")
                finally:
                    ctx.state.audio_processor.frames.clear()


