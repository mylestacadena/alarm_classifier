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

# === Top Navigation Bar ===
selected_page = option_menu(
    menu_title="Alarm Sound Classifier",
    options=["Dashboard", "Upload Audio File", "Use Microphone"],
    icons=["house", "file-earmark-arrow-down", "mic"],
    menu_icon="volume-up-fill",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {
            "padding": "10px 20px 20px 20px",  # top right bottom left
            "background-color": "#2f679f",
            "justify-content": "center",
        },
        "icon": {"color": "#f2f7fa", "font-size": "13px"},
        "nav-link": {
            "color": "#f2f7fa",
            "font-size": "13px",
            "margin": "0px 15px",  # space between nav items
            "padding": "10px 20px",
            "border-radius": "8px"
        },
        "nav-link-selected": {
            "background-color": "#e4b795",
            "color": "#ffffff",
            "font-weight": "bold"
        },
        "menu-title": {
            "color": "#ffffff",
            "font-size": "20px",
            "font-weight": "bold",
            "margin": "0px 20px 0px 0px"
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
    .stButton > button {
        font-size: 16px;
        background-color: #f2f7fa;
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

st.markdown("""
    <style>
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)


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
if selected_page == "Dashboard":
    st.markdown("Welcome to the alarm sound classifier. Choose a mode on the left.")

elif selected_page == "Upload Audio File":
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
                
                # Predict probabilities
                probabilities = model.predict_proba(features)
                prediction = model.predict(features)
                label = label_encoder.inverse_transform(prediction)[0]

                # === Display Main Prediction and Confidence ===
                confidence = np.max(probabilities) * 100
                st.success(f"Predicted Sound: **{label}**")
                st.info(f"Model Confidence: **{confidence:.2f}%**")

                # === Display Top 3 Predictions ===
                top_n = 3
                top_indices = np.argsort(probabilities[0])[::-1][:top_n]
                st.markdown("**Top 3 Predictions:**")
                for i in top_indices:
                    pred_label = label_encoder.inverse_transform([i])[0]
                    prob = probabilities[0][i] * 100
                    st.write(f"• {pred_label}: {prob:.2f}%")

                # === Load audio and generate spectrogram ===
                y, sr = librosa.load(temp_path, sr=16000)

                # === Display Raw Waveform ===
                st.markdown("### 📈 Raw Audio Waveform")
                fig_wave, ax_wave = plt.subplots()
                librosa.display.waveshow(y, sr=sr, ax=ax_wave)
                ax_wave.set_title('Waveform')
                ax_wave.set_xlabel('Time (s)')
                ax_wave.set_ylabel('Amplitude')
                st.pyplot(fig_wave)

                # === Spectrogram ===
                st.markdown("### 🔊 Spectrogram View")
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                fig_spec = go.Figure(data=go.Heatmap(z=D, colorscale='Viridis'))
                fig_spec.update_layout(title="Spectrogram", xaxis_title="Time", yaxis_title="Frequency (Hz)")
                st.plotly_chart(fig_spec, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing file: {e}")


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


