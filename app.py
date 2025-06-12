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
            with st.spinner("Analyzing audio..."):
                features = extract_features(temp_path)
                
                # Predict probabilities
                probabilities = model.predict_proba(features)
                prediction = model.predict(features)
                label = label_encoder.inverse_transform(prediction)[0]

                duration = librosa.get_duration(filename=temp_path)
                st.write(f"Audio Duration: {round(duration, 2)} seconds")

                # === Load audio and generate spectrogram ===
                y, sr = librosa.load(temp_path, sr=16000)

                # === Display Interactive Raw Waveform ===
                st.markdown("### Raw Audio Waveform")
                
                # Generate time axis for the waveform
                time_axis = np.linspace(0, len(y) / sr, num=len(y))
                
                # Create interactive plot with Plotly
                fig_wave_interactive = go.Figure()
                fig_wave_interactive.add_trace(go.Scatter(
                    x=time_axis,
                    y=y,
                    mode='lines',
                    name='Waveform',
                    line=dict(color='royalblue')
                ))
                
                fig_wave_interactive.update_layout(
                    title='Interactive Audio Waveform',
                    xaxis_title='Time (s)',
                    yaxis_title='Amplitude',
                    showlegend=False,
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                
                st.plotly_chart(fig_wave_interactive, use_container_width=True)

                # === Spectrogram ===
                st.markdown("### Spectrogram")
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                fig_spec = go.Figure(data=go.Heatmap(z=D, colorscale='Viridis'))
                fig_spec.update_layout(title="Spectrogram", xaxis_title="Time", yaxis_title="Frequency (Hz)")
                st.plotly_chart(fig_spec, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing file: {e}")


elif selected_page == "Use Microphone":
    st.markdown('<div class="app-title">🎙️ Use Microphone</div>', unsafe_allow_html=True)
    st.markdown("_Use your microphone to record and classify sounds in real-time or with manual analysis._")

    # === Live Prediction Setup ===
    st.subheader("🔴 Real-Time Sound Classification")
    st.caption("Allow mic permissions in your browser.")

    if "live_prediction" not in st.session_state:
        st.session_state["live_prediction"] = "Waiting..."

    def audio_callback(frame: av.AudioFrame):
        audio = frame.to_ndarray(format="flt32")
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        sr = frame.sample_rate
        try:
            # Save raw audio to temp WAV for feature extraction
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                sf.write(tmp_wav.name, audio, sr)
                features = extract_features(tmp_wav.name)
            if features.shape[1] == model.n_features_in_:
                pred = model.predict(features)[0]
                label = label_encoder.inverse_transform([pred])[0]
                st.session_state["live_prediction"] = f"🔊 {label} ({pred})"
            else:
                st.session_state["live_prediction"] = "⚠️ Feature mismatch"
        except Exception as e:
            st.session_state["live_prediction"] = f"❌ Error: {e}"
        return frame

    webrtc_streamer(
        key="live-audio",
        audio_frame_callback=audio_callback,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    st.info(f"🎧 Real-Time Prediction: **{st.session_state['live_prediction']}**")

    st.divider()

    # === Manual Recording and Analysis ===
    st.subheader("🎤 Manual Microphone Capture & Predict")

    class AudioProcessor(AudioProcessorBase):
        def __init__(self):
            self.frames = []

        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            audio = frame.to_ndarray().flatten()
            self.frames.append(audio)
            return frame

    ctx = webrtc_streamer(
        key="mic-recording",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=512,
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        audio_processor_factory=AudioProcessor,
    )

    if ctx and ctx.state.playing and hasattr(ctx.state, "audio_processor"):
        frames = ctx.state.audio_processor.frames
        st.write(f"🔊 Captured audio chunks: {len(frames)}")

        if st.button("🔍 Predict from Microphone Audio"):
            if not frames:
                st.warning("⚠️ No audio captured. Please speak into the microphone.")
            else:
                audio_data = np.concatenate(frames)
                sr = 16000  # Assume target sample rate

                if len(audio_data) < sr:
                    st.warning("⚠️ Captured audio is too short. Speak longer before predicting.")
                else:
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                            sf.write(tmp_wav.name, audio_data, sr)
                            features = extract_features(tmp_wav.name)

                        prediction = model.predict(features)
                        label = label_encoder.inverse_transform(prediction)[0]
                        st.success(f"🎯 Predicted Alarm Type: **{label}**")

                        # Optional: plot waveform
                        st.markdown("### Captured Audio Waveform")
                        time_axis = np.linspace(0, len(audio_data) / sr, num=len(audio_data))
                        fig_wave = go.Figure()
                        fig_wave.add_trace(go.Scatter(x=time_axis, y=audio_data, mode='lines', name='Waveform'))
                        fig_wave.update_layout(title="Captured Waveform", xaxis_title="Time (s)", yaxis_title="Amplitude")
                        st.plotly_chart(fig_wave, use_container_width=True)

                    except Exception as e:
                        st.error(f"❌ Error during prediction: {e}")
