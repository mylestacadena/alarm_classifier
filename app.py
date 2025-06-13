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
from scipy.signal import find_peaks

#Page setup
st.set_page_config(page_title="Alarm Sound Classifier", layout="wide")

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

#Navigation bar
selected_page = option_menu(
    menu_title="Alarm Sound Classifier",
    options=["Dashboard", "Audio File-based Sound Classification", "Mic-based Sound Classification"],
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
            "margin": "0px 15px",  
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


#Load model & encoder from Google Colab
model = joblib.load("decision_tree_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

#Feature extraction
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)  
    if sr != 16000:
        st.warning("Resampling to 16kHz for processing.")
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Extract features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    # zero_crossings = librosa.feature.zero_crossing_rate(y)  # Removed
    # rms = librosa.feature.rms(y=y)  # Removed
    duration = librosa.get_duration(y=y, sr=sr)

    # === Spectral peaks ===
    S = np.abs(librosa.stft(y))  # Magnitude spectrogram
    avg_spectrum = np.mean(S, axis=1)  # Average spectrum over time
    peaks, _ = find_peaks(avg_spectrum)  # Find spectral peaks
    num_peaks = len(peaks)  # Number of spectral peaks

    # Compile features
    features = {
        "duration": duration,
        "centroid": np.mean(spectral_centroid),
        "rolloff": np.mean(spectral_rolloff),
        "num_peaks": num_peaks
    }
    for i, coeff in enumerate(mfcc):
        features[f"mfcc_{i+1}"] = np.mean(coeff)

    return np.array(list(features.values())).reshape(1, -1)

    
#Page logic
if selected_page == "Dashboard":
    st.markdown("""
    Welcome to the **Alarm Sound Classifier**! 
    
    This is a machine learning-powered web application built to identify and classify common emergency sounds, such as school bells and fire alarms. This tool can assist in developing smart monitoring systems, safety automation, and noise-based alert systems.
    This app is designed for simplicity, speed, and clarity, which allow users to upload audio files or record live audio through a microphone, analyze the sound, and receive real-time classification results with supporting graphs.
    
    How it works:
    1. Sound input options
    Choose from two methods to submit your audio:
    • Audio File-based sound classification - Upload a .wav file that contains the sound of a school bell or fire alarm.
    • Mic-based sound classification - Record a sound in real time directly from your device’s microphone.
    
    2. Feature Extraction
    Once an audio file is uploaded or recorded:
    • The app processes the audio using Librosa - a powerful Python library for music and audio analysis.
    • It extracts key audio features, including:
           → MFCCs (Mel-Frequency Cepstral Coefficients) - This captures the timbral texture of the audio.
           → Spectral Centroid - This represents the "center of mass" of the spectrum.
           → Spectral Rolloff - This indicates the frequency below which a certain percentage of the total spectral energy lies.
           → Durationb - This is the total length of the audio.
           → Spectral Peaks - This is the number of significant frequency peaks in the signal.
    
    3. Sound Classification
    • The extracted features are fed into a pre-trained Decision Tree Classifier.
    • The classifier was trained to recognize the sound patterns of school bells and fire alarms.
    • To help users understand and verify the classification, it generates interactive:
            → Waveform plot - This displays the raw amplitude of the audio over time.
            → Spectrogram - This displays how the frequency content of the audio changes over time.

    > Navigate to the tabs above to start uploading or recording your alarm sounds!
    
    """)

elif selected_page == "Audio File-based Sound Classification":
    st.markdown("_Upload a .wav file of SCHOOL BELL or FIRE ALARM for classification._")

    uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
    if uploaded_file:
        st.audio(uploaded_file, format="audio/wav")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        try:
            with st.spinner("Analyzing audio..."):
                features = extract_features(temp_path)
                probabilities = model.predict_proba(features)
                prediction = model.predict(features)
                label = label_encoder.inverse_transform(prediction)[0]

                duration = librosa.get_duration(filename=temp_path)
                st.write(f"Audio Duration: {round(duration, 2)} seconds")

                st.markdown(f"<h3 style='font-size:18px;'>Predicted Sound: {label}</h3>", unsafe_allow_html=True)

                y, sr = librosa.load(temp_path, sr=16000)
                if duration < 1.0:
                    st.warning("Audio is too short (less than 1 second). Please upload a longer file.")
                    st.stop()

                # --- 1. Waveform ---
                st.markdown("### Raw Audio Waveform")
                time_axis = np.linspace(0, len(y) / sr, num=len(y))
                fig_wave = go.Figure()
                fig_wave.add_trace(go.Scatter(x=time_axis, y=y, mode='lines', line=dict(color='royalblue')))
                fig_wave.update_layout(
                    title=f'Raw Audio Waveform of {label}',
                    font=dict(size=18),
                    xaxis_title='Time (s)',
                    yaxis_title='Amplitude',
                    showlegend=False,
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                st.plotly_chart(fig_wave, use_container_width=True)

                # --- 2. Spectrogram ---
                st.markdown("### Spectrogram")
                S = librosa.stft(y)
                D = librosa.amplitude_to_db(np.abs(S), ref=np.max)
                frequencies = librosa.fft_frequencies(sr=sr)
                times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr)
                fig_spec = go.Figure(data=go.Heatmap(
                    z=D,
                    x=times,
                    y=frequencies,
                    colorscale='Viridis'))
                fig_spec.update_layout(
                    title=f"Spectrogram of {label}",
                    xaxis_title='Time (s)',
                    yaxis_title='Frequency (Hz)',
                    margin=dict(l=40, r=40, t=40, b=40))
                st.plotly_chart(fig_spec, use_container_width=True)

                # --- 3. Prediction Confidence ---
                st.markdown("### Prediction Confidence")
                prob_dict = dict(zip(label_encoder.classes_, probabilities[0]))
                fig_prob = go.Figure([go.Bar(x=list(prob_dict.keys()), y=list(prob_dict.values()))])
                fig_prob.update_layout(
                    title="Prediction Confidence",
                    yaxis=dict(title="Probability", range=[0, 1]),
                    xaxis=dict(title="Class"),
                    bargap=0.3
                )
                st.plotly_chart(fig_prob, use_container_width=True)

                # --- 4. Extracted Features ---
                with st.expander("See Extracted Features"):
                    feature_names = ["duration", "centroid", "rolloff", "num_peaks"] + [f"mfcc_{i+1}" for i in range(13)]
                    feature_dict = dict(zip(feature_names, features.flatten()))
                    st.json(feature_dict)

        except Exception as e:
            st.error(f"An error occurred while processing the audio file: {e}")



#Mic-based Sound Classification
elif selected_page == "Mic-based Sound Classification":
    audio_value = st.audio_input("Use your microphone to record SCHOOL BELL or FIRE ALARM for classification.")

    if audio_value:
        st.audio(audio_value, format='audio/wav')

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_value.getvalue())
            temp_audio_path = temp_audio.name

        try:
            with st.spinner("Analyzing audio..."):
                features = extract_features(temp_audio_path)
                probabilities = model.predict_proba(features)
                prediction = model.predict(features)
                label = label_encoder.inverse_transform(prediction)[0]

                duration = librosa.get_duration(filename=temp_audio_path)
                st.write(f"Audio Duration: {round(duration, 2)} seconds")

                st.markdown(f"<h3 style='font-size:18px;'>Predicted Sound: {label}</h3>", unsafe_allow_html=True)

                y, sr = librosa.load(temp_audio_path, sr=16000)
                if duration < 1.0:
                    st.warning("Audio is too short (less than 1 second). Please try recording a longer sound.")
                    st.stop()
                
                # --- 1. Waveform ---
                st.markdown("### Raw Audio Waveform")
                time_axis = np.linspace(0, len(y) / sr, num=len(y))
                fig_wave = go.Figure()
                fig_wave.add_trace(go.Scatter(x=time_axis, y=y, mode='lines', line=dict(color='royalblue')))
                fig_wave.update_layout(
                    title=f'Raw Audio Waveform of {label}',
                    font=dict(size=18),
                    xaxis_title='Time (s)',
                    yaxis_title='Amplitude',
                    showlegend=False,
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                st.plotly_chart(fig_wave, use_container_width=True)

                # --- 2. Spectrogram ---
                st.markdown("### Spectrogram")
                S = librosa.stft(y)
                D = librosa.amplitude_to_db(np.abs(S), ref=np.max)
                frequencies = librosa.fft_frequencies(sr=sr)
                times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr)
                fig_spec = go.Figure(data=go.Heatmap(
                    z=D,
                    x=times,
                    y=frequencies,
                    colorscale='Viridis'))
                fig_spec.update_layout(
                    title=f"Spectrogram of {label}",
                    xaxis_title='Time (s)',
                    yaxis_title='Frequency (Hz)',
                    margin=dict(l=40, r=40, t=40, b=40))
                st.plotly_chart(fig_spec, use_container_width=True)

                # --- 3. Prediction Confidence ---
                st.markdown("### Prediction Confidence")
                prob_dict = dict(zip(label_encoder.classes_, probabilities[0]))
                fig_prob = go.Figure([go.Bar(x=list(prob_dict.keys()), y=list(prob_dict.values()))])
                fig_prob.update_layout(
                    title="Prediction Confidence",
                    yaxis=dict(title="Probability", range=[0, 1]),
                    xaxis=dict(title="Class"),
                    bargap=0.3
                )
                st.plotly_chart(fig_prob, use_container_width=True)

                # --- 4. Extracted Features ---
                with st.expander("See Extracted Features"):
                    feature_names = ["duration", "centroid", "rolloff", "num_peaks"] + [f"mfcc_{i+1}" for i in range(13)]
                    feature_dict = dict(zip(feature_names, features.flatten()))
                    st.json(feature_dict)
                
                os.remove(temp_audio_path)

        except Exception as e:
            st.exception(e)
