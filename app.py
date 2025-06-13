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
    Welcome to the **Alarm Sound Classifier**! This is a web-based system that can detect and classify different types of alarm sounds from audio input. This app allows you to either upload .wav files or record live audio using your microphone to identify potential alarms such as fire alarms, sirens, and more.
    
    ### 🔧 Features:
    - Audio file and real-time mic input
    - Automated sound classification using a trained decision tree model
    - Visual feedback including waveform, spectrogram, and prediction label
    
    > Navigate to the tabs above to start uploading or recording your alarm sounds!
    
    """)

elif selected_page == "Audio File-based Sound Classification":
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
                probabilities = model.predict_proba(features)
                prediction = model.predict(features)
                label = label_encoder.inverse_transform(prediction)[0]

                duration = librosa.get_duration(filename=temp_path)
                st.write(f"Audio Duration: {round(duration, 2)} seconds")

                st.markdown(f"<h3 style='font-size:18px;'>Predicted Sound: {label}</h3>", unsafe_allow_html=True)
                
                y, sr = librosa.load(temp_path, sr=16000)

                #Audio Waveform
                st.markdown("### Raw Audio Waveform")
                time_axis = np.linspace(0, len(y) / sr, num=len(y))
                fig_wave_interactive = go.Figure()
                fig_wave_interactive.add_trace(go.Scatter(
                    x=time_axis,
                    y=y,
                    mode='lines',
                    name='Waveform',
                    line=dict(color='royalblue')
                ))
                
                fig_wave_interactive.update_layout(
                    title=f'Raw Audio Waveform of {label}',
                    font=dict(size=18),
                    xaxis_title='Time (s)',
                    yaxis_title='Amplitude',
                    showlegend=False,
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                
                st.plotly_chart(fig_wave_interactive, use_container_width=True)

                #Spectrogram
                st.markdown("### Spectrogram")
                S = librosa.stft(y)
                D = librosa.amplitude_to_db(np.abs(S), ref=np.max)
                frequencies = librosa.fft_frequencies(sr=sr)
                times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr)
                
                fig_spec = go.Figure(data=go.Heatmap(
                    z=D,
                    x=times,
                    y=frequencies,
                    colorscale='Viridis'
                ))
                fig_spec.update_layout(
                    title=f"Spectrogram of {label}",
                    font=dict(size=18),
                    xaxis_title="Time (s)",
                    yaxis_title="Frequency (Hz)"
                )
                st.plotly_chart(fig_spec, use_container_width=True)

                os.remove(temp_path)

        except Exception as e:
            st.error(f"Error processing file: {e}")


#Mic-based Sound Classification
elif selected_page == "Mic-based Sound Classification":
    st.markdown("_Use your microphone to record and classify sounds in real-time or with manual analysis._")

    audio_value = st.audio_input("Record below")

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

                #Audio Waveform
                st.markdown("### Raw Audio Waveform")
                time_axis = np.linspace(0, len(y) / sr, num=len(y))
                fig_wave = go.Figure()
                fig_wave.add_trace(go.Scatter(x=time_axis, y=y, mode='lines', line=dict(color='royalblue')))
                fig_wave.update_layout(
                    title=f'Raw Audio Waveform of {label}', 
                    font=dict(size=18),
                    xaxis_title='Time (s)', 
                    yaxis_title='Amplitude')
                st.plotly_chart(fig_wave, use_container_width=True)

                #Spectrogram
                st.markdown("### Spectrogram")
                S = librosa.stft(y)
                D = librosa.amplitude_to_db(np.abs(S), ref=np.max)
                frequencies = librosa.fft_frequencies(sr=sr)
                times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr)
                
                fig_spec = go.Figure(data=go.Heatmap(
                    z=D,
                    x=times,
                    y=frequencies,
                    colorscale='Viridis'
                ))
                fig_spec.update_layout(
                    title=f"Spectrogram of {label}",
                    font=dict(size=18),
                    xaxis_title="Time (s)",
                    yaxis_title="Frequency (Hz)"
                )
                st.plotly_chart(fig_spec, use_container_width=True)

                os.remove(temp_audio_path)

        except Exception as e:
            st.exception(e)
