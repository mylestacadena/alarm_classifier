import streamlit as st
import numpy as np
import tensorflow as tf
import soundfile as sf
import io
from utils.audio_utils import extract_mfcc

# Load TFLite model
@st.cache_resource
def load_model(model_path="your_model.tflite"):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# UI
st.title("🔊 Real-Time Alarm Sound Classifier")
st.write("Upload a `.wav` file to detect the type of alarm sound.")

uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

if uploaded_file:
    try:
        audio_data, sample_rate = sf.read(io.BytesIO(uploaded_file.read()))
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        mfcc_input = extract_mfcc(audio_data, sr=sample_rate)

        interpreter = load_model("alarm_classifier.lite")
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], mfcc_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        predicted_class = int(np.argmax(output))
        labels = {0: "Fire Alarm", 1: "School Bell", 2: "CO Detector"}  # customize
        st.success(f"Predicted Alarm: {labels.get(predicted_class, 'Unknown')}")
        st.bar_chart(output[0])

    except Exception as e:
        st.error(f"Error: {e}")
