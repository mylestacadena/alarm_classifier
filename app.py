import streamlit as st

# Page Config
st.set_page_config(page_title="Alarm Sound Classifier", layout="centered")

# Title/Header
st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='font-size: 2.5rem; font-weight: 700; color: #2c3e50;'>🔊 Alarm Sound Classifier</h1>
        <p style='font-size: 1.1rem; color: #7f8c8d;'>Upload a sound file to identify if it’s an alarm or not.</p>
    </div>
""", unsafe_allow_html=True)

# Upload Section
uploaded_file = st.file_uploader("Upload your alarm sound (.wav or .mp3)", type=["wav", "mp3"])

# Display file name and button
if uploaded_file is not None:
    st.success(f"File uploaded: {uploaded_file.name}")

    if st.button("🔍 Classify Sound"):
        with st.spinner("Analyzing sound..."):
            # Replace this with your model prediction function
            prediction = "Alarm"  # Example output
            confidence = 93.5     # Example confidence percentage

        # Result Display
        st.markdown(f"""
            <div style='text-align: center; padding: 1.5rem; background-color: #ecf0f1; border-radius: 10px;'>
                <h2 style='color: #27ae60;'>Prediction: {prediction}</h2>
                <p style='font-size: 1rem; color: #34495e;'>Confidence: {confidence:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)

# Optional: Display waveform or audio preview
if uploaded_file is not None:
    st.audio(uploaded_file)

# Footer
st.markdown("""
    <hr style="margin-top: 3rem;"/>
    <div style='text-align: center; font-size: 0.9rem; color: #95a5a6;'>
        Developed by Mylestacadena · Alarm Classifier Project · 2025
    </div>
""", unsafe_allow_html=True)
