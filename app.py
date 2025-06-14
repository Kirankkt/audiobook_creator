import streamlit as st
from TTS.api import TTS
import numpy as np
import soundfile as sf
from io import BytesIO

# --- Helpers ---
def chunk_text(text, max_len=2000):
    """
    Naively split text into chunks of up to max_len characters.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_len, len(text))
        # try to split at the last space before end
        if end < len(text) and ' ' in text[start:end]:
            end = text.rfind(' ', start, end)
        chunks.append(text[start:end].strip())
        start = end
    return chunks

# --- Streamlit UI ---
st.title("📚 Open-Source Audiobook Generator")
st.write(
    "Upload a text file and convert it into an audiobook using Coqui TTS (free, open-source)."
)

# File uploader
uploaded_file = st.file_uploader("Upload your book (.txt)", type=["txt"])

# Model selection
tts_model = st.selectbox(
    "Select a TTS model:",
    [
        "tts_models/en/vctk/vits",
        "tts_models/en/ljspeech/tacotron2-DDC"
    ]
)

if st.button("Generate Audiobook"):
    if not uploaded_file:
        st.error("Please upload a .txt file first.")
    else:
        text = uploaded_file.read().decode("utf-8")
        st.info("Splitting text into chunks...")
        chunks = chunk_text(text)
        st.write(f"Generated {len(chunks)} chunks.")

        # Initialize TTS
        tts = TTS(tts_model)

        # Generate audio per chunk
        all_audio = []
        sr = None
        progress_bar = st.progress(0)
        for i, chunk in enumerate(chunks):
            wav, sr = tts.tts(chunk)
            all_audio.append(wav)
            progress_bar.progress((i + 1) / len(chunks))

        # Concatenate and write to WAV
        st.info("Concatenating audio and preparing download...")
        combined = np.concatenate(all_audio)
        buf = BytesIO()
        sf.write(buf, combined, sr, format="WAV")
        buf.seek(0)

        # Download button
        st.download_button(
            label="📥 Download Audiobook",
            data=buf,
            file_name="audiobook.wav",
            mime="audio/wav"
        )

st.markdown("---")
st.write("""
**Requirements:**
- `pip install TTS soundfile streamlit`

**Instructions:**
1. Run `streamlit run streamlit_tts_app.py`.
2. Upload any `.txt` file.
3. Select the desired open-source TTS model.
4. Click **Generate Audiobook** and download your `.wav` file.
""
