import streamlit as st
from TTS.api import TTS
import numpy as np
import soundfile as sf
from io import BytesIO
import pdfplumber
import docx

# --- Helpers ---
def chunk_text(text, max_len=2000):
    """
    Split text into chunks not exceeding max_len characters, breaking at spaces.
    """
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_len, length)
        if end < length:
            split = text.rfind(" ", start, end)
            if split > start:
                end = split
        chunks.append(text[start:end].strip())
        start = end
    return chunks

# --- Streamlit UI ---
st.title("📚️ Open-Source Book-to-Audiobook")

uploaded = st.file_uploader("Upload a .txt, .pdf, or .docx file", type=["txt", "pdf", "docx"])
if uploaded:
    # Read text based on file type
    if uploaded.type == "application/pdf":
        try:
            with pdfplumber.open(uploaded) as pdf:
                text = "\n\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception as e:
            st.error(f"Failed to parse PDF: {e}")
            st.stop()
    elif uploaded.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
        try:
            doc = docx.Document(uploaded)
            text = "\n\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            st.error(f"Failed to parse Word document: {e}")
            st.stop()
    else:
        text = uploaded.read().decode("utf-8")

    if not text.strip():
        st.warning("No text found in the uploaded file.")
        st.stop()

    # Model selection
    models = TTS.list_models()
    default = "tts_models/en/ljspeech/tacotron2-DDC"
    model_name = st.selectbox("Choose a TTS model", models, index=models.index(default) if default in models else 0)
    tts = TTS(model_name)

    # Synthesize
    if st.button("Generate Audiobook 🏃‍♂️"):
        chunks = chunk_text(text)
        audio_segments = []
        progress = st.progress(0)
        for i, chunk in enumerate(chunks):
            wav = tts.tts(chunk)
            audio_segments.append(wav)
            progress.progress((i + 1) / len(chunks))
        # Concatenate
        combined = np.concatenate(audio_segments)
        # Export to WAV in-memory
        buf = BytesIO()
        sf.write(buf, combined, samplerate=tts.sampling_rate, format="WAV")
        buf.seek(0)
        st.audio(buf, format="audio/wav")
        st.download_button(
            "Download Audiobook",
            data=buf,
            file_name="audiobook.wav",
            mime="audio/wav"
        )

# --- requirements.txt ---
# streamlit>=1.24.0
# TTS>=0.14.0
# numpy
# soundfile
# pdfplumber
# python-docx
