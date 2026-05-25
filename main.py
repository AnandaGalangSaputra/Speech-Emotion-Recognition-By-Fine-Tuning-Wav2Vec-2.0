import streamlit as st
import torch
import torch.nn.functional as F

import librosa
import librosa.display

import matplotlib.pyplot as plt
import numpy as np

import tempfile
import os

from transformers import (
    Wav2Vec2FeatureExtractor,
    AutoModelForAudioClassification
)

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Speech Emotion Recognition",
    layout="wide"
)

st.title("Speech Emotion Recognition Indonesia")

st.write(
    "Deteksi emosi suara Bahasa Indonesia menggunakan Wav2Vec2"
)

# =========================
# LOAD MODEL
# =========================

MODEL_NAME = "alianurrahman/wav2vec2-base-indonesian-speech-emotion-recognition"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

@st.cache_resource
def load_model():

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_NAME
    )

    model.to(device)

    return feature_extractor, model

feature_extractor, model = load_model()

# =========================
# FILE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav", "mp3"]
)

# =========================
# PROCESS AUDIO
# =========================

if uploaded_file is not None:

    st.audio(uploaded_file)

    try:

        # =========================
        # SAVE TEMP FILE
        # =========================

        file_extension = uploaded_file.name.split(".")[-1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f".{file_extension}"
        ) as temp_file:

            temp_file.write(uploaded_file.read())

            temp_path = temp_file.name

        # =========================
        # LOAD AUDIO
        # =========================

        audio, sr = librosa.load(
            temp_path,
            sr=16000
        )

        st.success("Audio berhasil diproses")

        # =========================
        # AUDIO INFO
        # =========================

        duration = librosa.get_duration(
            y=audio,
            sr=sr
        )

        st.write(f"Sample Rate: {sr} Hz")

        st.write(f"Durasi Audio: {duration:.2f} detik")

        # =========================
        # WAVEFORM
        # =========================

        st.subheader("Waveform")

        fig_wave, ax = plt.subplots(
            figsize=(12,4)
        )

        librosa.display.waveshow(
            audio,
            sr=sr,
            ax=ax
        )

        ax.set_title("Waveform Audio")

        st.pyplot(fig_wave)

        # =========================
        # SPECTROGRAM
        # =========================

        st.subheader("Spectrogram")

        spectrogram = librosa.amplitude_to_db(
            np.abs(librosa.stft(audio)),
            ref=np.max
        )

        fig_spec, ax2 = plt.subplots(
            figsize=(12,4)
        )

        img = librosa.display.specshow(
            spectrogram,
            sr=sr,
            x_axis='time',
            y_axis='hz',
            ax=ax2
        )

        fig_spec.colorbar(
            img,
            ax=ax2,
            format="%+2.0f dB"
        )

        ax2.set_title("Spectrogram Audio")

        st.pyplot(fig_spec)

        # =========================
        # FEATURE EXTRACTION
        # =========================

        inputs = feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        # =========================
        # PREDICTION
        # =========================

        with torch.no_grad():

            logits = model(**inputs).logits

            probs = F.softmax(
                logits,
                dim=-1
            )

            predicted_id = torch.argmax(
                probs,
                dim=-1
            ).item()

        label = model.config.id2label[
            predicted_id
        ]

        confidence = (
            probs[0][predicted_id].item() * 100
        )

        # =========================
        # RESULT
        # =========================

        st.subheader("Prediction Result")

        st.success(
            f"Predicted Emotion: {label}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )

        # =========================
        # ALL PROBABILITIES
        # =========================

        st.subheader("Emotion Probabilities")

        all_probs = {

            model.config.id2label[i]:
            round(prob.item() * 100, 2)

            for i, prob in enumerate(probs[0])
        }

        st.bar_chart(all_probs)

    except Exception as e:

        st.error(
            f"Gagal memproses audio: {str(e)}"
        )

    finally:

        # hapus temp file
        if 'temp_path' in locals():

            if os.path.exists(temp_path):

                os.remove(temp_path)