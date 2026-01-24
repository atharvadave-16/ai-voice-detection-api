import numpy as np
import librosa
import base64
import tempfile
import os

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    f0 = librosa.yin(y, fmin=80, fmax=400, sr=sr)
    f0 = f0[np.isfinite(f0)]
    f0 = f0[f0 > 0]

    feats = {
        "rms_mean": np.mean(rms),
        "rms_std": np.std(rms),
        "centroid_mean": np.mean(centroid),
        "centroid_std": np.std(centroid),
    }

    feats["rms_cv"] = feats["rms_std"] / (feats["rms_mean"] + 1e-6)
    feats["centroid_cv"] = feats["centroid_std"] / (feats["centroid_mean"] + 1e-6)

    if len(f0) > 10:
        feats["pitch_cv"] = np.std(f0) / (np.mean(f0) + 1e-6)
    else:
        feats["pitch_cv"] = 0.0

    return feats


def detect_voice_from_base64(audio_base64):
    # Decode base64 to temp mp3
    audio_bytes = base64.b64decode(audio_base64)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        temp_path = tmp.name

    feats = extract_features(temp_path)
    os.remove(temp_path)

    score = 0.0

    if feats["pitch_cv"] < 0.25:
        score += 0.4
    if feats["centroid_cv"] > 0.8:
        score += 0.4
    if feats["rms_cv"] < 0.7:
        score += 0.2

    if score >= 0.5:
        classification = "AI_GENERATED"
    else:
        classification = "HUMAN"

    confidence = round(min(score, 1.0), 2)

    if confidence >= 0.5:
        explanation = (
            "Voice exhibits stable pitch, consistent spectral patterns, "
            "and regular loudness indicative of AI-generated speech."
        )
    elif confidence >= 0.2:
        explanation = (
            "Voice is largely human-like but shows minor regularities "
            "that can appear in synthetic speech."
        )
    else:
        explanation = (
            "Voice shows natural pitch variation and irregular rhythm "
            "consistent with human speech."
        )

    return classification, confidence, explanation
