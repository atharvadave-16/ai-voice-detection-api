import numpy as np
import librosa

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features(audio_path):
    y, sr = librosa.load(audio_path)

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    f0 = librosa.yin(y, fmin=80, fmax=400, sr=sr)
    f0 = f0[np.isfinite(f0)]
    f0 = f0[f0 > 0]

    features = {
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "centroid_mean": float(np.mean(centroid)),
        "centroid_std": float(np.std(centroid)),
    }

    features["rms_cv"] = features["rms_std"] / (features["rms_mean"] + 1e-6)
    features["centroid_cv"] = features["centroid_std"] / (features["centroid_mean"] + 1e-6)

    if len(f0) > 10:
        features["pitch_cv"] = float(np.std(f0) / (np.mean(f0) + 1e-6))
    else:
        features["pitch_cv"] = 0.0

    return features


# -----------------------------
# Final Classification Logic
# -----------------------------
def detect_voice_source(audio_path):
    features = extract_features(audio_path)

    score = 0.0
    reasons = []

    if features["pitch_cv"] < 0.25:
        score += 0.4
        reasons.append("stable pitch patterns")

    if features["centroid_cv"] > 0.8:
        score += 0.4
        reasons.append("spectral consistency")

    if features["rms_cv"] < 0.7:
        score += 0.2
        reasons.append("regular loudness profile")

    # Final decision
    if score >= 0.5:
        classification = "AI_GENERATED"
    elif score >= 0.2:
        classification = "HUMAN LIKE"
    else:
        classification = "HUMAN"

    confidence = round(min(score, 1.0), 2)

    if classification == "AI_GENERATED":
        explanation = ("Voice exhibits stable pitch, consistent spectral patterns, "
        "and regular loudness indicative of AI-generated speech") + ", ".join(reasons)
    elif classification == "HUMAN LIKE":
        explanation = ("Voice is largely human-like but shows minor regularities "
        "that can appear in synthetic speech.")
    else: 
        explanation = ("Voice shows natural pitch variation and irregular rhythm "
        "consistent with human speech.")


    return {
        "classification": classification,
        "confidenceScore": confidence,
        "explanation": explanation
    }


# -----------------------------
# Quick Local Test (temporary)
# -----------------------------
if __name__ == "__main__":
    result = detect_voice_source("ai2.wav")
    print(result)


#🟢 confidenceScore: 0.0

#No AI indicators triggered

#Voice behaves fully like natural human speech

#Classification: HUMAN (very confident)

#🟡 confidenceScore: 0.2

#One weak AI indicator triggered

#Voice is slightly regular

#Classification: HUMAN (low AI suspicion)

#🟠 confidenceScore: 0.4

#One strong AI indicator triggered

#Voice has noticeable regularity

#Still not enough to confidently call AI

#If you ever see:

#confidenceScore >= 0.5
#→ then it becomes AI_GENERATED
