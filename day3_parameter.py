import numpy as np
import librosa

def extract_features(audio_path):
    y, sr = librosa.load(audio_path)

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    # Pitch estimation (F0) using YIN
    f0 = librosa.yin(y, fmin=80, fmax=400, sr=sr)
    f0 = f0[np.isfinite(f0)]
    f0 = f0[(f0 > 0)]

    features = {
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "centroid_mean": float(np.mean(centroid)),
        "centroid_std": float(np.std(centroid)),
    }

    # CVs
    features["rms_cv"] = features["rms_std"] / (features["rms_mean"] + 1e-6)
    features["centroid_cv"] = features["centroid_std"] / (features["centroid_mean"] + 1e-6)

    # Pitch CV
    if len(f0) > 10:
        features["pitch_mean"] = float(np.mean(f0))
        features["pitch_std"] = float(np.std(f0))
        features["pitch_cv"] = features["pitch_std"] / (features["pitch_mean"] + 1e-6)
    else:
        features["pitch_mean"] = 0.0
        features["pitch_std"] = 0.0
        features["pitch_cv"] = 0.0

    return features



# ---------------------------
# STEP B: AI vs HUMAN LOGIC
# ---------------------------
def classify_ai_vs_human(features):
    score = 0.0

    # Pitch irregularity (human usually higher, but not guaranteed)
    if features["pitch_cv"] < 0.25:
        score += 0.4

    # Spectral consistency (AI often higher)
    if features["centroid_cv"] > 0.8:
        score += 0.4

    # Loudness regularity
    if features["rms_cv"] < 0.7:
        score += 0.2

    if score >= 0.5:
        return "AI_GENERATED", score
    else:
        return "HUMAN", score




# ---------------------------
# STEP C: TEST ON FILES
# ---------------------------
for audio_file in ["human.wav", "ai2.wav"]:
    feats = extract_features(audio_file)
    result = classify_ai_vs_human(feats)

    print("\nFile:", audio_file)
    print("pitch_cv:", feats["pitch_cv"], "centroid_cv:", feats["centroid_cv"])
    print("Classification:", result)

label, score = classify_ai_vs_human(feats)
print("Classification:", label, "confidence:", round(score, 2))
#Modern AI voices can closely mimic human prosody. Therefore, instead of relying on a single cue, our system combines pitch variability, spectral consistency, and loudness regularity to produce a probabilistic confidence score.

