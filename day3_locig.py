import numpy as np
import librosa
y, sr = librosa.load("beat.wav")
#1 loudness
rms = librosa.feature.rms(y=y)
#2 smoothness/noise
zcr = librosa.feature.zero_crossing_rate(y)
#3 sharpness
centroid = librosa.feature.spectral_centroid(y=y, sr = sr)
features = {
    "rms_mean": np.mean(rms),
    "rms_std": np.std(rms),
    "zcr_mean": np.mean(zcr),
    #need to add zrd std for varience stability
    "zcr_std": np.std(zcr),
    "centroid_mean": np.mean(centroid),
    "centroid_std": np.std(centroid)
   

}

print(features)

def is_voice(features):
    rms_mean = features["rms_mean"]
    rms_std = features["rms_std"]
    zcr_mean = features["zcr_mean"]
    zcr_std = features["zcr_std"]
    centroid_mean = features["centroid_mean"]
    centroid_std = features["centroid_std"]

    # RMS modulation
    rms_mod = rms_std / (rms_mean + 1e-6)

    # --- STRONG VOICE GATE ---
    # 1. Too quiet → non-voice
    if rms_mean < 0.01:
        return False

    # 2. Too noisy / beep / metallic → non-voice
    if zcr_mean > 0.12:
        return False

    # 3. Too sharp → non-voice
    if centroid_mean > 3000:
        return False

    # 4. Too stable → likely synthetic / percussion
    if zcr_std < 0.02 and centroid_std < 300:
        return False

    # 5. RMS modulation must be moderate → speech has rhythm
    if rms_mod < 0.35 or rms_mod > 1.5:
        return False

    # 6. Optional: detect short spike patterns → percussion
    if max(rms_std / (rms_mean + 1e-6), centroid_std / (centroid_mean + 1e-6)) > 10:
        return False

    return True


if is_voice(features):
    print("VOICE DETECTED")
else:
    print("NON-VOICE AUDIO")






#That is speech-like amplitude modulation.

#Meaning:

#Energy goes up and down like syllables

#There are pauses / fades

#The envelope is not flat

#From an audio-signal perspective, that file behaves more like speech than a tone.

#So your classifier is doing its job.
