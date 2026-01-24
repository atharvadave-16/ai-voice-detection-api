import librosa
audio_path = "test.wav"
y, sr = librosa.load(audio_path)
#y is sound itself like a list of numbers and sr is sample rate how many numbers per sec
print("sample rate:", sr)
print("audio length:", len(y))
duration = len(y)/sr
print("duration in sec", duration)
print("first 10 samples", y[:10])