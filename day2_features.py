import librosa
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
y, sr = librosa.load("test.wav")
#1 loudness
rms = librosa.feature.rms(y=y)
print("rms energy", np.mean(rms))
#2 smoothness/noise
zcr = librosa.feature.zero_crossing_rate(y)
print("zero crossing rate", np.mean(zcr))
#3 sharpness
centroid = librosa.feature.spectral_centroid(y=y, sr = sr)
print("spectral centroid", np.mean(centroid))
#Your three features are on completely different numeric scales:

#Feature	Typical Range
#pectral Centroid	500 – 5000
#ZCR	0.01 – 0.2
#RMS	0.001 – 0.1

#So when  plot  together:

#Spectral centroid dominates the graph
# ZCR & RMS get squashed near zero
# It looks like they’re missing

#They are there, just invisible.we need to normalise them
#[0] helps flatten the 2d array output of librosa to 1d to plot graph
def n(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))
centroid = (n(centroid))[0]
zcr = (n(zcr))[0]
rms = (n(rms))[0]
frames = range(len(centroid))
t = librosa.frames_to_time(frames, sr = sr)
plt.figure(figsize=(10,4))
plt.plot(t,centroid,color='b')
plt.plot(t,zcr,color='r')
plt.plot(t,rms,color='y')

plt.show()
