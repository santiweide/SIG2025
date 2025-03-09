import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as signal

# Load the audio file
sample_rate, audio = wav.read("../TestSignals/Sound samples/progression.wav")

# Convert to mono if stereo
if len(audio.shape) > 1:
    audio = np.mean(audio, axis=1)

# Define STFT parameters
window_size = 1024  # You can modify this to see the impact
hop_size = window_size // 2
window = signal.windows.hann(window_size)

# Compute Short-Time Fourier Transform (STFT)
frequencies, times, spectrogram = signal.spectrogram(audio, fs=sample_rate, 
                                                     window=window, 
                                                     nperseg=window_size, 
                                                     noverlap=hop_size, 
                                                     mode='magnitude')

# Filter to show only 0-1000 Hz
freq_limit = 1000
freq_mask = frequencies <= freq_limit

# Plot the spectrogram
plt.figure(figsize=(10, 6))
plt.pcolormesh(times, frequencies[freq_mask], spectrogram[freq_mask], shading='gouraud', cmap='inferno')
plt.colorbar(label="Magnitude")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Spectrogram of progression.wav (0-1000 Hz)")
plt.show()

