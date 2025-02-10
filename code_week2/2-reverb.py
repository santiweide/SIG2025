import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import convolve
from IPython.display import Audio, display

# Load the first sound file
soundbyte, soundbyte_samplerate = sf.read('../TestSignals/Sound samples/laugh2.wav')
print('The shape of our soundbyte is', soundbyte.shape)

# Extract left and right channels
Left = soundbyte[:, 0]
Right = soundbyte[:, 1]

# Plot the waveforms
plt.figure(figsize=(20, 5))
plt.plot(Left, label='Left')
plt.plot(Right, label='Right')
plt.legend()
plt.title("Waveform of laugh2.wav")
plt.show()

# Print sample rate and duration
print('The samplerate of our soundbyte is', soundbyte_samplerate)
print('This means that the soundbyte is', soundbyte.shape[0] / soundbyte_samplerate, 'seconds long')

# Play the audio
display(Audio(soundbyte.T, rate=soundbyte_samplerate, autoplay=False))

# Save the audio file
sf.write('example_write.wav', soundbyte, soundbyte_samplerate)

# Load impulse responses
impulse_clap, _ = sf.read("../TestSignals/Sound impulses/Claps/Concrete-Garage-Avenue--Hand-Clap-Sample--(Schoeps-mk2-omni).wav")
impulse_splash, _ = sf.read("../TestSignals/Sound impulses/Splashes/Splash 9.wav")

# Apply convolution
reverb_clap = convolve(soundbyte, impulse_clap, mode='full')
reverb_splash = convolve(soundbyte, impulse_splash, mode='full')

# Plot original and convolved signals
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(soundbyte, label="Original Signal")
plt.title("Original Audio Signal")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(reverb_clap, label="Reverb with Clap")
plt.title("Signal Convolved with Clap Impulse")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(reverb_splash, label="Reverb with Splash")
plt.title("Signal Convolved with Splash Impulse")
plt.legend()

plt.tight_layout()
plt.show()

# Explanation:
# - Reverb is the persistence of sound after the original sound stops, simulating reflections from surfaces.
# - The convolution process blends the original sound with an impulse response, adding an echo or spatial effect.
# - The altered sound file is longer because convolution extends the duration by adding the impulse response to the tail.

# Play the altered sounds
display(Audio(reverb_clap.T, rate=soundbyte_samplerate, autoplay=False))
display(Audio(reverb_splash.T, rate=soundbyte_samplerate, autoplay=False))

sf.write('reverb_clap.wav', reverb_clap, soundbyte_samplerate)
sf.write('reverb_splash.wav', reverb_splash, soundbyte_samplerate)
