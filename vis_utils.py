import numpy as np
import matplotlib.pyplot as plt
from preprocess_utils import log_specgram

def show_wave(samples, sample_rate=16000, figsize=(20, 4)):
    fig = plt.figure(figsize=figsize)
    plt.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples)
    plt.title('Raw wave')
    plt.ylabel('Amplitude')
    plt.show()
        
def show_spectrogram(wav, sample_rate=16000, figsize=(20, 4)):
    spect = log_specgram(wav)
    print(spect.shape)
    fig = plt.figure(figsize=figsize)
    plt.imshow(spect, aspect='auto', origin='lower')
    plt.title('Spectrogram')
    plt.ylabel('Freqs in Hz')
    plt.xlabel('Seconds')
    plt.show()