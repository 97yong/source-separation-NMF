import numpy as np
import librosa
import matplotlib.pyplot as plt

def compute_sdr(reference, estimated):
    reference = reference[:len(estimated)]
    estimated = estimated[:len(reference)]
    noise = estimated - reference
    return 10 * np.log10(np.sum(reference ** 2) / (np.sum(noise ** 2) + 1e-8))


def compute_spectrogram(signal, sr=48000, n_fft=1024, hop_length=512):
    D = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    return S_db


def plot_spectrograms(spectrograms, titles, sr=48000, output_path='./outputs/spectrograms.png'):
    plt.figure(figsize=(14, len(spectrograms) * 2.5))
    for i, (spec, title) in enumerate(zip(spectrograms, titles), 1):
        plt.subplot(len(spectrograms), 1, i)
        librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()