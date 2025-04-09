import numpy as np
import librosa
from sklearn.decomposition import NMF

def learn_nmf_dictionary(signal, n_components=64, n_fft=1024, hop_length=512, max_iter=1000):
    S = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))
    model = NMF(n_components=n_components, init='random', random_state=0, max_iter=max_iter)
    model.fit(S.T)
    return model.components_


def separate_sources_with_nmf(
    mixed_signal, B_source1, B_source2, sr=48000, n_fft=1024, hop_length=512, max_iter=1000
):
    # Compute magnitude spectrogram
    D_mix = librosa.stft(mixed_signal, n_fft=n_fft, hop_length=hop_length)
    S_mix = np.abs(D_mix).T.astype(np.float32)

    # Combine dictionaries
    B_total = np.vstack([B_source1, B_source2]).astype(np.float32)
    n_total = B_total.shape[0]

    # Apply NMF with custom initialization
    model = NMF(n_components=n_total, init='custom', max_iter=max_iter, solver='mu',
                beta_loss='frobenius', random_state=0)
    W_init = np.abs(np.random.rand(S_mix.shape[0], n_total)).astype(np.float32)
    H_init = B_total.copy()
    W = model.fit_transform(S_mix, W=W_init, H=H_init)

    # Reconstruct sources
    G1 = W[:, :B_source1.shape[0]]
    G2 = W[:, B_source1.shape[0]:]
    S1_hat = G1 @ B_source1
    S2_hat = G2 @ B_source2
    total = S1_hat + S2_hat + 1e-8
    S1_mask = (S1_hat / total).T
    S2_mask = (S2_hat / total).T

    D1 = S1_mask * D_mix
    D2 = S2_mask * D_mix
    y1 = librosa.istft(D1, hop_length=hop_length)
    y2 = librosa.istft(D2, hop_length=hop_length)
    return y1, y2