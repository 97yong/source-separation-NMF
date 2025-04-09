import os
import librosa

def load_audio_signals(data_path='./data', sr=48000, align_length='min'):
    """
    Load audio signals from pre-defined paths:
    - ./data/train/source_1.wav
    - ./data/train/source_2.wav
    - ./data/test/test_1.wav
    """
    file_paths = {
        'source_1': os.path.join(data_path, 'train', 'source_1.wav'),
        'source_2': os.path.join(data_path, 'train', 'source_2.wav'),
        'test':     os.path.join(data_path, 'test', 'test_1.wav')
    }

    signals = {}
    for key, path in file_paths.items():
        signal, _ = librosa.load(path, sr=sr)
        signals[key] = signal

    if align_length == 'min':
        min_len = min(len(sig) for sig in signals.values())
        for key in signals:
            if key == 'test':
                mid = len(signals[key]) // 2
                signals[key] = signals[key][mid:mid+min_len]
            else:
                signals[key] = signals[key][:min_len]

    return signals, sr


def mix_signals(signal1, signal2, ratio=1.0):
    signal1 = signal1[:len(signal2)]
    return signal1 + ratio * signal2