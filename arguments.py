import argparse

def arguments():
    parser = argparse.ArgumentParser(description="NMF-based source separation for pump and background audio")

    # Paths
    parser.add_argument('--data_path', type=str, default='./data', help='Path to audio data')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save separated audio')

    # Audio & Mix
    parser.add_argument('--sr', type=int, default=48000, help='Sampling rate')
    parser.add_argument('--mix_ratio', type=float, default=1.0, help='Ratio of background sound in mix')

    # NMF settings
    parser.add_argument('--n_components', type=int, default=64, help='Number of NMF components')
    parser.add_argument('--n_fft', type=int, default=1024, help='FFT window size')
    parser.add_argument('--hop_length', type=int, default=512, help='Hop length for STFT')

    # Options
    parser.add_argument('--plot', type = bool, default=True, help='Plot spectrograms')
    parser.add_argument('--play_audio', type = bool, default=True, help='Play separated audio')
    parser.add_argument('--save_audio', type = bool, default=True, help='Save separated audio')
    parser.add_argument('--eval_sdr', type = bool, default=True, help='Compute SDR')
    parser.add_argument('--plot_spectrogram', type = bool, default=True, help='Plot spectrograms before/after separation')


    opt = parser.parse_args()
    
    return opt