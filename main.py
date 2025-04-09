from data_loader import load_audio_signals, mix_signals
from source_separation import learn_nmf_dictionary, separate_sources_with_nmf
from utils import compute_sdr, compute_spectrogram, plot_spectrograms
from arguments import arguments

import soundfile as sf
from IPython.display import Audio, display
import os

def main():
    opt = arguments()

    # Load signals
    signals, sr = load_audio_signals(data_path=opt.data_path, sr=opt.sr)

    source_1 = signals['source_1']
    source_2 = signals['source_2']
    test_mixed = signals['test']

    # Compute SDR before separation
    if opt.eval_sdr:
        sdr_1 = compute_sdr(source_1, test_mixed)
        sdr_2 = compute_sdr(source_2, test_mixed)
        print(f"[Before Separation] SDR(source_1 vs test): {sdr_1:.2f} dB")
        print(f"[Before Separation] SDR(source_2 vs test): {sdr_2:.2f} dB")

    # Learn NMF dictionaries
    B1 = learn_nmf_dictionary(source_1, n_components=opt.n_components,
                              n_fft=opt.n_fft, hop_length=opt.hop_length)
    B2 = learn_nmf_dictionary(source_2, n_components=opt.n_components,
                              n_fft=opt.n_fft, hop_length=opt.hop_length)

    # Separate sources
    y1_est, y2_est = separate_sources_with_nmf(
        test_mixed, B1, B2, sr=sr,
        n_fft=opt.n_fft, hop_length=opt.hop_length
    )

    # Optional SDR after separation
    if opt.eval_sdr:
        sdr_y1 = compute_sdr(source_1, y1_est)
        sdr_y2 = compute_sdr(source_2, y2_est)
        print(f"[After Separation] SDR(source_1 vs estimate): {sdr_y1:.2f} dB")
        print(f"[After Separation] SDR(source_2 vs estimate): {sdr_y2:.2f} dB")

    # Save outputs
    if opt.save_audio:
        os.makedirs(opt.output_dir, exist_ok=True)
        sf.write(os.path.join(opt.output_dir, 'estimated_source_1.wav'), y1_est, sr)
        sf.write(os.path.join(opt.output_dir, 'estimated_source_2.wav'), y2_est, sr)
        print(f"✅ Separated audio saved in: {opt.output_dir}")

    # Play audio
    if opt.play_audio:
        print("🔊 Estimated Source 1")
        display(Audio(y1_est, rate=sr))
        print("🔊 Estimated Source 2")
        display(Audio(y2_est, rate=sr))


    if opt.plot_spectrogram:
        specs = [
            compute_spectrogram(signals['source_1'], sr, opt.n_fft, opt.hop_length),
            compute_spectrogram(signals['source_2'], sr, opt.n_fft, opt.hop_length),
            compute_spectrogram(test_mixed, sr, opt.n_fft, opt.hop_length),
            compute_spectrogram(y1_est, sr, opt.n_fft, opt.hop_length),
            compute_spectrogram(y2_est, sr, opt.n_fft, opt.hop_length)
        ]
        titles = [
            'Source 1 (train)',
            'Source 2 (train)',
            'Test Mixed Signal',
            'Estimated Source 1',
            'Estimated Source 2'
        ]
        plot_spectrograms(specs, titles, sr)



if __name__ == '__main__':
    main()
