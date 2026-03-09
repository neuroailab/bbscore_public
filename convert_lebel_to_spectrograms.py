"""
convert_lebel_to_spectrograms.py

Converts LeBel2023 audio stimuli into log-mel spectrograms compatible with SSAST
(ssast-base-frame-400). Run this AFTER pulling the latest bbscore_public changes
that updated the LeBel dataset.

SSAST expects:
  - 16 kHz mono audio
  - 128 mel filterbanks
  - Per-dataset mean/std normalization (AudioSet stats used by default, matching
    the ssast-base-frame-400 pretraining setup)
  - Output shape per clip: (num_frames, 128) — where num_frames = (duration_s * 100)
    since SSAST uses a 10ms frame shift

Usage:
    python convert_lebel_to_spectrograms.py \
        --data_dir $SCIKIT_LEARN_DATA/bbscore_data/lebel2023 \
        --output_dir $SCIKIT_LEARN_DATA/bbscore_data/lebel2023_spectrograms \
        [--target_sr 16000] \
        [--n_mels 128] \
        [--n_fft 400] \
        [--hop_length 160] \
        [--norm_mean -4.2677393] \
        [--norm_std 4.5689974]

After running, Lillian's SSAST wrapper should point its stimulus loader at
--output_dir to load pre-computed spectrograms instead of raw WAVs.
"""

import os
import argparse
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from tqdm import tqdm


# ── AudioSet normalization constants used during ssast-base-frame-400 pretraining ──
# Source: SSAST repo (MIT CSAIL), patch_based_finetuning.py
AUDIOSET_MEAN = -4.2677393
AUDIOSET_STD  =  4.5689974


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert LeBel2023 WAV stimuli to SSAST-compatible log-mel spectrograms."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to the LeBel2023 audio stimulus directory (contains .wav files)."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Where to save .npy spectrogram files (one per stimulus)."
    )
    parser.add_argument(
        "--target_sr", type=int, default=16000,
        help="Target sample rate. SSAST was pretrained at 16 kHz. (default: 16000)"
    )
    parser.add_argument(
        "--n_mels", type=int, default=128,
        help="Number of mel filterbanks. Must match SSAST model config. (default: 128)"
    )
    parser.add_argument(
        "--n_fft", type=int, default=1024,
        help="FFT window size in samples. 1024 gives 513 freq bins, sufficient for 128 mel filterbanks. (default: 1024)"
    )
    parser.add_argument(
        "--hop_length", type=int, default=160,
        help="Hop length in samples (10ms at 16kHz = 160). (default: 160)"
    )
    parser.add_argument(
        "--norm_mean", type=float, default=AUDIOSET_MEAN,
        help=f"Dataset mean for normalization. (default: {AUDIOSET_MEAN}, AudioSet stats)"
    )
    parser.add_argument(
        "--norm_std", type=float, default=AUDIOSET_STD,
        help=f"Dataset std for normalization.  (default: {AUDIOSET_STD}, AudioSet stats)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-process files even if output .npy already exists."
    )
    return parser.parse_args()


def build_mel_transform(target_sr: int, n_fft: int, hop_length: int, n_mels: int):
    """Returns a torchaudio MelSpectrogram transform."""
    return T.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=0.0,
        f_max=8000,
        power=2.0,            # power spectrogram → convert to dB below
        norm="slaney",        # matches librosa default used in SSAST pretraining
        mel_scale="htk",
    )


def wav_to_log_mel(
    wav_path: Path,
    mel_transform: T.MelSpectrogram,
    target_sr: int,
    norm_mean: float,
    norm_std: float,
) -> np.ndarray:
    """
    Load a WAV file and return a normalized log-mel spectrogram.

    Returns:
        np.ndarray of shape (num_frames, n_mels) — time-major, matching SSAST's
        expected input layout.
    """
    try:
        waveform, sr = torchaudio.load(str(wav_path))
    except Exception:
        import soundfile as sf
        audio_np, sr = sf.read(str(wav_path), always_2d=True)  # (samples, channels)
        waveform = torch.from_numpy(audio_np.T).float()  # (channels, samples)

    # ── 1. Convert to mono ──────────────────────────────────────────────────────
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # ── 2. Resample if needed ───────────────────────────────────────────────────
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    # ── 3. Compute mel spectrogram (power) ──────────────────────────────────────
    mel_spec = mel_transform(waveform)          # (1, n_mels, num_frames)

    # ── 4. Convert to log scale matching SSAST preprocessing ────────────────────
    # SSAST uses torch.log + epsilon, NOT AmplitudeToDB. AmplitudeToDB outputs
    # values in [-80, 0] dB which are incompatible with AudioSet norm constants.
    log_mel = torch.log(mel_spec + 1e-7)        # (1, n_mels, T)

    # ── 5. Normalize with dataset mean/std ──────────────────────────────────────
    log_mel = (log_mel - norm_mean) / (norm_std + 1e-8)

    # ── 6. Reshape to (num_frames, n_mels) ──────────────────────────────────────
    # SSAST patch embedding reads: (batch, time, freq)
    log_mel = log_mel.squeeze(0).T  # (T, n_mels)

    return log_mel.numpy().astype(np.float32)


def main():
    args = parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(data_dir.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(
            f"No .wav files found under {data_dir}. "
            "Did you run `git pull` to get the updated LeBel dataset and set "
            "SCIKIT_LEARN_DATA correctly?"
        )

    print(f"Found {len(wav_files)} WAV files in {data_dir}")
    print(f"Output directory: {output_dir}")
    print(
        f"Config: sr={args.target_sr}, n_mels={args.n_mels}, "
        f"n_fft={args.n_fft}, hop={args.hop_length}, "
        f"norm=({args.norm_mean:.4f}, {args.norm_std:.4f})"
    )

    mel_transform = build_mel_transform(
        target_sr=args.target_sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
    )

    skipped = 0
    errors  = []

    for wav_path in tqdm(wav_files, desc="Converting"):
        # Mirror subdirectory structure inside output_dir
        rel_path   = wav_path.relative_to(data_dir)
        out_path   = output_dir / rel_path.with_suffix(".npy")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            log_mel = wav_to_log_mel(
                wav_path=wav_path,
                mel_transform=mel_transform,
                target_sr=args.target_sr,
                norm_mean=args.norm_mean,
                norm_std=args.norm_std,
            )
            np.save(str(out_path), log_mel)
        except Exception as e:
            errors.append((wav_path, str(e)))
            print(f"\n  ERROR processing {wav_path.name}: {e}")

    # ── Summary ─────────────────────────────────────────────────────────────────
    converted = len(wav_files) - skipped - len(errors)
    print(f"\nDone. Converted: {converted}  |  Skipped (exist): {skipped}  |  Errors: {len(errors)}")

    if errors:
        print("\nFailed files:")
        for path, msg in errors:
            print(f"  {path.name}: {msg}")

    # Quick sanity check on one output
    sample_npy = next(output_dir.rglob("*.npy"), None)
    if sample_npy:
        arr = np.load(str(sample_npy))
        print(f"\nSample spectrogram shape: {arr.shape}  (expected: (num_frames, {args.n_mels}))")
        print(f"Value range: [{arr.min():.3f}, {arr.max():.3f}]  (should be ~[-2, 2] after normalization)")


if __name__ == "__main__":
    main()