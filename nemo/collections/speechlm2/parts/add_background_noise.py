import math
import os
import random
import warnings
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, Optional

import torch
from torch import Tensor
import numpy as np
from numpy.typing import NDArray
import soundfile as sf
import torchaudio.functional as AF

from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torch_audiomentations.utils.object_dict import ObjectDict

SUPPORTED_EXTENSIONS = (".wav",)

def _read_wav_segment(
    file_path: str | Path,
    offset_s: float,
    duration_s: float,
) -> tuple[np.ndarray, int]:
    """Fast WAV segment read using soundfile (no decoding warnings, supports seek)."""
    file_path = str(file_path)
    info = sf.info(file_path)
    sr = int(info.samplerate)
    if sr <= 0:
        raise RuntimeError(f"Invalid samplerate for {file_path}: {sr}")

    start_frame = max(0, int(round(float(offset_s) * sr)))
    num_frames = max(0, int(round(float(duration_s) * sr)))

    # Clamp to file length to avoid exceptions on edge offsets.
    if start_frame >= int(info.frames):
        return np.zeros((0,), dtype=np.float32), sr
    if num_frames <= 0:
        return np.zeros((0,), dtype=np.float32), sr
    num_frames = min(num_frames, int(info.frames) - start_frame)

    with sf.SoundFile(file_path, mode="r") as f:
        f.seek(start_frame)
        # always_2d=True for consistent channel handling
        audio = f.read(frames=num_frames, dtype="float32", always_2d=True)

    # Mixdown to mono
    if audio.shape[1] > 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    else:
        audio = audio[:, 0]

    return audio.astype(np.float32, copy=False), sr


def _resample_if_needed(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return wav
    # torchaudio resample expects float tensor on CPU; keep it there and move later.
    return AF.resample(wav, orig_freq=orig_sr, new_freq=target_sr)


def calculate_desired_noise_rms(clean_rms, snr):
    """
    Given the Root Mean Square (RMS) of a clean sound and a desired signal-to-noise ratio (SNR),
    calculate the desired RMS of a noise sound to be mixed in.
    Based on https://github.com/Sato-Kunihiko/audio-SNR/blob/8d2c933b6c0afe6f1203251f4877e7a1068a6130/create_mixed_audio_file.py#L20
    :param clean_rms: Root Mean Square (RMS) - a value between 0.0 and 1.0
    :param snr: Signal-to-Noise (SNR) Ratio in dB - typically somewhere between -20 and 60
    :return:
    """
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a)
    return noise_rms


def calculate_rms(samples: np.ndarray) -> float:
    """Root mean square of a 1D waveform (time axis flattened)."""
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean(x * x) + 1e-20))


def calculate_rms_without_silence(samples: NDArray[np.float32], sample_rate: int):
    """
    This function returns the rms of a given noise whose silent periods have been removed. This ensures
    that the rms of the noise is not underestimated. Is most useful for short non-stationary noises.
    """

    window = int(0.025 * sample_rate)

    # Need at least one full analysis window; otherwise windowing logic yields empty arrays.
    if samples.shape[-1] <= window:
        return calculate_rms(samples)

    rms_all_windows = np.zeros(samples.shape[-1] // window)
    current_time = 0

    while current_time < samples.shape[-1] - window:
        rms_all_windows[current_time // window] += calculate_rms(
            samples[current_time : current_time + window]
        )
        current_time += window

    rms_threshold = np.max(rms_all_windows) / 25

    # The segments with a too low rms are identified and discarded
    rms_all_windows = rms_all_windows[rms_all_windows > rms_threshold]
    if rms_all_windows.shape[-1] > 0:
        # Beware that each window must have the same number of samples so that this calculation of the rms is valid.
        return calculate_rms(rms_all_windows)
    else:
        # Handle edge case: No windows remain. This can happen if there was just one window before discarding.
        return calculate_rms(samples)


def convert_decibels_to_amplitude_ratio(decibels):
    return 10 ** (decibels / 20)


def find_audio_files(
    root_path,
    filename_endings=SUPPORTED_EXTENSIONS,
    traverse_subdirectories=True,
    follow_symlinks=True,
):
    """Return a list of paths to all audio files with the given extension(s) in a directory.
    Also traverses subdirectories by default.
    """
    root_path = os.path.abspath(str(root_path))
    if not os.path.isdir(root_path):
        return []

    file_paths = []

    for root, dirs, filenames in os.walk(root_path, followlinks=follow_symlinks):
        filenames = sorted(filenames)
        for filename in filenames:
            input_path = os.path.abspath(root)
            file_path = os.path.join(input_path, filename)

            if filename.lower().endswith(filename_endings):
                file_paths.append(Path(file_path))
        if not traverse_subdirectories:
            # prevent descending into subfolders
            break

    return file_paths


def find_audio_files_in_paths(
    paths: Sequence[Path] | Sequence[str] | Path | str,
    filename_endings=SUPPORTED_EXTENSIONS,
    traverse_subdirectories=True,
    follow_symlinks=True,
):
    """Return a list of paths to all audio files with the given extension(s) contained in the list or in its directories.
    Also traverses subdirectories by default.
    """

    file_paths = []

    if isinstance(paths, (list, tuple, set)):
        paths = list(paths)
    else:
        paths = [paths]

    for p in paths:
        if str(p).lower().endswith(SUPPORTED_EXTENSIONS):
            file_path = Path(os.path.abspath(p))
            file_paths.append(file_path)
        elif os.path.isdir(p):
            file_paths += find_audio_files(
                p,
                filename_endings=filename_endings,
                traverse_subdirectories=traverse_subdirectories,
                follow_symlinks=follow_symlinks,
            )
    return file_paths


class AddBackgroundNoise(BaseWaveformTransform):
    """Mix in another sound, e.g. a background noise. Useful if your original sound is clean and
    you want to simulate an environment where background noise is present.
    Can also be used for mixup, as in https://arxiv.org/pdf/1710.09412.pdf
    A folder of (background noise) sounds to be mixed in must be specified. These sounds should
    ideally be at least as long as the input sounds to be transformed. Otherwise, the background
    sound is tiled: a repeat score ``duration_input / duration_noise`` (in seconds) sets how many
    whole copies are concatenated (``ceil(score)``) before trimming to the input length, instead of
    repeatedly doubling the waveform.
    Note that the gain of the added noise is relative to the signal level in the input if the parameter noise_rms
    is set to "relative" (default option). This implies that if the input is completely silent, no noise will be added.
    Here are some examples of datasets that can be downloaded and used as background noise:
    * https://github.com/karolpiczak/ESC-50#download
    * https://github.com/microsoft/DNS-Challenge/
    """

    def __init__(
        self,
        sounds_path: Sequence[Path] | Sequence[str] | Path | str,
        min_snr_db: float = 3.0,
        max_snr_db: float = 30.0,
        noise_rms: Literal["relative", "absolute"] = "relative",
        min_absolute_rms_db: float = -45.0,
        max_absolute_rms_db: float = -15.0,
        noise_transform: Callable[[NDArray[np.float32], int], NDArray[np.float32]]
        | None = None,
        p: float = 0.5,
        lru_cache_size: int | None = None,
        cache_audio: bool = False,
        max_cache_items: int = 64,
        silence_aware_rms: bool = True,
    ):
        """
        :param sounds_path: A path or list of paths to audio file(s) and/or folder(s) with
            audio files. Can be str or Path instance(s). The audio files given here are
            supposed to be background noises.
        :param min_snr_db: Minimum signal-to-noise ratio in dB. Is only used if noise_rms is set to "relative"
        :param max_snr_db: Maximum signal-to-noise ratio in dB. Is only used if noise_rms is set to "relative"
        :param noise_rms: Defines how the background noise will be added to the audio input. If the chosen
            option is "relative", the RMS of the added noise will be proportional to the RMS of
            the input sound. If the chosen option is "absolute", the background noise will have
            an RMS independent of the RMS of the input audio file. The default option is "relative".
        :param min_absolute_rms_db: Is only used if noise_rms is set to "absolute". It is
            the minimum RMS value in dB that the added noise can take. The lower the RMS is,
            the lower the added sound will be. Default: -45.0
        :param max_absolute_rms_db: Is only used if noise_rms is set to "absolute". It is
            the maximum RMS value in dB that the added noise can take. Note that this value
            can not exceed 0. Default: -15.0
        :param noise_transform: A callable waveform transform (or composition of transforms) that
            gets applied to the noise before it gets mixed in. The callable is expected
            to input audio waveform (numpy array) and sample rate (int).
        :param p: The probability of applying this transform
        :param lru_cache_size: No longer supported as of audiomentations v0.43.0, because the cache has been removed.
            If this is set to any value other than None, a ValueError will be raised.
        :param silence_aware_rms: If True (default), RMS uses 25 ms non-overlapping windows and drops
            low-energy windows (matches original behavior). If False, uses plain RMS on the full
            segment — much faster on GPU and usually fine for stationary background noise (e.g. DNS).
        """

        # BaseWaveformTransform expects mode=..., p=... (positional p would bind to mode).
        super().__init__(mode="per_example", p=p, output_type="tensor")
        self.sounds_path = sounds_path
        self.sound_file_paths = find_audio_files_in_paths(self.sounds_path)
        self.sound_file_paths = [str(fp) for fp in self.sound_file_paths]

        assert len(self.sound_file_paths) > 0

        if min_snr_db > max_snr_db:
            raise ValueError("min_snr_db must not be greater than max_snr_db")
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

        if min_absolute_rms_db > max_absolute_rms_db:
            raise ValueError(
                "min_absolute_rms_db must not be greater than max_absolute_rms_db"
            )
        if max_absolute_rms_db > 0:
            raise ValueError("max_absolute_rms_db must not be greater than 0")
        self.min_absolute_rms_db = min_absolute_rms_db
        self.max_absolute_rms_db = max_absolute_rms_db

        self.noise_rms = noise_rms
        if lru_cache_size is not None:
            raise ValueError(
                "Passing lru_cache_size is no longer supported, as the cache has been removed (since v0.43.0)."
            )
        self.noise_transform = noise_transform
        # Cache per-file metadata to avoid repeated stat/parse on Lustre.
        self._duration_s: dict[str, float] = {}
        self._samplerate: dict[str, int] = {}

        # Optional decoded-audio cache (full file, mono float32, resampled to target SR).
        self._cache_audio = bool(cache_audio)
        self._max_cache_items = int(max_cache_items)
        # Key includes target sr since we resample before caching.
        self._audio_cache: OrderedDict[tuple[str, int], torch.Tensor] = OrderedDict()
        self.silence_aware_rms = bool(silence_aware_rms)

    def randomize_parameters(
        self,
        samples: Optional[Tensor] = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        super().randomize_parameters(
            samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )

        tp = self.transform_parameters
        # ``samples`` is the already-selected subset; parent only calls here when augmenting.

        num_samples = int(samples.shape[-1])
        sr = float(sample_rate or self.sample_rate or 0)
        if sr <= 0:
            raise RuntimeError("sample_rate is required for AddBackgroundNoise.randomize_parameters")

        tp["snr_db"] = random.uniform(self.min_snr_db, self.max_snr_db)
        tp["rms_db"] = random.uniform(self.min_absolute_rms_db, self.max_absolute_rms_db)
        file_idx = random.randint(0, len(self.sound_file_paths) - 1)
        tp["noise_file_path"] = self.sound_file_paths[file_idx]

        nfp = tp["noise_file_path"]
        if nfp not in self._duration_s:
            info = sf.info(nfp)
            sr_native = int(info.samplerate)
            frames = int(info.frames)
            self._samplerate[nfp] = sr_native
            self._duration_s[nfp] = float(frames) / float(sr_native) if sr_native > 0 else 0.0

        noise_duration = float(self._duration_s.get(nfp, 0.0))
        signal_duration = num_samples / sr

        min_noise_offset = 0.0
        max_noise_offset = max(0.0, noise_duration - signal_duration)

        tp["offset"] = random.uniform(min_noise_offset, max_noise_offset)
        tp["duration"] = signal_duration

    @staticmethod
    def _rms_torch(x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1).to(dtype=torch.float32)
        return torch.sqrt(torch.mean(x * x) + 1e-20)

    @classmethod
    def _rms_without_silence_torch(cls, x: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Match numpy version semantics, but keep on-device."""
        x = x.reshape(-1).to(dtype=torch.float32)
        window = int(0.025 * sample_rate)
        if x.numel() <= window or window <= 0:
            return cls._rms_torch(x)

        nwin = x.numel() // window
        if nwin <= 0:
            return cls._rms_torch(x)

        # Non-overlapping windows
        xw = x[: nwin * window].unfold(0, window, window)  # (nwin, window)
        rms_windows = torch.sqrt(torch.mean(xw * xw, dim=1) + 1e-20)  # (nwin,)
        thr = torch.max(rms_windows) / 25.0
        kept = rms_windows[rms_windows > thr]
        if kept.numel() > 0:
            # numpy implementation returns calculate_rms(rms_all_windows) => sqrt(mean(rms^2))
            return torch.sqrt(torch.mean(kept * kept) + 1e-20)
        return cls._rms_torch(x)

    def _get_noise_segment(
        self,
        file_path: str,
        target_sample_rate: int,
        offset_s: float,
        duration_s: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Load mono noise segment as torch tensor (on target device/dtype).

        If caching is enabled, we cache the FULL waveform resampled to target SR once,
        then slice segments cheaply. This avoids repeated Lustre reads + repeated resampling.
        """
        key = (file_path, int(target_sample_rate))

        if self._cache_audio and key in self._audio_cache:
            full_wav = self._audio_cache.pop(key)
            self._audio_cache[key] = full_wav  # mark as most-recent
            sr_native = int(target_sample_rate)
        elif self._cache_audio:
            # Load full wav once, resample once, keep on CPU float32.
            info = sf.info(file_path)
            sr0 = int(info.samplerate)
            with sf.SoundFile(str(file_path), mode="r") as f:
                audio = f.read(dtype="float32", always_2d=True)
            if audio.shape[1] > 1:
                audio = np.mean(audio, axis=1, dtype=np.float32)
            else:
                audio = audio[:, 0]
            wav = torch.from_numpy(np.asarray(audio, dtype=np.float32))  # CPU
            if wav.numel() > 0 and sr0 != target_sample_rate:
                wav = _resample_if_needed(wav, orig_sr=sr0, target_sr=target_sample_rate)
                sr_native = int(target_sample_rate)
            else:
                sr_native = sr0

            # Optional noise transform (expects numpy) - do once for cached audio.
            if self.noise_transform is not None and wav.numel() > 0:
                wav_np2 = wav.detach().cpu().numpy().astype(np.float32, copy=False)
                wav_np2 = self.noise_transform(wav_np2, int(sr_native))
                wav = torch.from_numpy(np.asarray(wav_np2, dtype=np.float32))

            full_wav = wav.contiguous()
            self._audio_cache[key] = full_wav
            while len(self._audio_cache) > self._max_cache_items:
                self._audio_cache.popitem(last=False)
        else:
            # No caching: load only needed segment (fast seek).
            wav_np, sr_native = _read_wav_segment(file_path, offset_s=offset_s, duration_s=duration_s)
            full_wav = torch.from_numpy(wav_np)  # actually segment wav on CPU
            if full_wav.numel() > 0 and sr_native != target_sample_rate:
                full_wav = _resample_if_needed(full_wav, orig_sr=sr_native, target_sr=target_sample_rate)
                sr_native = int(target_sample_rate)
            if self.noise_transform is not None and full_wav.numel() > 0:
                wav_np2 = full_wav.detach().cpu().numpy().astype(np.float32, copy=False)
                wav_np2 = self.noise_transform(wav_np2, int(sr_native))
                full_wav = torch.from_numpy(np.asarray(wav_np2, dtype=np.float32))

        # If caching is enabled, slice the cached full waveform. If not, full_wav is already the segment.
        if self._cache_audio:
            start = max(0, int(round(float(offset_s) * sr_native)))
            n = max(0, int(round(float(duration_s) * sr_native)))
            if n <= 0 or start >= int(full_wav.numel()):
                wav_seg = torch.zeros((0,), dtype=torch.float32)
            else:
                end = min(int(full_wav.numel()), start + n)
                wav_seg = full_wav[start:end]
        else:
            wav_seg = full_wav

        if wav_seg.numel() == 0:
            return torch.zeros((0,), device=device, dtype=dtype)

        return wav_seg.to(device=device, dtype=dtype, non_blocking=True)

    def _mix_mono_torch(self, clean: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Load/scale/tile noise and add it to a mono waveform tensor."""
        tp = self.transform_parameters
        device = clean.device
        dtype = clean.dtype

        noise = self._get_noise_segment(
            file_path=tp["noise_file_path"],
            target_sample_rate=sample_rate,
            offset_s=float(tp["offset"]),
            duration_s=float(tp["duration"]),
            device=device,
            dtype=dtype,
        )

        if noise.numel() == 0:
            warnings.warn("Loaded noise has zero length; returning the input unchanged.")
            return clean

        if self.silence_aware_rms:
            noise_rms = self._rms_without_silence_torch(noise, sample_rate)
        else:
            noise_rms = self._rms_torch(noise)
        if float(noise_rms) < 1e-9:
            warnings.warn(
                f"The file {tp['noise_file_path']} is too silent to be added as noise. Returning the input unchanged."
            )
            return clean

        if self.silence_aware_rms:
            clean_rms = self._rms_without_silence_torch(clean, sample_rate)
        else:
            clean_rms = self._rms_torch(clean)

        if self.noise_rms == "relative":
            desired_noise_rms = float(clean_rms) / (10 ** (float(tp["snr_db"]) / 20.0))
            noise = noise * (desired_noise_rms / (noise_rms + 1e-20))
        elif self.noise_rms == "absolute":
            desired_noise_rms_amp = 10 ** (float(tp["rms_db"]) / 20.0)
            gain = desired_noise_rms_amp / (noise_rms + 1e-20)
            noise = noise * gain

        num_samples = int(clean.numel())
        if noise.numel() < num_samples:
            # Tile using repeat (torch) rather than numpy tile
            n_copies = max(1, int(math.ceil(num_samples / max(int(noise.numel()), 1))))
            noise = noise.repeat(n_copies)
        if noise.numel() > num_samples:
            noise = noise[:num_samples]

        return clean + noise

    def apply_transform(
        self,
        samples: Optional[Tensor] = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        if samples is None:
            raise RuntimeError("AddBackgroundNoise.apply_transform requires samples")
        sr = int(sample_rate or self.sample_rate or 0)
        if sr <= 0:
            raise RuntimeError("sample_rate is required for AddBackgroundNoise.apply_transform")

        batch_size, num_channels, _num_samples = samples.shape
        out = samples.clone()
        # ``samples`` is only examples/channels chosen for this transform; mix all rows.

        for b in range(batch_size):
            for c in range(num_channels):
                clean = samples[b, c]
                out[b, c] = self._mix_mono_torch(clean, sr)

        return ObjectDict(
            samples=out,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )