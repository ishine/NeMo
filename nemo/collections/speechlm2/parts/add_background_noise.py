import math
import os
import random
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, Optional

import librosa
import torch
from torch import Tensor
import numpy as np
from numpy.typing import NDArray

from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torch_audiomentations.utils.object_dict import ObjectDict

SUPPORTED_EXTENSIONS = (
    ".aac",
    ".aif",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
)

# from torch_audiomentations.core.audio_loading_utils import load_sound_file

# from torch_audiomentations.core.utils import (
#     calculate_desired_noise_rms,
#     calculate_rms_without_silence,
#     convert_decibels_to_amplitude_ratio,
#     find_audio_files_in_paths,
# )

def load_sound_file(file_path, sample_rate, mono=True, resample_type="auto", offset = 0.0, duration = None):
    """
    Load an audio file as a floating point time series. Audio will be automatically
    resampled to the given sample rate.

    :param file_path: str or Path instance that points to a sound file
    :param sample_rate: If not None, resample to this sample rate
    :param mono: If True, mix any multichannel data down to mono, and return a 1D array
    :param resample_type: "auto" means use "kaiser_fast" when upsampling and "kaiser_best" when
        downsampling
    """
    file_path = str(file_path)
    samples, actual_sample_rate = librosa.load(
        str(file_path), sr=None, mono=mono, dtype=np.float32, offset = offset, duration = duration
    )

    if sample_rate is not None and actual_sample_rate != sample_rate:
        if resample_type == "auto":
            if librosa.__version__.startswith("0.8."):
                resample_type = (
                    "kaiser_fast" if actual_sample_rate < sample_rate else "kaiser_best"
                )
            else:
                resample_type = "soxr_hq"
        samples = librosa.resample(
            samples,
            orig_sr=actual_sample_rate,
            target_sr=sample_rate,
            res_type=resample_type,
        )
        warnings.warn(
            "{} had to be resampled from {} Hz to {} Hz. This hurt execution time.".format(
                str(file_path), actual_sample_rate, sample_rate
            )
        )

    actual_sample_rate = actual_sample_rate if sample_rate is None else sample_rate

    if mono:
        assert len(samples.shape) == 1
    return samples, actual_sample_rate


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
        self.time_info_arr = np.zeros(
            shape=(len(self.sound_file_paths),),
            dtype=np.float32,
        )
        self.time_info_arr.fill(-1.0)

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

        if self.time_info_arr[file_idx] == -1.0:
            self.time_info_arr[file_idx] = librosa.get_duration(path=tp["noise_file_path"])

        noise_duration = float(self.time_info_arr[file_idx])
        signal_duration = num_samples / sr

        min_noise_offset = 0.0
        max_noise_offset = max(0.0, noise_duration - signal_duration)

        tp["offset"] = random.uniform(min_noise_offset, max_noise_offset)
        tp["duration"] = signal_duration

    def _mix_mono_numpy(self, samples_np: np.ndarray, sample_rate: int) -> np.ndarray:
        """Load/scales/tiles noise and adds it to a mono float32 numpy waveform."""
        tp = self.transform_parameters
        noise_sound, _ = load_sound_file(
            tp["noise_file_path"],
            sample_rate,
            offset=tp["offset"],
            duration=tp["duration"],
        )

        if self.noise_transform:
            noise_sound = self.noise_transform(noise_sound, sample_rate)

        noise_rms = calculate_rms_without_silence(noise_sound, sample_rate)
        if noise_rms < 1e-9:
            warnings.warn(
                "The file {} is too silent to be added as noise. Returning the input"
                " unchanged.".format(tp["noise_file_path"])
            )
            return samples_np

        clean_rms = calculate_rms_without_silence(samples_np, sample_rate)

        if self.noise_rms == "relative":
            desired_noise_rms = calculate_desired_noise_rms(clean_rms, tp["snr_db"])
            noise_sound = noise_sound * (desired_noise_rms / noise_rms)

        if self.noise_rms == "absolute":
            desired_noise_rms_db = tp["rms_db"]
            desired_noise_rms_amp = convert_decibels_to_amplitude_ratio(desired_noise_rms_db)
            gain = desired_noise_rms_amp / noise_rms
            noise_sound = noise_sound * gain

        num_samples = len(samples_np)
        len_noise = len(noise_sound)
        if len_noise == 0:
            warnings.warn("Loaded noise has zero length; returning the input unchanged.")
            return samples_np

        duration_input_s = num_samples / float(sample_rate)
        duration_noise_s = len_noise / float(sample_rate)
        repeat_score = duration_input_s / max(duration_noise_s, 1e-9)

        if len_noise < num_samples:
            n_copies = max(1, int(math.ceil(repeat_score)))
            noise_sound = np.tile(noise_sound, n_copies)

        if len(noise_sound) > num_samples:
            noise_sound = noise_sound[0:num_samples]

        return samples_np + noise_sound

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
                wav_np = samples[b, c].detach().cpu().numpy().astype(np.float32)
                mixed = self._mix_mono_numpy(wav_np, sr)
                out[b, c] = torch.from_numpy(mixed).to(device=samples.device, dtype=samples.dtype)

        return ObjectDict(
            samples=out,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )