# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Minimal offline Nemotron VoiceChat inference from an HF-format checkpoint.

Requires this repository to precede any installed nemo_toolkit on PYTHONPATH:
    export PYTHONPATH=/path/to/Speech:$PYTHONPATH

Usage:
    python offline_voicechat_infer.py \\
        --checkpoint /path/to/hf_checkpoint \\
        --wav /path/to/input.wav \\
        --system-prompt "You are a helpful assistant." \\
        --output-dir /path/to/output
"""

from __future__ import annotations

import argparse

from nemo.collections.speechlm2.inference.utils.offline_voicechat import (
    build_model,
    encode_system_prompt,
    load_wav_16k_mono,
    run_offline_inference,
    save_offline_outputs,
)

DEFAULT_SYSTEM_PROMPT = (
    "You are an AI voice assistant developed by NVIDIA. "
    "Your name is NVIDIA Voice Chat. "
    "Answer in a spoken, conversational style rather than a written one. "
    "Do not repeat the same sentence over and over again. "
    "Start the conversation by greeting the user."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline Nemotron VoiceChat inference")
    parser.add_argument("--checkpoint", required=True, help="HF checkpoint directory (config.json + weights)")
    parser.add_argument("--wav", required=True, help="Input wav file (resampled to 16 kHz mono)")
    parser.add_argument("--output-dir", required=True, help="Directory for {input_stem}_output.txt, _output.wav, _combined.wav")
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt (defaults to the NVIDIA Voice Chat prompt)",
    )
    parser.add_argument("--device", default="cuda", help="Torch device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Checkpoint : {args.checkpoint}")
    print(f"WAV        : {args.wav}")
    print(f"Output dir : {args.output_dir}")
    print()

    print("Building model...")
    model = build_model(args.checkpoint, device=args.device)
    print("Model ready.")

    wav_1d, input_signal, input_signal_lens = load_wav_16k_mono(args.wav, device=args.device)
    prompt_tokens, prompt_token_lens = encode_system_prompt(model, args.system_prompt, device=args.device)

    print(f"Running offline inference on {wav_1d.shape[0] / 16000:.2f}s of audio...")
    result = run_offline_inference(
        model,
        input_signal=input_signal,
        input_signal_lens=input_signal_lens,
        prompt_tokens=prompt_tokens,
        prompt_token_lens=prompt_token_lens,
    )

    text = result.get("text", [""])[0]
    print(f"\n=== Generated text ===\n{text}\n")

    paths = save_offline_outputs(result, wav_1d, args.output_dir, args.wav)
    for name, path in paths.items():
        print(f"{name:6} -> {path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
