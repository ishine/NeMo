#!/bin/bash
# Combine STT + TTS + RNNT using a local conda environment (no Docker).
# Activate your VoiceChat conda env before running this script
# (see README: Create the conda environment).

set -euo pipefail

NEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SPK_NAME=Aria
SEED=42
CONFIG_NAME=nemotron_voicechat_nano9b
GPU=0
CACHE=""
RESULTS_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stt-fsdp-ckpt) STT_FSDP_CKPT="$2"; shift 2 ;;
    --stt-ckpt-config) STT_CKPT_CONFIG="$2"; shift 2 ;;
    --tts-ckpt) TTS_CKPT="$2"; shift 2 ;;
    --rnnt-nemo) RNNT_NEMO="$2"; shift 2 ;;
    --speaker-wav) SPK_WAV="$2"; shift 2 ;;
    --speaker-name) SPK_NAME="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --step) STEP="$2"; shift 2 ;;
    --cache) CACHE="$2"; shift 2 ;;
    --results-root) RESULTS_ROOT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --config-name) CONFIG_NAME="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 1 ;;
  esac
done

for var in STT_FSDP_CKPT STT_CKPT_CONFIG TTS_CKPT RNNT_NEMO SPK_WAV OUTPUT_ROOT TAG STEP; do
  [[ -n "${!var:-}" ]] || { echo "ERROR: missing required argument for ${var}" >&2; exit 1; }
done
for path in "$STT_FSDP_CKPT" "$STT_CKPT_CONFIG" "$TTS_CKPT" "$RNNT_NEMO" "$SPK_WAV"; do
  [[ -e "$path" ]] || { echo "ERROR: path does not exist: $path" >&2; exit 1; }
done

command -v python >/dev/null || {
  echo "ERROR: python not found. Activate your VoiceChat conda env first" \
       "(see README: Create the conda environment)." >&2
  exit 1
}

CACHE="${CACHE:-${OUTPUT_ROOT}/cache}"
RESULTS_ROOT="${RESULTS_ROOT:-${OUTPUT_ROOT}/combined_ckpt_result_dir}"
MERGE_CONFIG_PATH="${NEMO_DIR}/examples/speechlm2/conf"

export CUDA_VISIBLE_DEVICES="${GPU#device=}"
export PYTHONPATH="${NEMO_DIR}:${PYTHONPATH:-}"
export HF_HOME="${CACHE}" TORCH_HOME="${CACHE}" NEMO_CACHE_DIR="${CACHE}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost
export WANDB_MODE=offline

TTS_NAME="$(basename "${TTS_CKPT}" .ckpt)"
STT_HF_CKPT="${STT_FSDP_CKPT%.ckpt}_hf"
HF_CKPT_NAME="${TAG}-step${STEP}-tts-eartts-${TTS_NAME}"
HF_EXPORT_DIR="${OUTPUT_ROOT}/${TAG}/s2s/${HF_CKPT_NAME}"
OUTPUT_DIR="${OUTPUT_ROOT}/${TAG}/s2s_rnnt/${HF_CKPT_NAME}"

mkdir -p "${CACHE}" "${RESULTS_ROOT}" "${OUTPUT_ROOT}"

# Step A needs a temporary config that uses the RNNT encoder architecture and
# loads weights from the supplied FSDP checkpoint instead of a warm start.
STT_CKPT_CONFIG_OVERRIDE="$(mktemp "${CACHE}/stt_exp_config.XXXXXX.yaml")"
cleanup() { rm -f "${STT_CKPT_CONFIG_OVERRIDE}"; }
trap cleanup EXIT

python - "${STT_CKPT_CONFIG}" "${STT_CKPT_CONFIG_OVERRIDE}" "${RNNT_NEMO}" <<'PY'
import pathlib
import re
import sys

source, destination, pretrained_asr = map(pathlib.Path, sys.argv[1:])
lines = source.read_text().splitlines(keepends=True)
asr_matches = s2s_matches = 0
for index, line in enumerate(lines):
    newline = "\n" if line.endswith("\n") else ""
    if re.match(r"^  pretrained_asr\s*:", line):
        lines[index] = f"  pretrained_asr: {pretrained_asr}{newline}"
        asr_matches += 1
    elif re.match(r"^  pretrained_s2s_model\s*:", line):
        lines[index] = f"  pretrained_s2s_model: null{newline}"
        s2s_matches += 1
if asr_matches != 1 or s2s_matches != 1:
    raise SystemExit(
        f"Expected one pretrained_asr and pretrained_s2s_model; got {asr_matches}, {s2s_matches}"
    )
destination.write_text("".join(lines))
PY

echo "============================================================"
echo " Python     : $(command -v python)"
echo " GPU        : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo " FSDP ckpt  : ${STT_FSDP_CKPT}"
echo " STT HF     : ${STT_HF_CKPT}"
echo " S2S HF     : ${HF_EXPORT_DIR}"
echo " Final out  : ${OUTPUT_DIR}"
echo "============================================================"

echo ""
echo "=== Step A: FSDP -> STT HF ==="
CONVERSION_DONE="${STT_HF_CKPT}/.conversion_done"
if [[ -f "${CONVERSION_DONE}" ]]; then
  echo "Already done: ${STT_HF_CKPT}"
else
  mkdir -p "${STT_HF_CKPT}"
  python "${NEMO_DIR}/examples/speechlm2/to_hf.py" \
    class_path=nemo.collections.speechlm2.models.DuplexSTTModel \
    ckpt_path="${STT_FSDP_CKPT}" \
    ckpt_config="${STT_CKPT_CONFIG_OVERRIDE}" \
    output_dir="${STT_HF_CKPT}"
  touch "${CONVERSION_DONE}"
fi

echo ""
echo "=== Step B1 pre-processing: convert TTS .ckpt to safetensors ==="
TTS_SAFETENSORS="${TTS_CKPT%.ckpt}.safetensors"
if [[ ! -f "${TTS_SAFETENSORS}" ]]; then
    python - "${TTS_CKPT}" "${TTS_SAFETENSORS}" <<'PY'
import sys, torch
from safetensors.torch import save_file
src, dst = sys.argv[1], sys.argv[2]
print(f"Converting {src} -> {dst}")
ckpt = torch.load(src, map_location="cpu")
save_file(ckpt["state_dict"], dst)
print("Done.")
PY
else
    echo "Safetensors already exists: ${TTS_SAFETENSORS}"
fi
TTS_CKPT="${TTS_SAFETENSORS}"

echo ""
echo "=== Step B1: STT HF + TTS -> S2S HF ==="
HF_CKPT_BASENAME="$(basename "${HF_EXPORT_DIR}")"
RESULTS_DIR="${RESULTS_ROOT}/${HF_CKPT_BASENAME}"
mkdir -p "${RESULTS_DIR}"
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')"

HYDRA_FULL_ERROR=1 TORCH_CUDNN_V8_API_ENABLED=1 \
python "${NEMO_DIR}/examples/speechlm2/nemotron_voicechat_infer_voice_lock.py" \
  --config-path="${MERGE_CONFIG_PATH}" \
  --config-name="${CONFIG_NAME}" \
  exp_manager.name="${HF_CKPT_BASENAME}" \
  exp_manager.explicit_log_dir="${RESULTS_DIR}" \
  ++model.stt.model.pretrained_s2s_model="${STT_HF_CKPT}" \
  ++model.speech_generation.model.pretrained_model="${TTS_CKPT}" \
  ++model.inference_speaker_name="${SPK_NAME}" \
  ++model.speech_generation.model.tts_config.backbone_config.sliding_window=7500 \
  ++model.speech_generation.model.tts_config.use_audio_prompt_frozen_projection=True \
  ++model.speech_generation.model.inference_guidance_scale=0.2 \
  ++model.speech_generation.model.inference_guidance_enabled=True \
  ++model.speech_generation.model.inference_top_p_or_k=0.95 \
  ++model.speech_generation.model.inference_noise_scale=0.001 \
  ++model.speech_generation.model.use_system_prompt=False \
  ++model.stt.model.eval_text_turn_taking=True \
  trainer.num_nodes=1 \
  data.train_ds.seed="${SEED}" \
  data.validation_ds.seed="${SEED}" \
  data.validation_ds.batch_size=1 \
  ++trainer.limit_val_batches=0.0 \
  ++trainer.precision=32 \
  ++trainer.max_steps=1 \
  ++trainer.val_check_interval=1 \
  ++hf_export_dir="${HF_EXPORT_DIR}" \
  ++model.stt.model.incremental_loading=True \
  ++model.stt.model.use_function_head=True \
  '++model.stt.model.override_tokens.bos_token="<s>"' \
  '++model.stt.model.override_tokens.eos_token="</s>"' \
  '++model.stt.model.override_tokens.pad_token="<SPECIAL_12>"' \
  '++model.stt.model.bos_token="<s>"' \
  '++model.stt.model.eos_token="</s>"' \
  '++model.stt.model.pad_token="<SPECIAL_12>"' \
  "+register_speaker_dict={${SPK_NAME}: ${SPK_WAV}}" \
  ++reinit_audio_prompt_frozen_projection=True

[[ -d "${HF_EXPORT_DIR}" ]] || { echo "ERROR: Step B1 output not found: ${HF_EXPORT_DIR}" >&2; exit 1; }

echo ""
echo "=== Step B2: S2S HF + RNNT -> final checkpoint ==="
python "${NEMO_DIR}/examples/speechlm2/combine_s2s_rnnt_checkpoint.py" \
  --s2s_checkpoint_dir "${HF_EXPORT_DIR}" \
  --rnnt_nemo_path "${RNNT_NEMO}" \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite

echo ""
echo "Done. Final checkpoint: ${OUTPUT_DIR}"
