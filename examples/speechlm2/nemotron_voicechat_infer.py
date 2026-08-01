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

import os

import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

from nemo.collections.speechlm2 import DataModule, DuplexS2SDataset
from nemo.collections.speechlm2.models.nemotron_voicechat import NemotronVoiceChat
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

# Same as used by HF save/load (config.json in checkpoint dir)
HF_CONFIG_NAME = "config.json"

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def _merge_stt_config_from_checkpoint(cfg, model_config):
    """
    S2T-style flow: load STT config from checkpoint (if HF dir), then merge
    inference overrides on top. Modifies model_config in place so that
    model.stt.model = merge(ckpt_config["model"], inference model.stt.model).
    """
    stt_ckpt = (
        OmegaConf.select(cfg, "model.stt.model.pretrained_s2s_model", default=None)
        or OmegaConf.select(cfg, "model.pretrained_s2s_model", default=None)
    )
    stt_ckpt = str(stt_ckpt) if stt_ckpt else None
    if not stt_ckpt or not os.path.isdir(stt_ckpt):
        return
    config_path = os.path.join(stt_ckpt, HF_CONFIG_NAME)
    if not os.path.isfile(config_path):
        logging.warning(
            "[nemotron_voicechat_infer] No %s in STT checkpoint dir %s; using inference config only.",
            HF_CONFIG_NAME,
            stt_ckpt,
        )
        return
    ckpt_config = OmegaConf.load(config_path)
    ckpt_model = ckpt_config.get("model", ckpt_config)
    if not hasattr(ckpt_model, "keys") and not isinstance(ckpt_model, dict):
        ckpt_model = OmegaConf.to_container(ckpt_model, resolve=True)
    else:
        ckpt_model = OmegaConf.to_container(OmegaConf.create(ckpt_model), resolve=True)
    inference_stt_model = model_config.get("model", {}).get("stt", {}).get("model", {})
    merged = OmegaConf.merge(OmegaConf.create(ckpt_model), OmegaConf.create(inference_stt_model))
    model_config["model"]["stt"]["model"] = OmegaConf.to_container(merged, resolve=True)
    # Match S2T: do not load pretrained LLM/ASR; we load full checkpoint weights instead
    model_config["model"]["stt"]["model"]["pretrained_weights"] = False
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logging.info(
            "[nemotron_voicechat_infer] Merged STT config: checkpoint=%s, inference overrides on top (S2T-style).",
            stt_ckpt,
        )


@hydra_runner(config_path="conf", config_name="s2s_duplex_speech_decoder")
def inference(cfg):
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.allow_tf32 = True
    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    OmegaConf.save(cfg, log_dir / "exp_config.yaml")

    with trainer.init_module():
        model_config = OmegaConf.to_container(cfg, resolve=True)
        # S2T-style: STT config = checkpoint config first, then inference YAML overrides
        _merge_stt_config_from_checkpoint(cfg, model_config)
        model = NemotronVoiceChat(model_config)

    # Save merged config (checkpoint + inference overrides) as soon as model is ready
    OmegaConf.save(OmegaConf.create(model_config), log_dir / "merged_config.yaml")
    logging.info("Saving merged config (from model_config used to build model) to %s", log_dir / "merged_config.yaml")

    # Log which checkpoints were loaded (rank 0 only)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        stt_ckpt = getattr(model.stt_model, "cfg", None) and model.stt_model.cfg.get("pretrained_s2s_model", None)
        tts_ckpt = getattr(model.tts_model, "cfg", None) and model.tts_model.cfg.get("pretrained_model", None)
        speaker_ref = getattr(model, "cfg", None) and model.cfg.get("inference_speaker_reference", None)
        logging.info("=" * 60)
        logging.info("[nemotron_voicechat_infer] Loaded checkpoints:")
        logging.info("  STT (pretrained_s2s_model): %s", stt_ckpt or "(none)")
        logging.info("  TTS (pretrained_model):     %s", tts_ckpt or "(none)")
        logging.info("  Speaker reference:         %s", speaker_ref or "(none)")
        logging.info("=" * 60)

    model_cfg = (
        OmegaConf.to_container(model.stt_model.cfg, resolve=True)
        if hasattr(model, "stt_model") and hasattr(model.stt_model, "cfg")
        else {}
    )

    dataset = DuplexS2SDataset(
        tokenizer=model.stt_model.tokenizer,
        frame_length=cfg.data.frame_length,
        source_sample_rate=cfg.data.source_sample_rate,
        target_sample_rate=cfg.data.target_sample_rate,
        input_roles=cfg.data.input_roles,
        output_roles=cfg.data.output_roles,
        include_turn_metadata=True,  # Enable detailed turn metadata for validation
        force_align_user_text=False,
        cfg=OmegaConf.to_container(cfg.data, resolve=True),
        model_cfg=model_cfg,
        early_interruption_prob=0.0,
    )
    datamodule = DataModule(cfg.data, tokenizer=model.stt_model.tokenizer, dataset=dataset)

    hf_export_dir = model_config.get("hf_export_dir", None)
    if hf_export_dir:
        model.save_pretrained(hf_export_dir, config=model_config)
        print("Hugging face compatible checkpoint saved at:", hf_export_dir)

    trainer.validate(model, datamodule)




if __name__ == "__main__":
    inference()
