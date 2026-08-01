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

from nemo.collections.speechlm2 import DataModule, DuplexS2SDataset, DuplexSTTModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

HF_CONFIG_NAME = "config.json"


def _merge_model_config_from_checkpoint(cfg, model_config):
    """
    Load model config from HF checkpoint's config.json, then merge inference
    YAML overrides on top.  Also sets pretrained_weights=False so that
    __init__ skips downloading base LLM/ASR weights (the fine-tuned weights
    will be loaded from pretrained_s2s_model instead).
    """
    stt_ckpt = OmegaConf.select(cfg, "model.pretrained_s2s_model", default=None)
    stt_ckpt = str(stt_ckpt) if stt_ckpt else None
    if not stt_ckpt or not os.path.isdir(stt_ckpt):
        return
    config_path = os.path.join(stt_ckpt, HF_CONFIG_NAME)
    if not os.path.isfile(config_path):
        logging.warning(
            "[s2s_duplex_stt_infer] No %s in checkpoint dir %s; using inference config only.",
            HF_CONFIG_NAME,
            stt_ckpt,
        )
        return
    ckpt_config = OmegaConf.load(config_path)
    ckpt_model = ckpt_config.get("model", ckpt_config)
    ckpt_model = OmegaConf.to_container(OmegaConf.create(ckpt_model), resolve=True)
    inference_model = model_config.get("model", {})
    merged = OmegaConf.merge(OmegaConf.create(ckpt_model), OmegaConf.create(inference_model))
    model_config["model"] = OmegaConf.to_container(merged, resolve=True)
    model_config["model"]["pretrained_weights"] = False
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logging.info(
            "[s2s_duplex_stt_infer] Merged model config: checkpoint=%s, inference overrides on top.",
            stt_ckpt,
        )


@hydra_runner(config_path="conf", config_name="s2s_duplex_stt")
def inference(cfg):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.allow_tf32 = True
    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    OmegaConf.save(cfg, log_dir / "exp_config.yaml")

    with trainer.init_module():
        if cfg.ckpt_path and os.path.isdir(cfg.ckpt_path):
            # Hugging Face format via from_pretrained
            model = DuplexSTTModel.from_pretrained(cfg.ckpt_path)
            model.validation_save_path = os.path.join(log_dir, "validation_logs")
            if hasattr(cfg, "model") and hasattr(model, "cfg"):
                model.cfg = OmegaConf.merge(model.cfg, cfg.model)
            model_config = OmegaConf.to_container(model.cfg.model) if hasattr(model, 'cfg') and hasattr(model.cfg, 'model') else OmegaConf.to_container(cfg.model) if hasattr(cfg, 'model') else {}
        else:
            model_config = OmegaConf.to_container(cfg, resolve=True)
            _merge_model_config_from_checkpoint(cfg, model_config)
            model = DuplexSTTModel(model_config)
            model_config = OmegaConf.to_container(cfg.model) if hasattr(cfg, 'model') else {}

    # Save merged config (checkpoint + inference overrides) as soon as model is ready
    merged_config = OmegaConf.to_container(cfg, resolve=True)
    merged_config["model"] = OmegaConf.to_container(model.cfg, resolve=True)
    OmegaConf.save(OmegaConf.create(merged_config), log_dir / "merged_config.yaml")
    logging.info("Saving merged config (from loaded model.cfg) to %s", log_dir / "merged_config.yaml")

    dataset = DuplexS2SDataset(
        tokenizer=model.tokenizer,
        frame_length=cfg.data.frame_length,
        source_sample_rate=cfg.data.source_sample_rate,
        target_sample_rate=cfg.data.target_sample_rate,
        input_roles=cfg.data.input_roles,
        output_roles=cfg.data.output_roles,
        include_turn_metadata=True,  # Enable detailed turn metadata for validation
        force_align_user_text=False,
        early_interruption_prob=0.0,
        cfg=cfg.data, 
        model_cfg=model_config,
    )
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)

    trainer.validate(model, datamodule)




if __name__ == "__main__":
    inference()
