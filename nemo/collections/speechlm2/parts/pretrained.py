# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from contextlib import contextmanager
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM
from nemo.utils import logging

def load_pretrained_nemo(cls, model_path_or_name: str):
    """
    Load pretrained NeMo 1.0 model (inheriting from ModelPT). Works with ASR, TTS, codec models.

    Setting ``pretrained_weights=False`` returns a model that has identical architecture with the checkpoint,
    but is randomly initialized.
    """
    if Path(model_path_or_name).exists() and model_path_or_name.endswith(".nemo"):
        return cls.restore_from(model_path_or_name)
    else:
        return cls.from_pretrained(model_path_or_name)


def load_pretrained_hf(model_path_or_name: str, pretrained_weights: bool = True, dtype=torch.float32):
    """
    Load pretrained HuggingFace AutoModelForCausalLM.

    Setting ``pretrained_weights=False`` returns a model that has identical architecture with the checkpoint,
    but is randomly initialized.
    """
    if pretrained_weights:
        return AutoModelForCausalLM.from_pretrained(model_path_or_name, torch_dtype=dtype)
    else:
        config = AutoConfig.from_pretrained(model_path_or_name)
        return AutoModelForCausalLM.from_config(config, torch_dtype=dtype)


@contextmanager
def move_embedding(model):
    """Temporarily restores the embedding layer into HF LLM. Supports LoRA models."""
    if isinstance(model.llm, PeftModel):
        model.llm.base_model.model.model.embed_tokens = model.embed_tokens
    else:
        model.llm.model.embed_tokens = model.embed_tokens
    yield
    if isinstance(model.llm, PeftModel):
        del model.llm.base_model.model.model.embed_tokens
    else:
        del model.llm.model.embed_tokens


def set_model_dict_for_partial_init(pretrained_dict, model_dict):
    # 1. filter out different size layers
    for k, v in list(pretrained_dict.items()):
        if k in model_dict and hasattr(model_dict[k], "numel") and v.numel() != model_dict[k].numel():
            del pretrained_dict[k]
            logging.info(" | > Layer with shape mismatach in the model definition: {}".format(k)) 
    # 2. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 3. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    logging.info(" | > {} / {} layers are restored.".format(len(pretrained_dict), len(model_dict)))
    return model_dict