import copy
import itertools
import json
import math
import os
import random
import re
import string
import tempfile
from collections import OrderedDict
from typing import List, Optional, Union

import hydra
import librosa
import numpy as np
import sacrebleu
import soundfile as sf
import torch
import torchaudio
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torchaudio.pipelines import SQUIM_SUBJECTIVE

from nemo.collections.asr.parts.utils.eval_utils import remove_punctuations
from nemo.collections.common.metrics import MetricStringToTorchMetric, TextMetricsSet
from nemo.collections.common.parts.utils import apply_rope_scaling, extend_instance
from nemo.collections.multimodal.speech_llm.models.modular_models import ModularAudioGPTModel
from nemo.collections.multimodal.speech_llm.modules.common.audio_text_generation_utils import generate
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import EmbeddingScalingMixin, get_specs
from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
from nemo.collections.nlp.modules.common.text_generation_utils import get_computeprob_response
from nemo.collections.nlp.modules.common.transformer import transformer_modules
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.collections.tts.modules import t5tts_transformer
from nemo.core.classes.module import NeuralModule
from nemo.utils import AppState, logging, model_utils

try:
    from megatron.core import InferenceParams, parallel_state, tensor_parallel
    from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
    from megatron.core.models.gpt import GPTModel as MCoreGPTModel
    from megatron.core.transformer.transformer_config import TransformerConfig

    try:
        from megatron.core.num_microbatches_calculator import (
            get_num_microbatches,
            reconfigure_num_microbatches_calculator,
        )

    except (ImportError, ModuleNotFoundError):
        logging.warning("Megatron num_microbatches_calculator not found, using Apex version.")
        from apex.transformer.pipeline_parallel.utils import (
            _reconfigure_microbatch_calculator as reconfigure_num_microbatches_calculator,
        )
        from apex.transformer.pipeline_parallel.utils import get_num_microbatches
    from megatron.core.packed_seq_params import PackedSeqParams

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False

default_inference_config = {'tokens_to_generate': 30}


class SumVocabParallelEmbedding(tensor_parallel.VocabParallelEmbedding):

    def __init__(
        self,
        proj_head_dims,
        include_proj=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.proj_head_dims = proj_head_dims
        self.include_proj = include_proj
        if include_proj:
            self.output_proj = tensor_parallel.ColumnParallelLinear(
                kwargs['embedding_dim'] * len(proj_head_dims),
                output_size=kwargs['embedding_dim'],
                config=kwargs['config'],
                init_method=kwargs['init_method'],
            )

    def forward(self, input_):

        if input_.ndim == 3:
            assert input_.shape[2] == len(self.proj_head_dims)
            input_ = input_.clone()
            for i in range(len(self.proj_head_dims)):
                # shuold consider the offset of previous projection heads
                input_[:, :, i] += sum(self.proj_head_dims[:i])
            assert input_.max() < sum(self.proj_head_dims)
        embeddings = super().forward(input_)
        if input_.ndim == 3:
            if self.include_proj:
                new_embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[1], -1)
                new_embeddings, _ = self.output_proj(new_embeddings)
                embeddings = embeddings[:, :, 0] + new_embeddings
            else:
                # sum the multi proj embeddings as the final embeddings
                embeddings = torch.sum(embeddings, axis=2)
        return embeddings


class SumMultiEmbedding(LanguageModelEmbedding):
    """Language model embeddings with multiple tokens at each time step. The embeddings of the tokens of the same time step will be computed separately and then be summed together."""

    def __init__(
        self,
        proj_head_dims,
        include_proj=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        del self.word_embeddings
        self.word_embeddings = SumVocabParallelEmbedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.config.hidden_size,
            init_method=self.config.init_method,
            reduce_scatter_embeddings=self.reduce_scatter_embeddings,
            config=self.config,
            proj_head_dims=proj_head_dims,
            include_proj=include_proj,
        )


class SpeechDecoder(NeuralModule):
    def __init__(
        self,
        speech_decoder_parms: DictConfig,
        lantent_dim: int,
        num_audio_codebooks: int,
        num_audio_tokens_per_codebook: int,
    ):
        super().__init__()
        self.use_input_cache = False
        self.speech_decoder_parms = speech_decoder_parms
        self.lantent_dim = lantent_dim
        self.num_audio_codebooks = num_audio_codebooks
        self.num_audio_tokens_per_codebook = num_audio_tokens_per_codebook
        # optional configs
        self.cfg_unconditional_prob = self.speech_decoder_parms.pop("cfg_unconditional_prob", None)
        self.cfg_scale = self.speech_decoder_parms.pop("cfg_scale", 2.5)
        self.cond_on_prev_audio_tokens = self.speech_decoder_parms.pop("cond_on_prev_audio_tokens", False)
        self.detach_input = self.speech_decoder_parms.pop("detach_input", False)

        # projection to adapt llm embeddings into the same shape of speech decoder expected input
        if lantent_dim != self.speech_decoder_parms["d_model"]:
            self.input_proj = nn.Linear(lantent_dim, self.speech_decoder_parms["d_model"])
        else:
            self.input_proj = None

        # instanciate T5-TTS decoder to full compatibility and potentialy load pretrained model
        self.t5_decoder = t5tts_transformer.Transformer(**self.speech_decoder_parms)

        # projection to predict audio codes
        self.final_proj = nn.Linear(
            self.speech_decoder_parms["d_model"], num_audio_codebooks * num_audio_tokens_per_codebook
        )

        # create embeddings for encode input tokens
        if self.cond_on_prev_audio_tokens:
            audio_embeddings = []
            for _ in range(self.num_audio_codebooks):
                audio_embeddings.append(
                    nn.Embedding(num_audio_tokens_per_codebook, self.speech_decoder_parms["d_model"])
                )

            self.audio_embeddings = nn.ModuleList(audio_embeddings)

    def forward(self, hidden_states, speech_mask, input_audio_tokens=None, return_raw_logits=False):
        # Megatron LLM parallel training returns T, B, F so reshape it
        # T, B, F = hidden_states.size()
        hidden_states = hidden_states.transpose(0, 1).contiguous()  # .reshape(B, T, F) # from [T, B, F] to [B, T, F]
        # input cache needed due our transformer kv cache implementation expect the whole left context
        if self.use_input_cache:
            if self.cache["hidden_states"] is None:
                self.cache["hidden_states"] = hidden_states
            else:
                self.cache["hidden_states"] = torch.cat([self.cache["hidden_states"], hidden_states], dim=1)
                hidden_states = self.cache["hidden_states"]

            if self.cache["speech_mask"] is None:
                self.cache["speech_mask"] = speech_mask
            else:
                self.cache["speech_mask"] = torch.cat([self.cache["speech_mask"], speech_mask], dim=1)
                speech_mask = self.cache["speech_mask"]

            if self.cache["input_audio_tokens"] is None:
                self.cache["input_audio_tokens"] = input_audio_tokens
            else:
                self.cache["input_audio_tokens"] = torch.cat(
                    [self.cache["input_audio_tokens"], input_audio_tokens], dim=1
                )
                input_audio_tokens = self.cache["input_audio_tokens"]

        if self.detach_input:
            hidden_states = hidden_states.detach()

        # map hidden states to the shape of the
        if self.input_proj is not None:
            speech_decoder_input = self.input_proj(hidden_states)
        else:
            speech_decoder_input = hidden_states

        # workaround for inference, because during inference speech_mask will be None
        if speech_mask is None:
            speech_mask = torch.ones((speech_decoder_input.size(0), speech_decoder_input.size(1))).to(
                speech_decoder_input.device
            )

        if self.cfg_unconditional_prob:
            if self.training:
                # if training drop the "text" conditioning in a percentage of batch
                if torch.rand(1).item() < self.cfg_unconditional_prob:
                    # make the whole batch zeros to the unconditional model
                    # ToDo: move it to cache to need to just create a 1 frame tensor in inference
                    speech_decoder_input = torch.zeros_like(speech_decoder_input)
            else:
                # if inference or evaluation create a zero tensor for speech decoder input and concatenate it to compute unconditional logits
                speech_decoder_input_zeros = torch.zeros_like(speech_decoder_input)
                speech_decoder_input = torch.cat([speech_decoder_input, speech_decoder_input_zeros], dim=0)
                # duplicate mask to match the new shape
                speech_mask = torch.cat([speech_mask, speech_mask], dim=0)
                # if cond on prev tokens enabled, so duplicate the tokens to the new shape
                if self.cond_on_prev_audio_tokens:
                    input_audio_tokens = torch.cat([input_audio_tokens, input_audio_tokens], dim=0)

        if self.cond_on_prev_audio_tokens:
            if self.detach_input:
                input_audio_tokens = input_audio_tokens.detach()

            audio_tokens_embedded = self.embed_audio_tokens(
                input_audio_tokens.transpose(1, 2).contiguous()
            )  # (B, T', E)
            speech_decoder_input = speech_decoder_input + audio_tokens_embedded

        decoder_out = self.t5_decoder(x=speech_decoder_input, x_mask=speech_mask)['output']

        # if it is true we need to return just the last autoregressive step, it is valid because for 1 frame input we produce 1 frame ouput
        if self.use_input_cache:
            decoder_out = decoder_out[:, -1:, :]

        # get the logits of all codebooks
        all_code_logits = self.final_proj(decoder_out)

        # if using cfg and it is in inference or evaluation mix unconditional and coditional logits
        if self.cfg_unconditional_prob and not self.training:
            batch_size = all_code_logits.size(0) // 2
            cond_logits = all_code_logits[:batch_size]
            uncond_logits = all_code_logits[batch_size:]
            all_code_logits = (1 - self.cfg_scale) * uncond_logits + self.cfg_scale * cond_logits

        if return_raw_logits:
            return all_code_logits

        # convert the logits from the single projection to a list with logits separated by codebook
        all_codebook_logits = self.all_logits_to_each_codebooks_logits(all_code_logits)

        return all_codebook_logits, all_code_logits

    def sample_codes_from_logits(self, all_code_logits_t, temperature=0.7, topk=80):
        # all_code_logits_t: (B, num_codebooks * num_tokens_per_codebook), logits at a given timestep
        all_preds = []
        for idx in range(self.num_audio_codebooks):
            si = idx * self.num_audio_tokens_per_codebook
            ei = si + self.num_audio_tokens_per_codebook
            codebook_logits = all_code_logits_t[:, si:ei]  # (B, num_tokens_per_codebook)
            codebook_logits_topk = torch.topk(codebook_logits, topk, dim=-1)[0]  # (B, topk)
            indices_to_remove = codebook_logits < codebook_logits_topk[:, -1].unsqueeze(
                -1
            )  # (B, num_tokens_per_codebook)
            codebook_logits_rescored = codebook_logits.clone()
            codebook_logits_rescored[indices_to_remove] = float('-inf')

            codebook_probs = torch.softmax(codebook_logits / temperature, dim=-1)  # (B, num_tokens_per_codebook)
            codebook_preds = torch.multinomial(codebook_probs, 1)  # (B, 1)
            all_preds.append(codebook_preds)
        all_preds = torch.cat(all_preds, dim=1).long()  # (B, num_codebooks)
        return all_preds

    def all_logits_to_each_codebooks_logits(self, logits):
        all_codebook_logits = []
        for idx in range(self.num_audio_codebooks):
            si = idx * self.num_audio_tokens_per_codebook
            ei = si + self.num_audio_tokens_per_codebook
            codebook_logits = logits[:, :, si:ei]  # (B, num_tokens_per_codebook)
            # B, T, F = codebook_logits.size()
            codebook_logits = codebook_logits.transpose(
                0, 1
            ).contiguous()  # .reshape(T, B, F) # transpose for compatibility with megatron format
            all_codebook_logits.append(codebook_logits)
        return all_codebook_logits

    def embed_audio_tokens(self, audio_tokens):
        # Add and average the embeddings of the audio tokens across the codebooks
        audio_embedding = None
        for c in range(self.num_audio_codebooks):
            embedding = self.audio_embeddings[c](audio_tokens[:, c, :])
            if audio_embedding is None:
                audio_embedding = embedding
            else:
                audio_embedding = audio_embedding + embedding
        audio_embedding = audio_embedding / audio_tokens.size(1)
        return audio_embedding

    def reset_input_and_kv_cache(self, use_cache):
        if use_cache:
            print("Enabling input and KV cache!")
        else:
            print("Disabling input and KV cache!")

        self.use_input_cache = use_cache
        self.cache = self._init_cache()
        self.t5_decoder.reset_cache(use_cache=use_cache)

    @staticmethod
    def _init_cache():
        return {
            'hidden_states': None,
            'speech_mask': None,
            'input_audio_tokens': None,
        }


# ToDo: if condition speech tokens on LLM-backbone does not bring good results, we should decouple speech decoder with MCoreGPTModel to avoid the unnecessary complexity
class S2sMCoreGPTModelSpeechDecoder(MCoreGPTModel):
    def __init__(
        self,
        config: TransformerConfig,
        proj_head_dims: List[int],
        proj_head_loss_weights: List[float],
        speech_decoder_parms: DictConfig = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(config=config, *args, **kwargs)
        self.n_proj_heads = len(proj_head_dims)
        self.proj_head_dims = proj_head_dims
        self.proj_head_loss_weights = proj_head_loss_weights
        self.speech_decoder_parms = dict(speech_decoder_parms) if speech_decoder_parms is not None else None

        num_audio_codebooks = len(self.proj_head_dims) - 1  # -1 to not consider the text channel
        num_audio_tokens_per_codebook = self.proj_head_dims[
            -1
        ]  # the first in the list is the vocab size of llm and the rest is the codec vocab, so get the last one for simplicity

        self.speech_decoder = SpeechDecoder(
            speech_decoder_parms=dict(self.speech_decoder_parms),
            lantent_dim=config.hidden_size,
            num_audio_codebooks=num_audio_codebooks,
            num_audio_tokens_per_codebook=num_audio_tokens_per_codebook,
        )

    def extend_embedding(self, vocab_size: int, include_proj=False):
        """Extend the embedding layer with new vocab size."""

        # Extend word embedding table if self.padded_vocab_size is larger than the size of the pre-trained word embedding
        pretrained_emb = self.embedding

        self.embedding = SumMultiEmbedding(
            config=self.config,
            vocab_size=vocab_size,
            max_sequence_length=self.max_sequence_length,
            position_embedding_type=self.position_embedding_type,
            proj_head_dims=self.proj_head_dims,
            include_proj=include_proj,
        )
        self.embedding.word_embeddings.weight.data[: pretrained_emb.word_embeddings.weight.shape[0]] = (
            pretrained_emb.word_embeddings.weight.data
        )
        # Zero out the new embeddings to make the model behave the same as it was pre-trained
        self.embedding.word_embeddings.weight.data[pretrained_emb.word_embeddings.weight.shape[0] :].zero_()
        del pretrained_emb
        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        speech_mask: Tensor = None,
        input_audio_tokens: Tensor = None,
    ) -> Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units
        """
        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        if self.position_embedding_type == 'rope':
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params, self.decoder, decoder_input, self.config
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )

        if not self.post_process:
            return hidden_states

        # logits and loss
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        else:
            output_weight = None

        # if text batch (VTBlender)
        if input_ids is not None and input_ids.dim() == 2:  # pure text example
            logits, _ = self.output_layer(
                hidden_states, weight=output_weight[: self.vocab_size] if output_weight is not None else None
            )

            if labels is None:
                # [s b h] => [b s h]
                return logits.transpose(0, 1).contiguous()

            loss = self.compute_language_model_loss(labels, logits)

            return loss

        else:
            # if speech batch
            # generate speech logits
            audio_logits, audio_logits_tensor = self.speech_decoder(
                hidden_states, speech_mask, input_audio_tokens=input_audio_tokens
            )

            # generate text logits
            text_logits, _ = self.output_layer(
                hidden_states, weight=output_weight[: self.vocab_size] if output_weight is not None else None
            )
            # create all logits by adding text_logits in 0 position and adding the audio logits list in the end
            all_logits = [text_logits] + audio_logits

            if labels is None:
                # [s b h] => [b s h]
                return_logits = [logits.transpose(0, 1).contiguous() for logits in all_logits]
                return torch.cat(return_logits, dim=-1)  # cat the last dim together to make other mcore code happy

            # compute loss
            tokens_loss = torch.stack(
                [self.compute_language_model_loss(labels[:, :, i], all_logits[i]) for i in range(self.n_proj_heads)],
                axis=2,
            )
            tokens_loss = (
                tokens_loss
                * torch.FloatTensor(self.proj_head_loss_weights).to(tokens_loss.device)
                / sum(self.proj_head_loss_weights)
            )

            return tokens_loss


class S2sModularAudioGPTModelSpeechDecoder(ModularAudioGPTModel):
    """S2S version of Modularized speech GPT model with Speech Decoder."""

    gpt_model_cls = S2sMCoreGPTModelSpeechDecoder

    def model_provider_func(self, pre_process, post_process):
        """Model depends on pipeline paralellism."""
        if self.mcore_gpt:
            if not hasattr(self.cfg, 'decoder_reduction_factor'):
                self.decoder_reduction_factor = 1
            else:
                self.decoder_reduction_factor = self.cfg.decoder_reduction_factor
            self.proj_head_dims = self.cfg.proj_head_dims
            self.proj_head_loss_weights = self.cfg.get('proj_head_loss_weights', [1.0])
            if self.decoder_reduction_factor != 1:
                if getattr(self.cfg, 'predict_source_text', False):
                    self.proj_head_dims = (
                        [self.proj_head_dims[0]]
                        + self.proj_head_dims[1:-1] * self.decoder_reduction_factor
                        + [self.proj_head_dims[-1]]
                    )
                    self.proj_head_loss_weights = (
                        [self.cfg.proj_head_loss_weights[0]]
                        + self.cfg.proj_head_loss_weights[1:-1] * self.decoder_reduction_factor
                        + [self.cfg.proj_head_loss_weights[-1]]
                    )
                else:
                    self.proj_head_dims = [self.proj_head_dims[0]] + self.proj_head_dims[
                        1:
                    ] * self.decoder_reduction_factor
                    self.proj_head_loss_weights = [
                        self.cfg.proj_head_loss_weights[0]
                    ] + self.cfg.proj_head_loss_weights[1:] * self.decoder_reduction_factor

            model = self.gpt_model_cls(
                config=self.transformer_config,
                transformer_layer_spec=get_specs(
                    self.spec_name,
                    self.transformer_config,
                    self.transformer_engine,
                    self.cfg.get('hyena', None),
                ),
                vocab_size=self.padded_vocab_size,  # later can be updated to s2s_vocab_size
                max_sequence_length=self.cfg.get('encoder_seq_length', 512),
                pre_process=pre_process,
                post_process=post_process,
                parallel_output=True,
                share_embeddings_and_output_weights=self.cfg.get('share_embeddings_and_output_weights', True),
                position_embedding_type=self.cfg.get('position_embedding_type', 'learned_absolute'),
                rotary_percent=self.cfg.get('rotary_percentage', 1.0),
                seq_len_interpolation_factor=self.cfg.get('seq_len_interpolation_factor', None),
                rotary_base=self.cfg.get('rotary_base', 10000),
                proj_head_dims=self.proj_head_dims,
                proj_head_loss_weights=self.proj_head_loss_weights,
                speech_decoder_parms=self.cfg.get('speech_decoder_parms', None),
            )

            if self.cfg.get('scale_positional_embedding', False):
                model.rotary_pos_emb.inv_freq = apply_rope_scaling(model.rotary_pos_emb.inv_freq)

            if self.cfg.get("apply_embedding_scaling", False) and parallel_state.is_pipeline_first_stage():
                extend_instance(model.embedding, EmbeddingScalingMixin)
        else:
            raise ValueError("S2S ModularAudioGPTModel requires Megatron-core GPT model.")
        return model

    def post_restore_from_pretrained_models(cls, model, cfg):

        codec_model, codec_model_cfg = cls.get_codec_models_and_configs(cfg)
        logging.info(f"Loaded Codec Model: {codec_model}")

        asr_model, asr_model_cfg = cls.get_asr_models_and_configs(cfg)
        logging.info(f"Loaded ASR Model: {asr_model}")

        mos_model = cls.get_mos_models_and_configs(cfg)
        logging.info(f"Loaded MOS Model: {mos_model}")

        def overwrite_state_dict_with_ckpt_path(ckpt_path, ignore=[], nemo_path='model_weights.ckpt'):
            if ckpt_path is not None:
                # this may only work for tp=1
                # check scripts/nlp_language_modeling/merge_lora_weights/merge_salm.py on tp>1
                salm_model_path = ckpt_path
                if '.nemo' in salm_model_path:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        NLPSaveRestoreConnector._unpack_nemo_file(salm_model_path, tmpdir)
                        salm_model_path = f"{tmpdir}/{nemo_path}"
                        torch_state_dict = torch.load(salm_model_path)
                else:
                    torch_state_dict = torch.load(salm_model_path)['state_dict']
                torch_state_dict = {k: v for k, v in torch_state_dict.items() if not any([i in k for i in ignore])}
                model.setup_complete = False
                model.load_state_dict(torch_state_dict, strict=False)
                logging.info(f"loading from {ckpt_path}: {torch_state_dict.keys()}")

        overwrite_state_dict_with_ckpt_path(cfg.model.get('salm_model_path'))
        overwrite_state_dict_with_ckpt_path(cfg.model.get('s2s_salm_model_path'), ignore=['model.'])

        model.padded_vocab_size = cfg.model.s2s_vocab_size

        if cfg.model.get('megatron_amp_O2', False):
            base_model = model.model.module
        else:
            base_model = model.model

        # if cond llm backbone on speech we need to expand the vocab and instance the sum embedding class
        if getattr(cfg, 'cond_llm_backbone_on_speech_tokens', True):
            base_model.extend_embedding(
                model.padded_vocab_size, include_proj=cfg.model.get('combine_emb_by_proj', False)
            )

        # print out params in more details
        model.summarize(max_depth=2)

        cls.codec_model = codec_model.cuda()
        cls.asr_model = asr_model.cuda()
        cls.mos_model = mos_model.cuda()

    @classmethod
    def restore_from_pretrained_models(
        cls,
        cfg: Optional[Union[OmegaConf, str]] = None,
        trainer: Optional[Trainer] = None,
    ):
        trainer.time_event_callback.logtimeevent.on_model_init_start()
        model = super().restore_from_pretrained_models(cfg, trainer)
        trainer.time_event_callback.logtimeevent.on_model_init_end()
        trainer.time_event_callback.logtimeevent.on_load_checkpoint_start()
        cls.post_restore_from_pretrained_models(cls, model, cfg)
        trainer.time_event_callback.logtimeevent.on_load_checkpoint_end()
        # memory
        cls.codec_model = cls.codec_model.to(torch.bfloat16)
        return model

    def load_state_dict(self, state_dict, strict: bool = True):
        try:
            super().load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            logging.info(f"Error loading model: {e} retrying with extend_embedding")
            with open_dict(self.cfg):
                self.cfg.model = self.cfg
            self.post_restore_from_pretrained_models(self, self.cfg)
            super().load_state_dict(state_dict, strict=strict)

    # change to add one more dimension
    def _shift_labels_by_emb_len(self, labels, label_lens, emb_lens, max_len, pad_token=0):
        """Shift labels to the right by the length of the audio embeddings."""
        shifted_labels = []
        for label, label_len, emb_len in zip(labels, label_lens, emb_lens):
            shifted_label = torch.full([max_len, label[0].shape[0]], pad_token, device=label.device)
            shifted_label[emb_len : emb_len + label_len] = label[:label_len]
            shifted_labels.append(shifted_label)
        shifted_labels = torch.stack(shifted_labels, dim=0)
        return shifted_labels

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: Optional[int] = None):
        """
        Used to get LLM predictions for validation and test steps based on the given inference config.
        """
        # TODO: we expect only one modality in each batch of inference. In lhotse, can we specify a list of datasets which only have one modality either audio-text or text-only?
        # TODO: support text-only part of mini-batch
        # the following supports STT (audio-text) inference

        # enable input and kv cache to the inference
        self.model.speech_decoder.reset_input_and_kv_cache(use_cache=True)

        inference_config = self.get_inference_config()
        if inference_config is not None:
            # need to overwrite some configuration, make it immutable
            inference_config = inference_config.copy()
        else:
            self.set_inference_config(inference_config=default_inference_config)
            logging.warning(f'inference_config is not set. Use default: {default_inference_config}')
            inference_config = self.get_inference_config()

        if self.cfg.data.get('end_string', None):
            inference_config['end_strings'] = [self.cfg.data.end_string]

        global_batch_size_per_gpu = batch['tokens'].size(0)
        num_micro_batches_before_decode = get_num_microbatches()

        compute_logprob = inference_config.get('compute_logprob', False)
        if compute_logprob:
            inference_config['inputs'] = batch
            inference_config['tokens_to_generate'] = 1
            inference_config['all_probs'] = True
            inference_config["add_BOS"] = False
            inference_config['greedy'] = True
            response = generate(self, **inference_config)
            response = get_computeprob_response(self.tokenizer, response, batch)
        else:
            # for megatron_gpt_eval.py
            if isinstance(batch, list):
                inference_config['inputs'] = batch
            elif 'num_audios' in batch:
                # peft_eval.py
                inference_config['inputs'] = (
                    batch['contexts'].cuda(),
                    batch['context_lengths'].cuda(),
                    batch['audio_signal'].cuda(),
                    batch['audio_signal_length'].cuda(),
                    batch['num_audios'].cuda(),
                    batch['context_start_idx'],
                )
            else:
                # peft_eval.py
                inference_config['inputs'] = (
                    batch['contexts'].cuda(),
                    batch['context_lengths'].cuda(),
                    batch['audio_signal'].cuda(),
                    batch['audio_signal_length'].cuda(),
                )

            response = generate(self, **inference_config)

        app_state = AppState()
        reconfigure_num_microbatches_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=global_batch_size_per_gpu * parallel_state.get_data_parallel_world_size(),
            micro_batch_size=global_batch_size_per_gpu // num_micro_batches_before_decode,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )

        # add audio offsets to context lengths for properly decoding only the response
        batch['context_lengths'] = batch['context_lengths'].cuda() + response['audio_feat_lens']

        # disable input and kv cache to the inference
        self.model.speech_decoder.reset_input_and_kv_cache(use_cache=False)
        return response

    def get_forward_output_only_func(self):
        def fwd_output_only_func(dataloader_iter, model):
            batch = next(dataloader_iter)
            extra_arg = {}
            # take the batch produced by prepare_batch_at_step
            (
                tokens,
                audiotokens2use,
                input_embeddings,
                attention_mask,
                position_ids,
                set_inference_key_value_memory,
                inference_max_sequence_len,
            ) = batch
            tokens = tokens.cuda()

            if attention_mask is not None:
                attention_mask = attention_mask.cuda()
                attention_mask = attention_mask[0:1]
            if self.mcore_gpt:
                # if first step, then clear KV cache, otherwise reuse inference_paarms
                if set_inference_key_value_memory[0].item():
                    self.inference_params = InferenceParams(
                        max_batch_size=tokens.size(0), max_sequence_length=inference_max_sequence_len[0].item()
                    )
                extra_arg['inference_params'] = self.inference_params
            else:
                extra_arg['set_inference_key_value_memory'] = set_inference_key_value_memory[0].item()
                extra_arg['inference_max_sequence_len'] = inference_max_sequence_len[0].item()

            # Currently for all MCore transformer layer specs causal attention mask
            # is used so we can delegate creating it to MCore/TE and pass None below
            if (
                isinstance(model, MCoreGPTModel)
                or hasattr(model, "module")
                and isinstance(model.module, MCoreGPTModel)
            ):
                attention_mask = None

            if self.megatron_amp_O2:
                input_embeddings = input_embeddings.type(self.model.module.embedding.word_embeddings.weight.dtype)
            output_tensor = model(
                input_ids=None,
                position_ids=None,
                decoder_input=input_embeddings,
                attention_mask=attention_mask,
                input_audio_tokens=audiotokens2use,
                **extra_arg,
            )

            # Advance inference sequence offset.
            if self.inference_params:
                # if last stage, then (final) output is [b, s, h], otherwise it's [s, b, h]
                if parallel_state.is_pipeline_last_stage():
                    self.inference_params.sequence_len_offset += output_tensor.size(1)
                else:
                    self.inference_params.sequence_len_offset += output_tensor.size(0)

            def id_func(output_tensor):
                return output_tensor, {'logits': output_tensor}

            return output_tensor, id_func

        return fwd_output_only_func

    def inference_step(self, dataloader_iter, mode):
        """
        Used for validation and test steps, added postprocessing after calling self.predict_step().
        """
        # Evaluation of multimodal data follows the same pattern as training except predict_step
        batch, batch_idx, dataloader_idx = next(dataloader_iter)
        data_cfg = self.cfg.data.validation_ds if mode == 'validation' else self.cfg.data.test_ds
        self._reconfigure_and_process_inference_batch(batch, data_cfg)
        # Meta data from dataset
        metadata = batch.get('metadata', [{}] * len(batch['tokens']))
        loss = super(MegatronGPTSFTModel, self).validation_step(itertools.chain([batch]), dataloader_idx)

        # make sure that the model is in eval mode
        self.eval()

        # We need _inference_config to get generation params
        # add_BOS and tokens_to_generate are set in dataset
        if self.get_inference_config() is None:
            logging.warning(f'inference_config is not set. Use default: {default_inference_config}')
            self.set_inference_config(inference_config=default_inference_config)
        self._inference_config['add_BOS'] = data_cfg.add_bos
        self._inference_config['tokens_to_generate'] = data_cfg.get('tokens_to_generate')

        output = self.predict_step(batch, batch_idx, dataloader_idx)

        inputs_text = (
            [self.tokenizer.ids_to_text(c.tolist()) for c in batch['instructions']]
            if batch['instructions'] is not None
            else [""] * len(batch['target_texts'])
        )
        labels_text = [self.tokenizer.ids_to_text(a.tolist()) for a in batch['target_texts']]
        # only do ids_to_text on the first channel which is text
        output['token_ids_text'] = (np.array(output['token_ids'])[:, :, 0]).tolist()
        preds_text = [
            self.tokenizer.ids_to_text(t[l.item() :][: data_cfg.get('tokens_to_generate')])
            for t, l in zip(output['token_ids_text'], batch['context_lengths'])
        ]

        if data_cfg.get("end_string", None):
            # sometimes data_cfg.end_string != self.tokenizer.ids_to_text(self.tokenizer.text_to_ids(data_cfg.end_string))
            # for example when data_cfg.end_string = "<end>", the end_string_re will start with " ?? "
            end_string_re = self.tokenizer.ids_to_text(self.tokenizer.text_to_ids(data_cfg.end_string))
            preds_text_cleaned = []
            labels_text_cleaned = []
            for p, l in zip(preds_text, labels_text):
                # remove end_string from the end of the string
                for es in [end_string_re, data_cfg.end_string]:
                    if p.endswith(es):
                        p = p[: -len(es)].strip()
                    if l.endswith(es):
                        l = l[: -len(es)].strip()
                preds_text_cleaned.append(p)
                labels_text_cleaned.append(l)
            # TODO: remove preds_text here since it is not used. the real preds_text is obtained by parse_decoder_outputs()
            preds_text = preds_text_cleaned
            labels_text = labels_text_cleaned

        if data_cfg.get("remove_text_pc", False):
            preds_text = [remove_punctuations(p.lower(), data_cfg.get("punctuations", None)) for p in preds_text]
            labels_text = [remove_punctuations(l.lower(), data_cfg.get("punctuations", None)) for l in labels_text]

        # if loss is nan, print the input, label and pred
        if loss.isnan():
            logging.info("++++++++++++++ NaN loss detected ++++++++++++++")
            for i in range(len(inputs_text)):
                logging.info(f"Input: `{inputs_text[i]}`")
                logging.info(f"Label: `{labels_text[i]}`")
                logging.info(f"Pred: `{preds_text[i]}`")
            logging.info("++++++++++++++++++++++++++++++++++++++++++++++++")

        outputs = {
            'loss': loss,
            'preds': output['token_ids'],
            'context_lengths': batch['context_lengths'],
            'labels': batch['answers'],  # [str]
            'labels_text': labels_text,  # [str]
            'inputs': inputs_text,  # [str]
            'metadata': metadata,  # [dict]
            'batch_idx': batch_idx,
            'audio_signal': batch.get('audio_signal', None),
            'system_prompts': batch.get('system_prompts', None),
            'system_prompts_length': batch.get('system_prompts_length', None),
        }

        if mode == 'validation':
            if len(self._validation_dl) > 1:
                # super().validation_step appends just loss to self.validation_step_outputs, replace the last appended loss with the outputs dict
                self.validation_step_outputs[dataloader_idx][-1] = outputs
            else:
                # super().validation_step appends just loss to self.validation_step_outputs, replace the last appended loss with the outputs dict
                self.validation_step_outputs[-1] = outputs
        else:
            if len(self._test_dl) > 1:
                self.test_step_outputs[dataloader_idx][-1] = outputs
            else:
                self.test_step_outputs[-1] = outputs

        # make sure that the model is in training mode after inference is done
        self.train()
        return outputs

    def post_inference_step(self, list_outputs, mode, data_cfg):
        # inference is done so make sure that input and KV cache is disabled
        self.model.speech_decoder.reset_input_and_kv_cache(use_cache=False)

        deduplicated_outputs = {
            'preds': [],
            'labels': [],
            'inputs': [],
            'metadata': [],
            'speech_preds': [],
            'speech_answers': [],
            'text_answers': [],
            'batch_idx': [],
        }
        for outputs in list_outputs:
            for answer, pred, input, metadata, labels_text, pred_context_length in zip(
                outputs['labels'],
                outputs['preds'],
                outputs['inputs'],
                outputs['metadata'],
                outputs['labels_text'],
                outputs['context_lengths'],
            ):
                context_length = 0
                batch_idx = outputs['batch_idx']
                text_answer, speech_answer = self.parse_decoder_outputs(
                    answer,
                    self.tokenizer.eos_id,
                    context_length,
                    self.cfg.data.train_ds.speech_pad_id,
                    self.cfg.data.train_ds.speech_eos_id,
                )
                key = input + self.tokenizer.ids_to_text(text_answer) + str(metadata)

                text_pred, speech_pred = self.parse_decoder_outputs(
                    torch.Tensor(pred),
                    self.tokenizer.eos_id,
                    pred_context_length,
                    self.cfg.data.train_ds.speech_pad_id,
                    self.cfg.data.train_ds.speech_eos_id,
                )

                def normalize_text(text):
                    return text.strip().replace('⁇', '')

                # TODO
                if speech_answer == None:
                    speech_answer = torch.zeros_like(speech_pred)
                text_pred_text = self.tokenizer.ids_to_text(text_pred)
                deduplicated_outputs['preds'].append(normalize_text(text_pred_text))
                deduplicated_outputs['labels'].append(normalize_text(labels_text))
                text_answer_text = self.tokenizer.ids_to_text(text_answer)
                deduplicated_outputs['text_answers'].append(normalize_text(text_answer_text))
                deduplicated_outputs['speech_preds'].append(speech_pred.cpu().numpy())
                deduplicated_outputs['speech_answers'].append(speech_answer.cpu().numpy())

                deduplicated_outputs['inputs'].append(input)
                deduplicated_outputs['metadata'].append(metadata)
                deduplicated_outputs['batch_idx'].append(batch_idx)

        # Compute metric score
        metric_name = self.val_metric_name if mode == 'validation' else self.test_metric_name
        metric = self.val_metric if mode == 'validation' else self.test_metric
        averaged_metric = [[] for _ in range(len(metric_name))]
        output_dir = data_cfg.get("output_dir", "./")
        run_codec = any(("asr" in metric_name or "mos" in metric_name) for metric_name in metric_name)
        run_asr = any("asr" in metric_name for metric_name in metric_name)
        run_mos = any("mos" in metric_name for metric_name in metric_name)

        # TODO: move the following model init code to init() function
        if run_codec:
            self.additional_models['codec_model'] = self.codec_model
            assert 'codec_model' in self.additional_models
            codec_model = self.additional_models['codec_model']
            codec_model.to(self.device)
            codec_model.eval()

            with torch.no_grad():
                logging.info(f"Decoding and saving audio")
                pred_wavs, start_end_time = self.decode_and_save_wavs(
                    codec_model,
                    deduplicated_outputs['speech_preds'],
                    os.path.join(output_dir, "wav", "pred"),
                    deduplicated_outputs['metadata'],
                )
                answer_wavs, _ = self.decode_and_save_wavs(
                    codec_model,
                    deduplicated_outputs['speech_answers'],
                    os.path.join(output_dir, "wav", "answer"),
                    deduplicated_outputs['metadata'],
                )

        if run_asr:
            self.additional_models['asr_model'] = self.asr_model
            assert 'asr_model' in self.additional_models
            asr_model = self.additional_models['asr_model']

            with torch.no_grad():
                if not self.cfg.get('segment_asr_decode', False):
                    logging.info(f"Running ASR on speech preds")
                    asr_batch_size = min(64, len(pred_wavs))
                    speech_preds_transcribed = asr_model.transcribe(pred_wavs, batch_size=asr_batch_size)
                    speech_answers_transcribed = asr_model.transcribe(answer_wavs, batch_size=asr_batch_size)
                else:
                    logging.info(f"Running ASR on segmented speech preds")
                    asr_batch_size = min(64, len(answer_wavs))
                    speech_answers_transcribed = asr_model.transcribe(answer_wavs, batch_size=asr_batch_size)
                    if isinstance(speech_answers_transcribed, tuple):
                        speech_answers_transcribed = speech_answers_transcribed[0]
                    speech_preds_transcribed = []
                    new_pred_wav = []
                    num_turns = []
                    max_length = 0
                    trans_new_pred_wav = []
                    for pred_wav, each_start_end_time in zip(pred_wavs, start_end_time):
                        if len(each_start_end_time) == 0:
                            num_turns.append(0)
                            continue
                        max_length = max(
                            max_length,
                            int(max([self.codec_sample_rate * (end - start) for start, end in each_start_end_time])),
                        )
                        num_turn = 0
                        for start, end in each_start_end_time:
                            start = int(self.codec_sample_rate * start)
                            end = int(self.codec_sample_rate * end)
                            if end > start:
                                num_turn += 1
                                trans_new_pred_wav.append(pred_wav[start:end])
                        num_turns.append(num_turn)
                    if len(trans_new_pred_wav) < 1:
                        trans_new_pred_wav = pred_wavs
                        logging.info(
                            f"Segmented speech preds are empty, using original speech preds. {deduplicated_outputs['metadata']}"
                        )
                    asr_batch_size = min(64, len(trans_new_pred_wav))
                    segmented_speech_preds_transcribed = asr_model.transcribe(
                        trans_new_pred_wav, batch_size=asr_batch_size
                    )
                    if isinstance(segmented_speech_preds_transcribed, tuple):
                        segmented_speech_preds_transcribed = segmented_speech_preds_transcribed[0]
                    prev_turns = 0
                    speech_preds_transcribed = []
                    for i, num_turn in enumerate(num_turns):
                        speech_preds_transcribed.append(
                            "                ".join(
                                [''] + segmented_speech_preds_transcribed[prev_turns : (prev_turns + num_turn)] + ['']
                            )
                        )
                        prev_turns += num_turn
                deduplicated_outputs['speech_preds_transcribed'] = speech_preds_transcribed
                deduplicated_outputs['speech_answers_transcribed'] = speech_answers_transcribed

        if run_mos:
            self.additional_models['squim_mos_model'] = self.mos_model
            assert 'squim_mos_model' in self.additional_models
            squim_mos_model = self.additional_models['squim_mos_model']
            codec_sample_rate = self.codec_sample_rate

            # TODO: use trans_new_pred_wav here too
            with torch.no_grad():
                if not self.cfg.get('segment_asr_decode', False):
                    logging.info(f"Running MOS prediction")

                else:
                    logging.info(f"Running MOS prediction on segmented speech preds")
                    pred_wavs_resampled = trans_new_pred_wav
                pred_wavs_resampled = [
                    torchaudio.functional.resample(wav.cuda(), codec_sample_rate, 16000).unsqueeze(0)
                    for wav in pred_wavs
                ]
                answer_wavs_resampled = [
                    torchaudio.functional.resample(wav.cuda(), codec_sample_rate, 16000).unsqueeze(0)
                    for wav in answer_wavs
                ]
                if self.cfg.get('segment_asr_decode', False):
                    squim_mos_scores = [
                        squim_mos_model(pred_wav, answer_wav.reshape([1, -1]).cuda()).cpu()
                        for pred_wav, answer_wav in zip(
                            pred_wavs_resampled, list_outputs[0]['audio_signal'][:1] * len(pred_wavs_resampled)
                        )
                    ]
                else:
                    squim_mos_scores = [
                        squim_mos_model(pred_wav, answer_wav).cpu()
                        for pred_wav, answer_wav in zip(pred_wavs_resampled, answer_wavs_resampled)
                    ]
                deduplicated_outputs['mos_scores'] = squim_mos_scores

        return deduplicated_outputs

    def parse_decoder_outputs(
        self, input_decoder_output, text_separator, context_length, speech_pad_id=1001, speech_eos_id=1004
    ):
        # remove text context
        max_len = input_decoder_output.shape[0]
        if len(input_decoder_output.shape) == 1:
            return input_decoder_output, None
        decoder_output = input_decoder_output[-1:].tile([max_len, 1])
        decoder_output[: max_len - context_length] = input_decoder_output[context_length:]

        # Do not split because text and speech are now aligned
        # Split text and speech part based on the position of the first separator token
        # sep_pos = (decoder_output[:, 0] == text_separator).long()
        # if torch.any(sep_pos):
        #     first_sep_pos = torch.argmax(sep_pos)
        #     text_tokens = decoder_output[:first_sep_pos, 0]
        #     speech_tokens = decoder_output[first_sep_pos + 1 :, 1:]
        # else:
        #     text_tokens = decoder_output[:, 0]
        #     speech_tokens = decoder_output[:, 1:]
        text_tokens = decoder_output[:, 0]
        if self.cfg.get('predict_source_text', False):
            speech_tokens = decoder_output[:, 1:-1]
        else:
            speech_tokens = decoder_output[:, 1:]
        # Get speech token ids
        if self.cfg.get('megatron_amp_O2', False):
            n_speech_codebooks = self.model.module.n_proj_heads - 1
        else:
            n_speech_codebooks = self.model.n_proj_heads - 1
        duplex_method = self.cfg.duplex_method
        if duplex_method != 'from_duplex':
            # Remove padded parts of speech tokens
            speech_eos_pos = torch.sum(speech_tokens == speech_eos_id, axis=1) == n_speech_codebooks
            speech_mask = torch.cumsum(speech_eos_pos, 0) == 0
            speech_tokens = speech_tokens[speech_mask]
        # Revert decoder output reduction
        new_shape = (
            speech_tokens.shape[0] * self.cfg.decoder_reduction_factor,
            speech_tokens.shape[1] // self.cfg.decoder_reduction_factor,
        )
        speech_tokens = speech_tokens.reshape(new_shape)
        if speech_tokens.shape[0] == 0:
            speech_tokens = torch.zeros([1, new_shape[1]]).long().cuda()
        return text_tokens.long(), speech_tokens.long()

    def decode_and_save_wavs(self, codec_model, codes_list, wav_dir, metadata_list):
        sample_rate = self.codec_sample_rate
        os.makedirs(wav_dir, exist_ok=True)
        wavs = []
        start_end_time = []
        for codes, metadata in zip(codes_list, metadata_list):
            codes = torch.tensor(codes).to(codec_model.device).T
            codec_len = torch.Tensor([codes.shape[1]]).long().to(codec_model.device)

            # get rid of bos and eos ids in the codec decoding
            def replace_speech_code(codes, id):
                return torch.where(codes == id, codes[:, :1], codes)

            def replace_speech_code_all(codes, id):
                return torch.where(codes == torch.ones_like(codes[:, :1]) * id, codes[:, :1], codes)

            def get_index_of_code(codes, id):
                # d, t
                idxs = torch.where(codes[0] == id)[0]
                return self.get_duration_by_steps(idxs)[0]

            # get start time of each turn
            start_times = get_index_of_code(codes, self.cfg.data.train_ds.speech_bos_id)
            codes = replace_speech_code(codes, self.cfg.data.train_ds.speech_bos_id)
            # get end time of each turn
            end_times = get_index_of_code(codes, self.cfg.data.train_ds.speech_eos_id)
            if len(start_times) == len(end_times) + 1:
                end_times = torch.cat(
                    [
                        end_times,
                        torch.full([1], self.get_duration_by_steps(codes.shape[1])[0], device=end_times.device),
                    ],
                    axis=0,
                )

            end_times = end_times[: len(start_times)]
            start_times = start_times[: len(end_times)]
            start_end_time.append([(s, e) for s, e in zip(start_times, end_times)])
            codes = replace_speech_code(codes, self.cfg.data.train_ds.speech_eos_id)
            codes = replace_speech_code(codes, self.cfg.data.train_ds.speech_unk_id)
            codes = replace_speech_code(codes, self.cfg.data.train_ds.speech_pad_id)
            codes = replace_speech_code_all(codes, 0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                wav, _ = codec_model.decode(tokens=codes.unsqueeze(0), tokens_len=codec_len)
            wav = wav[0].float()
            wavs.append(wav)
            sf.write(
                os.path.join(
                    wav_dir, re.sub("_repeat\d*", "", metadata['audio_filepath'].split('.wav')[0]) + ".gen.wav"
                ),
                wav.detach().cpu().numpy(),
                sample_rate,
            )

        return wavs, start_end_time

    def inference_epoch_end(self, outputs, mode, data_cfg):
        # TODO: use whisper normalizer https://pypi.org/project/whisper-normalizer/#:~:text=2%20by%20AssemblyAI-,How%20to%20use,over%20and%20pour%20me%20out%20'
        def normalize_text(text, table=str.maketrans('', '', string.punctuation), remove_extra_spaces=True):
            # remove punctuations and make it lower case
            text = text.translate(table).lower()
            if remove_extra_spaces:
                # remove extra spaces
                text = " ".join(text.split())
            return text

        # Parent class will handle logging of the loss.
        if not outputs or (all([not x for x in outputs])):
            return None

        if isinstance(outputs[0], dict):
            outputs = [outputs]

        averaged_loss = []
        # Log metrics for each provided validation/test dataset.
        for dataloader_idx, output in enumerate(outputs):
            if len(output) == 0:
                logging.warning(f"Empty output for dataloader_idx: {dataloader_idx}")
                continue
            # Expand on_validation_epoch_end from parent class MegatronGPTModel as on_validation_epoch_end doesnt take outputs arg
            loss_vals = [x['loss'] for x in output]
            if parallel_state.is_pipeline_last_stage():
                # only the last pipeline parallel stages return loss with their batch size
                if self.cfg.data.get('validation_drop_last', True):
                    loss = torch.stack(loss_vals).mean()
                else:
                    # Compute the avg loss by total_loss across all samples / total number of samples
                    total_loss_and_total_samples = torch.vstack(loss_vals).sum(axis=0)
                    avg_loss = total_loss_and_total_samples[0] / total_loss_and_total_samples[1]
                    loss = avg_loss.type(torch.float32).cuda()
            else:
                loss = torch.tensor(0.0, dtype=torch.float32).cuda()

            # we can only log on one rank if it is rank zero so we broadcast from last rank
            torch.distributed.broadcast(loss, get_last_rank())

            self.log('val_loss', loss, prog_bar=True, rank_zero_only=True, batch_size=1, sync_dist=True)

            # Determine the key used to log the loss based on the user provided name of the dataset or the dataloader index.
            loss_log_key = self._determine_log_key(data_cfg, dataloader_idx, "loss", mode)
            self.log(loss_log_key, loss, batch_size=1)
            averaged_loss.append(loss)

            output = self.post_inference_step(output, mode, data_cfg)

            # Gather the outputs object from all data parallel ranks since we are using the DistributedSampler which splits data across DDP ranks.
            gathered_outputs = [None for _ in range(parallel_state.get_data_parallel_world_size())]
            torch.distributed.all_gather_object(
                gathered_outputs,
                output,
                group=parallel_state.get_data_parallel_group(),
            )

            # Remove duplicate examples due to distributed sampler.
            inp_label_set = set()

            deduplicated_outputs = {}
            total_size = 0
            for rank in range(0, parallel_state.get_data_parallel_world_size()):
                for k, v in gathered_outputs[rank].items():
                    # TODO: add deduplication
                    if k not in deduplicated_outputs:
                        deduplicated_outputs[k] = []
                    deduplicated_outputs[k].extend(v)  # use extend for the b dim

            # Compute metric score
            metric_name = self.val_metric_name if mode == 'validation' else self.test_metric_name
            metric = self.val_metric if mode == 'validation' else self.test_metric
            averaged_metric = [[] for _ in range(len(metric_name))]

            if self.global_rank == 0:
                for (
                    labels,
                    text_answer_text,
                    preds,
                    speech_preds_transcribed,
                    speech_answer,
                    speech_pred,
                    inputs,
                    batch_idx,
                    speech_answers_transcribed,
                ) in zip(
                    deduplicated_outputs['labels'],
                    deduplicated_outputs['text_answers'],
                    deduplicated_outputs['preds'],
                    deduplicated_outputs['speech_preds_transcribed'],
                    deduplicated_outputs['speech_answers'],
                    deduplicated_outputs['speech_preds'],
                    deduplicated_outputs['inputs'],
                    deduplicated_outputs['batch_idx'],
                    deduplicated_outputs['speech_answers_transcribed'],
                ):
                    if (
                        data_cfg.get("log_every_n_steps", None) is not None
                        and batch_idx % data_cfg.log_every_n_steps == 0
                    ):
                        logging.info(f"Input: `{inputs}`")
                        logging.info(f"Label: `{labels}` text_answer_text: `{text_answer_text}`")
                        logging.info(f"Pred: `{preds}`")
                        logging.info(f"speech_preds_transcribed: `{speech_preds_transcribed}`")
                        logging.info(f"speech_answers_transcribed: `{speech_answers_transcribed}`")
                        logging.info(f"Speech out len: pred {speech_pred.shape} label {speech_answer.shape}")

            # Compute metric score
            for metric_name, metric_fn, averaged_metric in zip(metric_name, metric, averaged_metric):
                if metric_name != 'loss':
                    metric_log_key = self._determine_log_key(data_cfg, dataloader_idx, metric_name, mode)
                    labels = deduplicated_outputs['labels']
                    # sacrebleu.corpus_bleu is commonly used which does not share
                    # the same interface as other metrics. We handle it separately.
                    text_preds = deduplicated_outputs['preds']
                    if "asr-" in metric_name:
                        text_preds = deduplicated_outputs['speech_preds_transcribed']

                    text_metric_name = metric_name.replace("asr-", "")

                    def get_turn_split(input_preds, num_turn):
                        if all([t > num_turn for t in get_num_turn(input_preds)]):
                            return [re.split('   *', pred)[num_turn] for pred in input_preds]
                        else:
                            return input_preds

                    def get_num_turn(input_preds):
                        return [len(re.split('   *', pred)) for pred in input_preds]

                    if text_metric_name == 'bleu':  # asr-bleu, bleu
                        if self.cfg.get('norm_val_metrics', False):
                            # normalize texts
                            metric_text_preds = []
                            metric_labels = []
                            for pred, label in zip(text_preds, labels):
                                pred = normalize_text(pred)
                                label = normalize_text(label)
                                metric_text_preds.append(pred)
                                metric_labels.append(label)
                        else:
                            metric_text_preds = text_preds
                            metric_labels = labels

                        metric_result = torch.Tensor(
                            [sacrebleu.corpus_bleu(metric_text_preds, [metric_labels]).score]
                        ).to(self.device)
                    elif text_metric_name == 'wer':  # asr-wer, wer
                        for pred, label in zip(text_preds, labels):
                            # remove punctuationsa and extra spaces
                            if self.cfg.get('norm_val_metrics', False):
                                pred = normalize_text(pred)
                                label = normalize_text(label)
                            _ = metric_fn(pred, label)

                        metric_result = metric_fn.compute()
                        metric_fn.reset()
                    elif text_metric_name == "tts-wer":
                        for pred, label in zip(
                            deduplicated_outputs['speech_preds_transcribed'], deduplicated_outputs['preds']
                        ):
                            # remove punctuations and extra spaces
                            if self.cfg.get('norm_val_metrics', False):
                                pred = normalize_text(pred)
                                label = normalize_text(label)
                            _ = metric_fn(pred, label)

                        metric_result = metric_fn.compute()
                        metric_fn.reset()
                    elif metric_name == 'mos':
                        metric_result = sum(deduplicated_outputs['mos_scores']) / len(
                            deduplicated_outputs['mos_scores']
                        )
                    elif metric_name == 'bleu2':
                        if self.cfg.get('norm_val_metrics', False):
                            # normalize texts
                            metric_text_preds = []
                            metric_labels = []
                            for pred, label in zip(text_preds, labels):
                                pred = normalize_text(pred)
                                label = normalize_text(label)
                                metric_text_preds.append(pred)
                                metric_labels.append(label)
                        else:
                            metric_text_preds = text_preds
                            metric_labels = labels

                        metric_result = torch.Tensor(
                            [
                                sacrebleu.corpus_bleu(
                                    get_turn_split(metric_text_preds, 2), [get_turn_split(metric_labels, 2)]
                                ).score
                            ]
                        ).to(self.device)
                    elif metric_name == 'turndiff':
                        metric_result = torch.Tensor(
                            [np.abs(np.mean(np.subtract(get_num_turn(text_preds), get_num_turn(labels))))]
                        )
                    else:
                        for pred, label in zip(deduplicated_outputs['preds'], labels):
                            _ = metric_fn(pred, label)

                        metric_result = metric_fn.compute()
                        metric_fn.reset()

                    self.log(metric_log_key, metric_result.item(), sync_dist=True)
                    logging.info(f"{mode} {metric_name}: {metric_result.item()}")

                    averaged_metric.append(metric_result)

            # Write predictions to file
            if self.global_rank == 0 and data_cfg.get("write_predictions_to_file", False):
                logging.info(
                    f"Total deduplicated inference data size: {total_size} to {len(deduplicated_outputs['inputs'])}"
                )

                # Check if the user provided a prefix path to the file(s) they want to write.
                if not hasattr(data_cfg, "output_file_path_prefix") or data_cfg.output_file_path_prefix is None:
                    raise ValueError(
                        f"Cannot write predictions to file when output_file_path_prefix is not set or present in the yaml config file."
                    )
                filename_log_key = self._determine_log_key(data_cfg, dataloader_idx, None, mode)
                output_dir = data_cfg.get("output_dir", "./")
                self.write_predictions_to_file(
                    deduplicated_outputs, f"{data_cfg.output_file_path_prefix}_{filename_log_key}", output_dir
                )

            torch.distributed.barrier(group=parallel_state.get_data_parallel_group())
            outputs[dataloader_idx].clear()  # free memory

        # Logging of the averaged metrics:
        averaged_loss = sum(averaged_loss) / len(averaged_loss)
        averaged_metric = sum(averaged_metric) / len(averaged_metric) if len(averaged_metric) > 0 else None
        averaged_loss = averaged_loss.to(self.device)
        if averaged_metric is not None:
            averaged_metric = averaged_metric.to(self.device)

        # Handle case where metrics can be nan or inf. This can break checkpoint save/load.
        if averaged_metric is not None and (torch.isinf(averaged_metric) or torch.isnan(averaged_metric)):
            app_state = AppState()
            monitor_mode = app_state.checkpoint_callback_params.mode
            assert monitor_mode in ['min', 'max']
            averaged_metric = 0.0 if monitor_mode == 'max' else 1e5

        if mode == 'validation':
            self.log("validation_loss", averaged_loss, batch_size=1, sync_dist=True)
            if averaged_metric is not None:
                self.log(f"validation_{self.val_metric_name}", averaged_metric, sync_dist=True, batch_size=1)
        elif mode == 'test':
            self.log("test_loss", averaged_loss, batch_size=1, sync_dist=True)
            if averaged_metric is not None:
                self.log(f"test_{self.test_metric_name}", averaged_metric, sync_dist=True, batch_size=1)

        # Merge the functionality of previous on_inference_epoch_end() within inference_epoch_end() func here
        app_state = AppState()
        self._restore_activation_checkpointing_args()
        if hasattr(self, "_train_ds"):
            reconfigure_num_microbatches_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=self.cfg.data.train_ds.global_batch_size,
                micro_batch_size=self.cfg.data.train_ds.micro_batch_size,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )
        # When running `trainer.validate()`, the training dataset is not available.
        else:
            logging.warning('No training data found, reconfiguring microbatches based on validation batch sizes.')
            reconfigure_num_microbatches_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=data_cfg.global_batch_size,
                micro_batch_size=data_cfg.micro_batch_size,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )

        return averaged_loss, averaged_metric

    # consistent with speech models
    @rank_zero_only
    def write_predictions_to_file(self, outputs, output_file_path_prefix, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for folder_name in ['speech_pred', 'speech_answer', 'speaker_contexts']:
            os.makedirs(os.path.join(output_dir, 'npy', folder_name), exist_ok=True)
        output_file_path = output_file_path_prefix + "_inputs_preds_labels.jsonl"
        output_file_path = os.path.join(output_dir, output_file_path)
        with open(output_file_path, "w") as f_json:
            assert (
                len(outputs['inputs']) == len(outputs['preds']) == len(outputs['labels']) == len(outputs['metadata'])
            )
            for i, p, l, m, speech_preds_transcribed, speech_answers_transcribed in zip(
                outputs['inputs'],
                outputs['preds'],
                outputs['labels'],
                outputs['metadata'],
                outputs['speech_preds_transcribed'],
                outputs['speech_answers_transcribed'],
            ):
                json_string = {
                    'input': i,
                    'pred_text': p,
                    'text': l,
                    'speech_preds_transcribed': speech_preds_transcribed,
                    'speech_answers_transcribed': speech_answers_transcribed,
                }
                for k, v in m.items():
                    if k not in json_string:
                        json_string[k] = v
                f_json.write(json.dumps(json_string) + '\n')

        logging.info(f'Predictions saved to {output_file_path}')

    def de_concat_multiproj_logits(self, logits):
        logits_list = []
        prev = 0

        if self.cfg.get('megatron_amp_O2', False):
            base_model = self.model.module
        else:
            base_model = self.model

        for i in base_model.proj_head_dims:
            logits_list.append(logits[:, prev : prev + i])
            prev += i
        return logits_list

    def setup_metric(self, data_cfg):
        metric_name = "exact_string_match"
        if not hasattr(data_cfg, "metrics"):
            metrics = [(MetricStringToTorchMetric["exact_string_match"], "exact_string_match")]
        else:
            metrics = []
            for metric in data_cfg.metrics:
                if not hasattr(metric, "name"):
                    raise ValueError("Metric name is not provided in the metric config.")
                base_metric_name = metric.name.replace("asr-", "").replace("tts-", "")
                if metric.name == "loss" or metric.name == "mos":
                    metrics.append((None, metric.name))
                    continue
                if base_metric_name not in MetricStringToTorchMetric:
                    raise KeyError(
                        f"{metric.name} is not supported. List of supported metrics: {MetricStringToTorchMetric.keys()}"
                    )
                if base_metric_name in self._metrics_require_string2category_map:
                    if metric.average is None:
                        raise ValueError(
                            f"{metric.name} requires specifying whether you want to compute a micro or macro average. Found None."
                        )
                if (
                    metric.get('labels_are_strings', False)
                    and base_metric_name in self._metrics_require_string2category_map
                ):
                    if metric.num_classes is None:
                        raise ValueError(
                            "Number of classes is not provided in the metric section within the data config. "
                            f"Please provide the number of classes in the data config to use the {metric.name} metric."
                        )
                    if metric.get('class_labels', None) is None or not isinstance(
                        metric.get('class_labels', None), ListConfig
                    ):
                        raise ValueError(
                            "Class labels are not provided properly in the metric section witnin the data config. "
                            f"Please provide the class labels as a list of strings in the data config to use the {metric.name} metric."
                        )
                    if len(metric.get('class_labels', None)) != metric.num_classes:
                        raise ValueError(
                            f"Number of class labels {len(metric.get('class_labels', None))} does not match `num_classes` : {metric.num_classes}"
                        )

                metric_cls = MetricStringToTorchMetric[base_metric_name]
                if base_metric_name not in TextMetricsSet:
                    metric_fn = metric_cls(**data_cfg.metric)
                else:
                    metric_fn = metric_cls()
                metrics.append((metric_fn, metric.name))
        return zip(*metrics)

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        self.cfg = cfg
        self.additional_models = {}
        self.extract_codec_on_the_fly = cfg.get('extract_codec_on_the_fly', False)
        self.codec_model_downsampling_factor = cfg.get('codec_model_downsampling_factor', 1023.5)
        self.codec_sample_rate = cfg.data.train_ds.get("codec_sample_rate", 22050)
        self.speech_decoder_parms = cfg.get('speech_decoder_parms', None)
        super().__init__(cfg, trainer)
        if cfg.get('fixed_speaker_prompt', False):
            self.speaker_embeddings = nn.Embedding(16, cfg.hidden_size)
        self.model = self.model.to(torch.bfloat16)

    def _get_codec_embeddings(self, audio_signal, audio_signal_length):
        """Get codec embeddings for the input audio signal."""
        if 'codec_model' not in self.additional_models:
            self.additional_models['codec_model'] = self.codec_model
            self.additional_models['codec_model'].to(self.device)
            self.additional_models['codec_model'].eval()
        codec_model = self.additional_models['codec_model']
        codec_model.eval()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                original_codec_codes, _ = codec_model.encode(audio=audio_signal, audio_len=audio_signal_length)
                original_codec_codes = original_codec_codes.transpose(1, 2)
        out_codec_codes = []
        out_codec_lens = []
        n_speech_codebooks = original_codec_codes.shape[-1]
        decoder_reduction_factor = self.cfg.get("decoder_reduction_factor", 1)
        speech_pad_id = self.cfg.data.train_ds.speech_pad_id
        padded_original_codec_codes = torch.cat(
            [
                original_codec_codes,
                torch.ones([original_codec_codes.shape[0], decoder_reduction_factor, n_speech_codebooks]).long().cuda()
                * speech_pad_id,
            ],
            axis=1,
        )
        for sidx in range(audio_signal.shape[0]):
            codec_len = min(
                torch.ceil(audio_signal_length[sidx] / self.codec_model_downsampling_factor / decoder_reduction_factor)
                .int()
                .to(self.device),
                math.ceil(original_codec_codes[sidx].shape[0] / decoder_reduction_factor),
            )
            out_codec_codes.append(
                padded_original_codec_codes[sidx, : codec_len * decoder_reduction_factor]
                .reshape((-1, n_speech_codebooks * decoder_reduction_factor))
                .to(self.device)
            )
            out_codec_lens.append(codec_len)

        return out_codec_codes, out_codec_lens

    def get_duration_by_steps(self, steps):
        codec_model_downsampling_factor = torch.tensor(self.codec_model_downsampling_factor)
        decoder_reduction_factor = self.cfg.get("decoder_reduction_factor", 1)
        codec_sample_rate = self.codec_sample_rate
        seconds = steps * codec_model_downsampling_factor / codec_sample_rate * decoder_reduction_factor
        return seconds, (seconds * codec_sample_rate).int()

    def get_step_from_audio_len(self, audio_len):
        decoder_reduction_factor = self.cfg.get("decoder_reduction_factor", 1)
        return torch.ceil(audio_len / self.codec_model_downsampling_factor / decoder_reduction_factor).int() - 1

    def prepare_llm_input_duplex_from_multiturn(self, audio_batch):
        if self.cfg.get('noise_prob', 0.0) and random.random() < self.cfg.get('noise_prob', 0.0):
            if not (
                self.cfg.get("exclude_noise_on_s2s_duplex_overlap", False) and 's2s_duplex_overlap' in audio_batch
            ):
                self.add_noise_to_batch(
                    audio_batch,
                    os.path.join(self.cfg.noise_path, self.cfg.get('noise_path_name', 'train')),
                    random.randint(self.cfg.get('noise_min_snr', 10), self.cfg.get('noise_max_snr', 40)),
                )
        # real duplex data read from dataloader
        new_user_signal = audio_batch['audio_signal']
        new_user_signal_length = audio_batch['audio_signal_length']
        new_agent_signal = audio_batch['answer_audio']
        new_agent_signal_length = audio_batch['answer_audio_lens']
        loss_mask = None
        duplex_method = self.cfg.duplex_method
        assert duplex_method == "from_duplex"

        # [b, t, c]
        encoded, encoded_len = self.perception(
            input_signal=new_user_signal,
            input_signal_length=new_user_signal_length,
            processed_signal=None,
            processed_signal_length=None,
        )

        answer_codecs, answer_codecs_lens = self._get_codec_embeddings(
            new_agent_signal, new_agent_signal_length
        )  # list, list

        answer_codecs_lens = torch.Tensor(answer_codecs_lens).long().cuda()
        assert all(
            torch.isclose(answer_codecs_lens, encoded_len, atol=7)
        ), f"answer_codecs_lens: {answer_codecs_lens}, encoded_len: {encoded_len}, {new_agent_signal_length} {new_user_signal_length} {audio_batch['target_texts_merge']}"
        encoded_len = torch.minimum(answer_codecs_lens, encoded_len)
        answer_codecs_lens = encoded_len
        all_channels = []
        for i, answer_codec in enumerate(answer_codecs):
            text_channel = audio_batch['target_texts_merge'][i]
            sliced_text_channel = text_channel[: answer_codec.shape[0]].unsqueeze(-1)
            answer_codec = torch.where(
                sliced_text_channel == self.tokenizer.bos_id, self.cfg.data.train_ds.speech_bos_id, answer_codec
            )
            answer_codec = torch.where(
                sliced_text_channel == self.tokenizer.eos_id, self.cfg.data.train_ds.speech_eos_id, answer_codec
            )
            if getattr(self.cfg, 'predict_source_text', False):
                # Also use source_text
                source_text_channel = audio_batch['source_texts_merge'][i]
                sliced_source_text_channel = source_text_channel[: answer_codec.shape[0]].unsqueeze(-1)

            if getattr(self.cfg, 'predict_source_text', False):
                # TODO(kevinhu): Add delay to better predict user text.
                # Predict user text when the agent turn starts.
                all_channels.append(torch.cat([sliced_text_channel, answer_codec, sliced_source_text_channel], dim=-1))
            else:
                if getattr(self.cfg, 'speech_delay', False):
                    # TODO(kevinhu): Implement cascaded delays across all channels.
                    text_len, text_vocab = sliced_text_channel.shape
                    speech_len, speech_vocab = answer_codec.shape
                    assert text_len == speech_len
                    speech_pad_id = self.cfg.data.train_ds.speech_unk_id
                    text_pad_id = self.tokenizer.eos_id
                    answer_codec_padded = torch.full(
                        (self.cfg.speech_delay, speech_vocab), speech_pad_id, device=answer_codec.device
                    )
                    answer_codec_shifted = torch.cat([answer_codec_padded, answer_codec], dim=0)[:speech_len, :]
                    sliced_text_channel_padded = torch.full(
                        (self.cfg.speech_delay, text_vocab), text_pad_id, device=sliced_text_channel.device
                    )
                    sliced_text_channel_extended = torch.cat([sliced_text_channel, sliced_text_channel_padded], dim=0)[
                        :speech_len, :
                    ]
                    combined_channels = torch.cat([sliced_text_channel_extended, answer_codec_shifted], dim=-1)
                    all_channels.append(combined_channels)
                else:
                    # checked text_channel, loss_mask;  checked injecting bos and eos properly to control turn taking in inference
                    all_channels.append(torch.cat([sliced_text_channel, answer_codec], dim=-1))

        all_channels = pad_sequence(all_channels, batch_first=True)

        # inputs ids keep just the first channel (text channel)
        if not getattr(self.cfg, 'cond_llm_backbone_on_speech_tokens', True):
            input_ids = all_channels[:, :-1, 0]
        else:
            input_ids = all_channels[:, :-1]

        # get input audio tokens
        input_audio_tokens = all_channels[:, :-1, 1:]

        encoded = encoded[:, : input_ids.shape[1]]
        encoder_length = encoded_len - 1
        labels = all_channels[:, 1:]

        # assert labels.shape[1] == encoded.shape[1]
        # make sure that all inputs have the same number of frame as something it might deviate in 1 frame
        labels = labels[:, : encoded.shape[1]]
        input_ids = input_ids[:, : encoded.shape[1]]
        input_audio_tokens = input_audio_tokens[:, : encoded.shape[1]]

        loss_mask = torch.ones_like(labels)
        assert self.cfg.get(
            'duplex_loss_on_all_steps', False
        ), "only support duplex_loss_on_all_steps in real duplex data read from dataloader"
        # lookup input_ids
        if self.cfg.get('megatron_amp_O2', False):
            base_module = self.model.module
        else:
            base_module = self.model

        lm_embedding = (
            base_module.language_model.embedding if hasattr(base_module, 'language_model') else base_module.embedding
        )
        input_embeds = lm_embedding.word_embeddings(input_ids)
        # merge with encoded
        encoder_input = input_embeds + encoded * self.cfg.get("duplex_user_channel_weight", 0.3)

        scale_loss_mask_by = self.cfg.get("scale_loss_mask_by", None)
        if scale_loss_mask_by == 'bos_eos':
            for i, answer_codec in enumerate(answer_codecs):
                if 'target_texts_merge' in audio_batch:
                    text_channel = audio_batch['target_texts_merge'][i]
                    sliced_text_channel = text_channel[: loss_mask.shape[1]].unsqueeze(-1)
                    loss_mask = torch.where(sliced_text_channel == self.tokenizer.bos_id, 4.0, loss_mask)
                    loss_mask = torch.where(sliced_text_channel == self.tokenizer.eos_id, 4.0, loss_mask)
                else:
                    raise ValueError("scale_loss_mask_by=bos_eos is only supported for target_texts_merge")
        elif scale_loss_mask_by == 'non_sil':
            for i, answer_codec in enumerate(answer_codecs):
                if 'target_texts_merge' in audio_batch:
                    text_channel = audio_batch['target_texts_merge'][i]
                    sliced_text_channel = text_channel[: loss_mask.shape[1]].unsqueeze(-1)
                    loss_mask = torch.where(labels[:, :, :] != labels[:, :1, :], 4.0, loss_mask)
                else:
                    raise ValueError("scale_loss_mask_by=bos_eos is only supported for target_texts_merge")
        elif scale_loss_mask_by == 'agent_turn':
            # should exist in training loop; not in inference
            if 'agent_turns_merge' in audio_batch:
                loss_mask = torch.where(
                    audio_batch['agent_turns_merge'][:, : loss_mask.shape[1]].unsqueeze(-1) == 1, 4.0, loss_mask
                )
        elif scale_loss_mask_by == 'non_sil_t_agent_turn':
            # should exist in training loop; not in inference
            if 'agent_turns_merge' in audio_batch:
                loss_mask[:, :, 1:] = torch.where(
                    audio_batch['agent_turns_merge'][:, : loss_mask.shape[1]].unsqueeze(-1) == 1,
                    4.0,
                    loss_mask[:, :, 1:],
                )
                loss_mask[:, :, :1] = torch.where(labels[:, :, :1] != labels[i, :1, :1], 4.0, loss_mask[:, :, :1])
        elif scale_loss_mask_by == 'non_sil_st':
            if 'target_texts_merge' in audio_batch:
                loss_mask = torch.where(labels[:, :, :1] != labels[i, :1, :1], 4.0, loss_mask)
            else:
                raise ValueError("scale_loss_mask_by=bos_eos is only supported for target_texts_merge")
        elif scale_loss_mask_by == 'non_sil_t':
            if 'target_texts_merge' in audio_batch:
                loss_mask[:, :, :1] = torch.where(labels[:, :, :1] != labels[i, :1, :1], 4.0, loss_mask[:, :, :1])
            else:
                raise ValueError("scale_loss_mask_by=bos_eos is only supported for target_texts_merge")
        elif scale_loss_mask_by == None:
            pass
        else:
            raise ValueError(f"Unknown scale_loss_mask_by: {scale_loss_mask_by}")
        if self.cfg.get("exclude_speech_loss_on_s2s_duplex_overlap", False) and 's2s_duplex_overlap' in audio_batch:
            loss_mask[:, :, 1:] = 0.0
        limit_max_seq_length = self.cfg.get("limit_max_seq_length", None)
        if limit_max_seq_length is not None and limit_max_seq_length < labels.shape[1] and self.training:
            start = random.randint(0, labels.shape[1] - limit_max_seq_length - 1)
            encoder_input = encoder_input[:, start : start + limit_max_seq_length]
            labels = labels[:, start : start + limit_max_seq_length]
            loss_mask = loss_mask[:, start : start + limit_max_seq_length]
            encoder_length = torch.minimum(encoder_length, torch.tensor(limit_max_seq_length).long().cuda())
            encoded = encoded[:, start : start + limit_max_seq_length]

        encoder_input, labels, loss_mask, encoded, encoder_length = self.inject_speaker_prompt(
            audio_batch, encoder_input, labels, loss_mask, encoded, encoder_length
        )
        encoder_input, labels, loss_mask, encoded, encoder_length = self.inject_sys_prompt(
            audio_batch, encoder_input, labels, loss_mask, encoded, encoder_length
        )

        attention_mask = self._create_attention_mask(encoder_input)
        if not hasattr(lm_embedding, 'transpose_batch_sequence') or lm_embedding.transpose_batch_sequence:
            encoder_input = encoder_input.transpose(0, 1).contiguous()

        return encoder_input, attention_mask, labels, loss_mask, (encoded, encoder_length, input_audio_tokens)

    def inject_speaker_prompt(self, audio_batch, encoder_input, labels, loss_mask, encoded, encoder_length):
        fixed_speaker_prompt = self.cfg.get('fixed_speaker_prompt', False)
        if fixed_speaker_prompt == 1:  # concat
            speaker_ids = audio_batch['speaker_ids']
            speaker_embeds = self.speaker_embeddings(speaker_ids).unsqueeze(1)
            encoder_input = torch.cat([speaker_embeds, encoder_input], dim=1)
            labels = torch.cat([labels[:, :1], labels], dim=1)
            loss_mask = torch.cat([loss_mask[:, :1], loss_mask], dim=1)
            encoder_length += 1
            encoded = torch.cat([speaker_embeds, encoded], dim=1)
        elif fixed_speaker_prompt == 2:  # pool
            speaker_ids = audio_batch['speaker_ids']
            speaker_embeds = self.speaker_embeddings(speaker_ids).unsqueeze(1)
            encoder_input += speaker_embeds
            encoded += speaker_embeds
        return encoder_input, labels, loss_mask, encoded, encoder_length

    def inject_sys_prompt(self, audio_batch, encoder_input, labels, loss_mask, encoded, encoder_length):
        if 'system_prompts' in audio_batch:
            system_prompts = audio_batch['system_prompts']
            system_prompts_length = audio_batch['system_prompts_length']
            limit_max_seq_length = self.cfg.get("limit_context_max_seq_length", None)
            if limit_max_seq_length is not None and self.training:
                system_prompts = system_prompts[:, :limit_max_seq_length]
                system_prompts_length = torch.minimum(
                    system_prompts_length, torch.tensor(limit_max_seq_length).long().cuda()
                )
            embeddings2use = self._get_text_embeddings(system_prompts, None)
            # tmp simplified solution
            encoder_input = torch.cat([embeddings2use.transpose(1, 0), encoder_input], dim=1)
            labels = torch.cat(
                [torch.full(list(system_prompts.shape) + [labels.shape[2]], 0, device=labels.device), labels], dim=1
            )
            loss_mask = torch.cat(
                [torch.full(list(system_prompts.shape) + [labels.shape[2]], 0, device=labels.device), loss_mask], dim=1
            )
            encoder_length += system_prompts.shape[1]
            encoded = torch.cat([embeddings2use.transpose(1, 0), encoded], dim=1)
        return encoder_input, labels, loss_mask, encoded, encoder_length

    # TODO: move the following to dataloader
    def add_noise_to_batch(self, batch, noise_folder, snr_db=20):
        if self.cfg.get('debug_noise_audio', False):
            self.write_wave(
                batch['audio_signal'][0],
                "/lustre/fsw/portfolios/llmservice/users/zhehuaic/works/mod_speech_llm/tmp/dbg_original.wav",
            )

        batch_audio = batch['audio_signal'][:]  #  torch tensor，Shape: (batch_size, length)

        batch_size, audio_length = batch_audio.shape

        import glob

        noise_files = [f for f in glob.glob(noise_folder + "/*.wav")]
        if not noise_files:
            raise ValueError(f"No noise files found in {noise_folder}")

        for i in range(batch_size):

            def get_noise(noise_files):

                noise_path = random.choice(noise_files)
                noise, sr = sf.read(noise_path, dtype='float32')

                # resample noise from sr to self.cfg.data.train_ds.sample_rate
                if self.cfg.get('noise_resample', False) and sr != self.cfg.data.train_ds.sample_rate:
                    noise = librosa.resample(noise, orig_sr=sr, target_sr=self.cfg.data.train_ds.sample_rate)

                if len(noise.shape) > 1:
                    noise = np.mean(noise, axis=1)
                return noise

            noise = get_noise(noise_files)
            noise2 = get_noise(noise_files)
            noise = np.concatenate([noise, noise2], axis=0)
            if self.cfg.get('debug_noise_audio', False):
                self.write_wave(
                    torch.tensor(noise, dtype=batch_audio.dtype, device=batch_audio.device),
                    "/lustre/fsw/portfolios/llmservice/users/zhehuaic/works/mod_speech_llm/tmp/dbg_originalnoise.wav",
                )

            if len(noise) < audio_length:

                repeat_times = (audio_length // len(noise)) + 1
                noise = np.tile(noise, repeat_times)[:audio_length]
            else:

                start_idx = random.randint(0, len(noise) - audio_length)
                noise = noise[start_idx : start_idx + audio_length]

            noise_tensor = torch.tensor(noise, dtype=batch_audio.dtype, device=batch_audio.device)

            def get_scale_factor(signal, noise, snr_db):
                signal_power = torch.mean(signal**2) + 1e-8
                noise_power = torch.mean(noise**2) + 1e-8

                target_noise_power = signal_power / (10 ** (snr_db / 10))
                scaling_factor = torch.sqrt(target_noise_power / noise_power)
                return scaling_factor

            if random.random() < self.cfg.get('noise_prob_scale_user', 0.0):
                scaling_factor = get_scale_factor(
                    batch_audio[i],
                    batch_audio[i],
                    random.randint(
                        self.cfg.get('noise_prob_scale_user_min_snr', -15),
                        self.cfg.get('noise_prob_scale_user_max_snr', 24),
                    ),
                )
                batch_audio[i] = batch_audio[i] * scaling_factor
            scaling_factor = get_scale_factor(batch_audio[i], noise_tensor, snr_db)
            noise_tensor = noise_tensor * scaling_factor

            from scipy.signal import butter, lfilter

            # Function to create a low-pass filter
            def butter_lowpass(cutoff, fs, order=5):
                nyquist = 0.5 * fs
                normal_cutoff = cutoff / nyquist
                b, a = butter(order, normal_cutoff, btype='low', analog=False)
                return b, a

            # Function to apply the low-pass filter to data (tmp impl on cpu)
            def lowpass_filter(data, cutoff, fs, order=5):
                b, a = butter_lowpass(cutoff, fs, order=order)
                b = torch.tensor(b, dtype=torch.float32).cuda()
                a = torch.tensor(a, dtype=torch.float32).cuda()
                # Apply the filter using lfilter function from scipy..numpysig.numpynal (CPU)
                y_cpu = lfilter(b.cpu().numpy(), a.cpu().numpy(), data.cpu().numpy())
                # Convert the filtered data back to torch tensor and move to GPU.numpy
                y_gpu = torch.tensor(y_cpu, dtype=torch.float32).cuda()
                return y_gpu

            if random.random() < self.cfg.get('noise_prob_low_pass', 0.0):
                # Define the desired cutoff frequency (in Hz)
                cutoff = 1000.0
                # Apply low-pass filter to the WAV data
                noise_tensor = lowpass_filter(noise_tensor, cutoff, self.cfg.data.train_ds.sample_rate)

            batch_audio[i] = batch_audio[i] + noise_tensor

        if self.cfg.get('debug_noise_audio', False):
            self.write_wave(
                batch_audio[0], "/lustre/fsw/portfolios/llmservice/users/zhehuaic/works/mod_speech_llm/tmp/dbg_aug.wav"
            )
            self.write_wave(
                noise_tensor, "/lustre/fsw/portfolios/llmservice/users/zhehuaic/works/mod_speech_llm/tmp/dbg_noise.wav"
            )
            breakpoint()
        batch['audio_signal'] = batch_audio

    def write_wave(self, one_audio_signal, file_name):
        one_audio_signal = one_audio_signal.cpu().numpy()
        one_audio_signal = one_audio_signal.astype(np.float32)
        # one_audio_signal = np.clip(one_audio_signal, -1.0, 1.0)
        sf.write(file_name, one_audio_signal, self.cfg.data.train_ds.sample_rate)

    def prepare_llm_input(self, audio_batch):
        # handle duplex and singleturn s2s
        assert self.perception.cfg.preprocessor.sample_rate == self.cfg.data.train_ds.sample_rate
        duplex_method = self.cfg.duplex_method

        if duplex_method == 'from_duplex':
            # duplex data should go here
            assert 'target_texts_merge' in audio_batch
            return self.prepare_llm_input_duplex_from_multiturn(audio_batch)
        # the following branches are not used anymore
        elif duplex_method == 'from_multiturn':
            return self.prepare_llm_input_duplex_from_multiturn(audio_batch)
        elif duplex_method == None:
            pass
        else:
            raise ValueError(f"Unknown duplex method: {duplex_method}")

        # the following branch is used in single turn and multiturn but not duplex
        input_signal = audio_batch['audio_signal']
        logging.debug(f'input_signal.shape: {input_signal.shape}')
        input_signal_length = audio_batch['audio_signal_length']

        input_ids, input_length, labels, loss_mask = (
            audio_batch['tokens'],
            audio_batch['tokens_length'],
            audio_batch['labels'],
            audio_batch['loss_mask'],
        )
        context_lengths = audio_batch['context_lengths']

        if self.extract_codec_on_the_fly:
            answer_signal = audio_batch['answer_audio']
            answer_signal_length = audio_batch['answer_audio_lens']
            target_text_lengths = audio_batch['target_text_lengths']

            answer_codecs, answer_codecs_lens = self._get_codec_embeddings(
                answer_signal, answer_signal_length
            )  # list, list
            for i, answer_codec in enumerate(answer_codecs):
                base_length = target_text_lengths[i] + context_lengths[i]
                input_ids[i, base_length + 1 : base_length + 1 + answer_codecs_lens[i], 1:] = answer_codec
                labels[i, base_length : base_length + answer_codecs_lens[i], 1:] = answer_codec

        num_audios = audio_batch.get("num_audios", None)
        context_start_idx = audio_batch.get("context_start_idx", None)

        # [b, t, c]
        encoded, encoded_len = self.perception(
            input_signal=input_signal,
            input_signal_length=input_signal_length,
            processed_signal=None,
            processed_signal_length=None,
        )

        logging.debug(f'encoded.shape: {encoded.shape}')
        logging.debug(f'encoded_len.shape: {encoded_len.shape}')
        logging.debug(f'num_audios: {num_audios}')
        if num_audios is not None:
            # split the encoded and encoded_len by num_audios, used when there're multiple audio files per sample
            encoded = encoded.split(num_audios.tolist())
            encoded_len = encoded_len.split(num_audios.tolist())

        encoder_input, attention_mask, encoder_length, _, encoder_max_length = self.inject_perception_input(
            encoded, encoded_len, input_ids, input_length, context_start_idx
        )
        if num_audios is not None:
            # sum up the audio_feat_lens for each sample in the batch
            encoded_len = torch.stack([torch.sum(lens) for lens in encoded_len])

        # Shift labels to the right
        labels = self._shift_labels_by_emb_len(labels, input_length, encoded_len, encoder_max_length, pad_token=0)
        # Loss mask where answer tokens are 1.0 and all other tokens are 0.0
        loss_mask = self._shift_labels_by_emb_len(
            loss_mask, input_length, encoded_len, encoder_max_length, pad_token=0
        )
        # return if it is single turn or duplex
        if (
            all(audio_batch['num_turns'] == 2) or 'target_texts_merge' in audio_batch
        ):  # real duplex data read from dataloader
            return encoder_input, attention_mask, labels, loss_mask, encoder_length
        # special logic to handle multiturn half-duplex s2s
        # use num_turns to recover multiturn format and then merge them back to one sequence as LLM input/output
        new_encoder_input = []
        new_labels = []
        new_loss_mask = []
        new_encoder_length = []
        cnt = 0
        for num_turns in audio_batch['num_turns']:
            tmp_encoder_input = []
            tmp_labels = []
            tmp_loss_mask = []
            tmp_encoder_length = []
            for i in range(0, num_turns, 2):
                input_len = encoder_length[cnt]
                if i != num_turns - 2:  # last turn
                    input_len -= 1  # remove the last token as it is eos between the turns
                tmp_encoder_input.append(encoder_input.transpose(0, 1)[cnt][:input_len])
                tmp_labels.append(labels[cnt][:input_len])
                tmp_loss_mask.append(loss_mask[cnt][:input_len])
                tmp_encoder_length.append(input_len)
                cnt += 1
            new_encoder_input.append(torch.cat(tmp_encoder_input, dim=0))
            new_encoder_length.append(sum(tmp_encoder_length))
            new_labels.append(torch.cat(tmp_labels, dim=0))
            new_loss_mask.append(torch.cat(tmp_loss_mask, dim=0))
        new_encoder_input = pad_sequence(new_encoder_input, batch_first=True)
        new_encoder_length = torch.Tensor(new_encoder_length).long()
        new_labels = pad_sequence(new_labels, batch_first=True)
        new_loss_mask = pad_sequence(new_loss_mask, batch_first=True)
        assert cnt == encoder_length.shape[0]
        new_attention_mask = self._create_attention_mask(new_encoder_input)
        return (
            new_encoder_input.transpose(0, 1).contiguous(),
            new_attention_mask,
            new_labels,
            new_loss_mask,
            new_encoder_length,
        )

    def _gpt_forward(
        self,
        input_ids,
        position_ids,
        encoder_input,
        attention_mask,
        labels,
        checkpoint_activations_all_layers,
        speech_mask=None,
        input_audio_tokens=None,
    ):
        """Forward pass of the GPT model."""
        if self.megatron_amp_O2:
            encoder_input = encoder_input.type(self.model.module.embedding.word_embeddings.weight.dtype)
        if self.mcore_gpt:
            output = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                decoder_input=encoder_input,
                attention_mask=attention_mask,
                labels=labels,
                speech_mask=speech_mask,
                input_audio_tokens=input_audio_tokens,
            )
        else:
            output = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                encoder_input=encoder_input,
                attention_mask=attention_mask,
                labels=labels,
                checkpoint_activations_all_layers=checkpoint_activations_all_layers,
                speech_mask=speech_mask,
                input_audio_tokens=input_audio_tokens,
            )
        return output

    def forward(
        self,
        batch,
        checkpoint_activations_all_layers,
    ):
        """
        Forward pass of the model. We prepend audio embeddings to the instruction and label text tokens as the LLM input.
        """
        audio_batch = {k: v for k, v in batch.items() if not k.startswith("text_")}
        text_batch = {k: v for k, v in batch.items() if k.startswith("text_")}

        output, loss_mask = None, None

        multimodal_output = {}
        if 'audio_signal' in audio_batch:
            # in this branch, limit_max_seq_length is handled in prepare_llm_input
            encoder_input, attention_mask, labels, loss_mask, extra_inputs = self.prepare_llm_input(audio_batch)
            input_audio_tokens = extra_inputs[2]

            # use last position of loss mask as speech mask
            speech_mask = loss_mask[:, :, -1].reshape(loss_mask.size(0), loss_mask.size(1))
            output = self._gpt_forward(
                None,
                None,
                encoder_input,
                attention_mask,
                labels,
                checkpoint_activations_all_layers,
                speech_mask=speech_mask,
                input_audio_tokens=input_audio_tokens,
            )
            multimodal_output['audio_text'] = (output, loss_mask)

        if text_batch:
            input_ids = text_batch["text_input_ids"]
            labels = text_batch["text_labels_ids"]
            loss_mask = text_batch["text_loss_masks"]
            limit_max_seq_length = self.cfg.get("limit_max_seq_length", None)
            if limit_max_seq_length is not None and limit_max_seq_length < labels.shape[1] and self.training:
                # start = random.randint(0, labels.shape[1] - limit_max_seq_length - 1)
                labels = labels[:, :limit_max_seq_length]
                input_ids = input_ids[:, :limit_max_seq_length]
                loss_mask = loss_mask[:, :limit_max_seq_length]

            attention_mask = self._create_attention_mask(input_ids)
            output = self._gpt_forward(
                input_ids, None, None, attention_mask, labels, checkpoint_activations_all_layers
            )
            multimodal_output['text'] = (output, loss_mask)
        if not audio_batch and not text_batch:
            raise ValueError("No input data found for the model.")

        return multimodal_output

    @classmethod
    def get_codec_models_and_configs(cls, cfg):
        pretrained_codec_model = cfg.model.get("codec_model_path", None)
        pretrained_codec_model_class = cfg.model.get(
            "pretrained_codec_model_target", "nemo.collections.tts.models.audio_codec.AudioCodecModel"
        )

        model_class = hydra.utils.get_class(pretrained_codec_model_class)
        if pretrained_codec_model.endswith('.nemo'):
            logging.info(f'Loading pretrained codec model from local file: {pretrained_codec_model}')
            codec_model = model_class.restore_from(pretrained_codec_model, map_location='cpu')
        else:
            logging.info(f'Loading pretrained codec model from NGC: {pretrained_codec_model}')
            codec_model = model_class.from_pretrained(pretrained_codec_model, map_location='cpu')
        return codec_model, codec_model.cfg

    @classmethod
    def get_asr_models_and_configs(cls, cfg):

        pretrained_asr_model = cfg.model.get("asr_model_path", None)
        pretrained_asr_model_class = cfg.model.get(
            "pretrained_asr_model_target", "nemo.collections.asr.models.ASRModel"
        )

        model_class = hydra.utils.get_class(pretrained_asr_model_class)
        if pretrained_asr_model.endswith('.nemo'):
            logging.info(f'Loading pretrained codec model from local file: {pretrained_asr_model}')
            asr_model = model_class.restore_from(pretrained_asr_model, map_location='cpu')
        else:
            logging.info(f'Loading pretrained asr model from NGC: {pretrained_asr_model}')
            asr_model = model_class.from_pretrained(pretrained_asr_model, map_location='cpu')
        return asr_model, asr_model.cfg

    @classmethod
    def get_mos_models_and_configs(cls, cfg):
        return SQUIM_SUBJECTIVE.get_model()

    def setup_optimizer_param_groups(self):
        super().setup_optimizer_param_groups()
        freeze_llm = self.cfg.get('freeze_llm', True)
        if freeze_llm:
            # needs to be updated since vocab is changed
            for param in self.model.embedding.parameters():
                param.requires_grad = True
            for param in self.model.output_layers.parameters():
                param.requires_grad = True
            for param in self.model.output_layer.parameters():
                param.requires_grad = True
            for param in self.model.decoder.final_layernorm.parameters():
                param.requires_grad = True

            if self.speech_decoder_parms:
                for param in self.model.speech_decoder.parameters():
                    param.requires_grad = True

                for param in self.model.text_dim_to_speech_proj.parameters():
                    param.requires_grad = True

                for param in self.model.speech_dim_to_text_proj.parameters():
                    param.requires_grad = True
