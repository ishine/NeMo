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
import os
import torch
import torch.distributed as dist
import torchaudio
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf, open_dict
from peft import PeftModel
from torch import Tensor
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    loss_parallel,
    parallelize_module,
)
import tempfile
from transformers import DynamicCache

from nemo.collections.asr.models import ASRModel
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.collections.speechlm2.models.duplex_s2s_model import replace_control_speech_codes, tokens_to_str
from nemo.collections.speechlm2.modules import AudioPerceptionModule, TransformerARSpeechDecoder
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.metrics.asr_bleu import ASRBLEU
from nemo.collections.speechlm2.parts.metrics.bleu import BLEU
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.speechlm2.parts.pretrained import load_pretrained_hf, load_pretrained_nemo, set_model_dict_for_partial_init
from nemo.collections.tts.models import AudioCodecModel
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.utils import logging


class DuplexS2SSpeechDecoderModel(LightningModule, HFHubMixin):
    def __init__(self, cfg: dict) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to DuplexS2SModel as a Python dict to support hyperparameter serialization "
            f"in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()
        # convert dict to config
        cfg = DictConfig(cfg)
        self.cfg = cfg.model
        self.target_sample_rate = cfg.data.target_sample_rate
        self.source_sample_rate = cfg.data.source_sample_rate
        # compute source fps
        self.source_fps = self.source_sample_rate / (self.source_sample_rate * cfg.data.frame_length) # conver frame rate in fps

        self.setup_audio_codec()
        self._codebook_size = self.audio_codec.vector_quantizer.codebook_size_per_group
        self._num_codebooks = self.audio_codec.vector_quantizer.num_groups

        # compute target fps
        self.target_fps = self.target_sample_rate / self.audio_codec.samples_per_frame
        # compute interpolation factor to interpolate 
        self.interpolation_factor = self.target_fps / self.source_fps
        # x = torch.nn.functional.interpolate(x.unsqueeze(1), size=None, scale_factor=[1, self.interpolation_factor], mode='nearest-exact', align_corners=None, recompute_scale_factor=None, antialias=False)

        # We load the pretrained HF LLM using "ForCausalLM" variant so that we can obtain the
        # pretrained LM head weights.
        # However, for S2S we need to access the activations before LM head directly
        # to feed them to the audio codec head.
        self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)
        llm = load_pretrained_hf(self.cfg.pretrained_llm, pretrained_weights=self.cfg.pretrained_weights).train()
        self.llm = llm.model  # fetch PretrainedBaseModel from model "ForCausalLM"
        self.lm_head = llm.lm_head
        # Note: we have to "move out" the token embedding outside of LLM to avoid
        #       messing up FSDP/TP hooks.
        self.embed_tokens = self.llm.embed_tokens
        del self.llm.embed_tokens
        maybe_install_lora(self)

        # Load the pretrained streaming ASR model and copy its parameters into the audio perception module.
        asr = load_pretrained_nemo(ASRModel, self.cfg.pretrained_asr).eval()
        with open_dict(self.cfg):
            self.cfg.perception.preprocessor = asr.cfg.preprocessor
            self.cfg.perception.encoder = asr.cfg.encoder
            self.cfg.perception.output_dim = self.llm.config.hidden_size
        self.perception = AudioPerceptionModule(self.cfg.perception).train()
        self.perception.load_state_dict(asr.state_dict(), strict=False)


        llm_tokenizer_vocab_items = self.tokenizer.vocab
        # if vocab is a dict it already has the subword and token id, if not, get it from the tokenizer
        if isinstance(llm_tokenizer_vocab_items, dict):
            llm_tokenizer_vocab_items = llm_tokenizer_vocab_items.items()
        else:
            llm_tokenizer_vocab_items = [(subword, self.tokenizer.tokenizer._tokenizer.token_to_id(subword)) for subword in llm_tokenizer_vocab_items]

        self.speech_generation = TransformerARSpeechDecoder(
            speech_decoder_parms=OmegaConf.to_container(self.cfg.speech_decoder),
            lantent_dim=self.llm.config.hidden_size,
            num_audio_codebooks=self._num_codebooks,
            num_audio_tokens_per_codebook=self.speech_vocab_size,
            llm_tokenizer_vocab_items=llm_tokenizer_vocab_items,
        )

        # load pretrained TTS model
        if self.cfg.get("pretrained_tts", None):
            self.init_speech_generation_from_tts_checkpoint(self.cfg.pretrained_tts)

        if self.cfg.get("pretrained_tts_from_s2s", None):
            self.init_speech_generation_from_another_s2s_checkpoint(self.cfg.pretrained_tts_from_s2s)

        self.embed_audio_tokens = torch.nn.ModuleList(
            [
                torch.nn.Embedding(self.speech_vocab_size, self.embed_tokens.embedding_dim)
                for _ in range(self._num_codebooks)
            ]
        )
        self.audio_head = torch.nn.Linear(self.llm.config.hidden_size, self.speech_vocab_size * self._num_codebooks)

        # cached for quicker audio decoding
        self.register_buffer(
            "_control_codes",
            torch.tensor([self.speech_bos_id, self.speech_eos_id, self.speech_delay_id], device=self.device),
        )

        self._use_fsdp = False
        self._use_tp = False

    def init_speech_generation_from_tts_checkpoint(self, checkpoint_path):
        if checkpoint_path is not None:
            if '.nemo' in checkpoint_path:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        NLPSaveRestoreConnector._unpack_nemo_file(checkpoint_path, tmpdir)
                        checkpoint_path = f"{tmpdir}/model_weights.ckpt"
                        checkpoint_state = torch.load(checkpoint_path)
            else:
                checkpoint_state = torch.load(checkpoint_path, weights_only=False)['state_dict']

            checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, self.speech_generation.state_dict())
            self.speech_generation.load_state_dict(checkpoint_state, strict=True)

    def init_speech_generation_from_another_s2s_checkpoint(self, checkpoint_path):
        if checkpoint_path is not None:
            if '.nemo' in checkpoint_path:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        NLPSaveRestoreConnector._unpack_nemo_file(checkpoint_path, tmpdir)
                        checkpoint_path = f"{tmpdir}/model_weights.ckpt"
                        checkpoint_state = torch.load(checkpoint_path)
            else:
                checkpoint_state = torch.load(checkpoint_path, weights_only=False)['state_dict']

            # filter keys to keep only speech generation keys and also
            checkpoint_state = {k.replace("speech_decoder.", "").replace("speech_generation.", ""): v for k, v in checkpoint_state.items() if "speech_decoder." in k or "speech_generation." in k}
            checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, self.speech_generation.state_dict())
            self.speech_generation.load_state_dict(checkpoint_state, strict=True)

    def setup_audio_codec(self):
        """Workaround for PTL auto-downcasting the codec model to bf16 with bf16-true precision."""
        if hasattr(self, "audio_codec") and next(self.audio_codec.parameters()).dtype == torch.float:
            return  # skip if already set up and has the right dtype
        with fp32_precision():
            self.audio_codec = load_pretrained_nemo(AudioCodecModel, self.cfg.pretrained_audio_codec).eval()
        for p in self.audio_codec.parameters():
            p.requires_grad = False
        del self.audio_codec.discriminator  # free up some memory

    @property
    def speech_vocab_size(self):
        """Return the size of the audio codec codebook including extra speech BOS and EOS tokens."""
        return self._codebook_size + 3

    @property
    def speech_bos_id(self) -> int:
        """Indicates start of utterance generation (not start of inference!)."""
        return self._codebook_size

    @property
    def speech_eos_id(self) -> int:
        """Indicates end of utterance generation."""
        return self._codebook_size + 1

    @property
    def speech_delay_id(self) -> int:
        """Indicates start of inference (the very first frame)."""
        return self._codebook_size + 2

    @property
    def text_vocab_size(self):
        """Return the size of the text tokenizer."""
        return self.tokenizer.vocab_size

    @property
    def text_bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def text_eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def text_pad_id(self) -> int:
        """
        Text pad ID is used as a 'blank' for frames when the model is not speaking
        and for frames where the model is speaking but has already predicted the
        entire text channel's content.

        Example:

            flow:         |---user---||-------assistant--------||-user-|
            text channel:  0000000000  1xxxxxxx0000000000000002  000000

        Where 0 indicates PAD ID, 1 indicates BOS ID, 2 indacates EOS ID,
        and x indicates tokens corresponding to actual text

        """
        return get_pad_id(self.tokenizer)

    def forward(self, input_embeds: Tensor, cache=None, input_audio_tokens=None, loss_mask=None, target_text_tokens=None, modality_adapter_emb=None, speaker_encoder_emb=None) -> dict[str, Tensor]:
        """
        Separated text and speech prediction:
            - Speech prediction is achieved by a independent AR decoder based on last_hidden_state + audio tokens
            - For KV-cache:
                (1) llm cache depends on input cache is None or Not
                (2) speech_generation cache relys on reset_input_and_kv_cache function.
        """

        out = self.llm(
            inputs_embeds=input_embeds, past_key_values=cache, use_cache=cache is not None, return_dict=True
        )
        B, T = input_embeds.shape[:2]
        text_logits = self.lm_head(out['last_hidden_state'])  # (B, T, text_vocab_size)

        if loss_mask is not None:
            # This is training Mode
            loss_mask = loss_mask[:, :, -1].reshape(loss_mask.size(0), loss_mask.size(1))
            # disable cache in training mode
            if self.speech_generation.use_input_cache:
                self.speech_generation.reset_input_and_kv_cache(use_cache=False)

        # if inference time, uses the target text tokens sampled from the llm backbone
        if self.speech_generation.use_input_cache and not self.training:
            target_text_tokens = torch.argmax(text_logits, dim=-1).view(B, T).contiguous()
            # print(self.speech_generation.use_input_cache, target_text_tokens.shape if target_text_tokens is not None else target_text_tokens)
            # print(tokens_to_str(target_text_tokens[-1:], torch.tensor(target_text_tokens[-1:].shape[1]).long().unsqueeze(0), tokenizer=self.tokenizer, pad_id=self.text_pad_id))
            # print(self.speech_generation.cache["target_text_tokens"].shape if self.speech_generation.cache["target_text_tokens"] is not None else None)

        audio_logits, _  = self.speech_generation(
            out['last_hidden_state'].transpose(0, 1), loss_mask, input_audio_tokens=input_audio_tokens, target_text_tokens=target_text_tokens, modality_adapter_emb=modality_adapter_emb, speaker_encoder_emb=speaker_encoder_emb
        )

        audio_logits = audio_logits.view(B, T, self._num_codebooks, self.speech_vocab_size)

        ans = {
            "text_logits": text_logits,
            "audio_logits": audio_logits,
        }
        if cache is not None:
            ans["cache"] = out["past_key_values"]
        return ans

    def prepare_inputs(self, batch: dict):
        """
        Similar to DuplexS2SModel.prepare_inputs, with following changes:
            (1) Add 'input_audio_tokens' and 'loss_mask' in return value for TransformerARSpeechDecoder
            (2) Remove audio codec embedding from 'input_embeds'
        """
        # check if audios has the same batch size
        assert batch["source_audio"].size(0) == batch["target_audio"].size(0)
        assert batch["target_first_turn_audio"].size(0) == batch["target_audio"].size(0)

        source_encoded, source_encoded_lens = self.perception(
            input_signal=batch["source_audio"], input_signal_length=batch["source_audio_lens"]
        )

        # if inference return speaker embedding None and it will uses the cached speaker embedding
        if not self.training:
            speaker_encoder_emb = None
        else: # if training or eval extract embedding from first agent turn returned by the dataloader 
            if self.speech_generation.use_speaker_encoder:
                target_first_turn_audio = batch["target_first_turn_audio"]
                target_first_turn_audio_lens = batch["target_first_turn_audio_lens"]
                speaker_encoder_emb = self.speech_generation.get_speaker_embedding(target_first_turn_audio, target_first_turn_audio_lens, self.target_sample_rate)
            else:
                speaker_encoder_emb = None

        target_tokens = batch["target_tokens"]
        if (diff := target_tokens.shape[1] - source_encoded.shape[1]) < 0:
            target_tokens = torch.cat(
                [
                    target_tokens,
                    (
                        torch.ones(source_encoded.shape[0], abs(diff), device=source_encoded.device) * self.text_pad_id
                    ).to(torch.long),
                ],
                dim=-1,
            )
        elif diff > 0:
            target_tokens = target_tokens[:, : source_encoded.shape[1]]

        with fp32_precision(), torch.no_grad():
            target_codes, target_codes_lens = self.audio_codec.encode(
                audio=batch["target_audio"], audio_len=batch["target_audio_lens"]
            )
        target_codes = target_codes.transpose(1, 2)  # (B, K, T) -> (B, T, K)

        if (tl := target_codes.shape[1]) != (sl := source_encoded.shape[1]):
            if tl < sl:
                diff = sl - tl
                source_encoded = source_encoded[:, :tl]
                target_tokens = target_tokens[:, :tl]
                torch.clamp_(source_encoded_lens, max=tl)
            else:
                diff = tl - sl
                target_codes = target_codes[:, :sl]
                torch.clamp_(target_codes_lens, max=sl)
            if diff > 2:
                logging.warning(
                    f"A mismatch between source ({sl}) and target ({tl}) sequence length greater than 2 detected. "
                    f"This may indicate significant desynchronization in longer sessions."
                )

        btt = target_tokens[..., None]
        target_codes = torch.where(btt == self.text_bos_id, self.speech_bos_id, target_codes)
        target_codes = torch.where(btt == self.text_eos_id, self.speech_eos_id, target_codes)

        target_codes = torch.cat(
            [
                torch.full(
                    [target_codes.shape[0], 1, target_codes.shape[-1]],
                    fill_value=self.speech_delay_id,
                    device=self.device,
                    dtype=torch.long,
                ),
                target_codes[:, :-1],
            ],
            dim=1,
        )

        input_ids = torch.cat([target_codes, target_tokens[..., None]], dim=-1)
        if self._use_tp:
            tp_world_size = self.device_mesh["tensor_parallel"].size()
            if (remainder := (input_ids.shape[1] - 1) % tp_world_size) != 0:
                input_ids = input_ids[:, :-remainder]
                source_encoded = source_encoded[:, :-remainder]

        text_inputs = input_ids[:, :-1, -1]  # (B, T-1)
        text_labels = input_ids[:, 1:, -1]  # (B, T-1)
        audio_inputs = input_ids[:, :-1, :-1]  # (B, T-1, K)
        audio_labels = input_ids[:, 1:, :-1]  # (B, T-1, K)

        input_embeds = self.embed_tokens(text_inputs)

        input_embeds.add_(source_encoded[:, :-1] * self.cfg.get("duplex_user_channel_weight", 1.0))

        loss_mask = torch.ones_like(
            torch.cat([text_labels.unsqueeze(-1), audio_labels], dim=-1),
            device=self.device,
            dtype=torch.bool,
        )

        if self.cfg.get("mask_sequence_loss", True):
            # set the mask based on the target_token_lens to disconsider sequence padding in loss
            for i in range(batch["target_token_lens"].size(0)):
                speech_end_idx = batch["target_token_lens"][i]
                loss_mask[i, speech_end_idx:, :] = 0

            # check new mask consistency
            mask_lengths = loss_mask[:, :, 0].sum(-1)
            assert torch.allclose(batch["target_token_lens"].float(), mask_lengths.float(), atol=2.0)

        """
        # debug samples:
        def write_wave(one_audio_signal, file_name, sr=None):
            import numpy as np
            import soundfile as sf
            one_audio_signal = one_audio_signal.cpu().numpy()
            one_audio_signal = one_audio_signal.astype(np.float32)
            if sr is None:
                sr = self.target_sample_rate
            # one_audio_signal = np.clip(one_audio_signal, -1.0, 1.0)
            sf.write(file_name, one_audio_signal, sr)    

        write_wave(
            batch["target_audio"][-1],
            "/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/debug-samples/new_code_base_target_audio_5.wav",
            sr=22050
        )
        write_wave(
            batch["target_first_turn_audio"][-1],
            "/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/debug-samples/new_code_base_speaker_ref_5.wav",
            sr=22050
        )
        write_wave(
            batch["source_audio"][-1],
            "/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/debug-samples/new_code_base_input_5.wav",
            sr=16000
        )
        # reconstruct wav
        audio_labels = replace_control_speech_codes(audio_labels, self._control_codes)
        with fp32_precision(), torch.no_grad():
            lengths = torch.tensor([audio_labels.shape[1]]*audio_labels.shape[0]).to(self.audio_codec.device)
            predicted_audio, predicted_audio_lens = self.audio_codec.decode(
                tokens=audio_labels.transpose(1, 2), tokens_len=lengths
            )
        write_wave(
            predicted_audio[-1],
            "/lustre/fsw/portfolios/convai/users/ecasanova/S2S-Duplex-new-codebase/debug-samples/reconstructed_codec_audio_5.wav",
            sr=22050
        )

        # check text
        print("text_labels", text_labels)
        print("target labels from dataloader", batch["target_tokens"])
        print("text_labels", tokens_to_str(text_labels[-1:], target_codes_lens-1, tokenizer=self.tokenizer, pad_id=self.text_pad_id))
        print("target labels from dataloader",  tokens_to_str(batch["target_tokens"][-1:], target_codes_lens-1, tokenizer=self.tokenizer, pad_id=self.text_pad_id))

        zeros_begening = 0
        for t in text_labels[-1:].squeeze():
            if t == 0:
                zeros_begening += 1
            else:
                break

        print("Total aduio seconds padded input:", (zeros_begening*self.audio_codec.samples_per_frame)/ self.target_sample_rate)

        exit()
        """

        return {
            "input_embeds": input_embeds,
            "input_lens": source_encoded_lens - 1,
            "output_lens": target_codes_lens - 1,
            "text_labels": text_labels,
            "input_audio_tokens": audio_inputs,
            "audio_labels": audio_labels,
            "loss_mask": loss_mask,
            "perception_emb": source_encoded[:, :-1],
            "speaker_encoder_emb": speaker_encoder_emb,
        }

    def training_step(self, batch: dict, batch_idx: int):
        for m in (self.perception.preprocessor, self.perception.encoder, self.llm, self.speech_generation):
            if is_frozen(m):
                m.eval()
        inputs = self.prepare_inputs(batch)
        forward_outputs = self(
            inputs["input_embeds"],
            input_audio_tokens=inputs["input_audio_tokens"],
            loss_mask=inputs["loss_mask"],
            target_text_tokens=inputs["text_labels"],
            modality_adapter_emb=inputs["perception_emb"],
            speaker_encoder_emb=inputs["speaker_encoder_emb"],
        )
        num_frames = inputs["input_lens"].sum()
        with loss_parallel():
            # mask audio logits to ignore sequence padding
            text_logits = forward_outputs["text_logits"]
            if self.cfg.get("mask_sequence_loss", True):
                text_logits = text_logits * inputs["loss_mask"][:, :, 0].unsqueeze(-1)
            text_loss = (
                torch.nn.functional.cross_entropy(
                    text_logits.flatten(0, 1),  # (B, T, Vt) -> (*, Vt)
                    inputs["text_labels"].flatten(0, 1),
                    reduction="sum",
                )
                / num_frames
            )
            # mask audio logits to ignore sequence padding
            audio_logits = forward_outputs["audio_logits"]
            if self.cfg.get("mask_sequence_loss", True):
                print(inputs["loss_mask"][:, :, -1].unsqueeze(-1).shape, audio_logits.shape)
                exit()
                audio_logits = audio_logits * inputs["loss_mask"][:, :, -1].unsqueeze(-1).unsqueeze(-1)

            audio_loss = torch.nn.functional.cross_entropy(
                audio_logits.flatten(0, 2),  # (B, T, K, Vs) -> (*, Vs)
                inputs["audio_labels"].flatten(0, 2),
                reduction="sum",
            ) / (num_frames * self._num_codebooks)
        loss = self.cfg.text_loss_weight * text_loss + self.cfg.audio_loss_weight * audio_loss

        B, T = inputs["input_embeds"].shape[:2]
        ans = {
            "loss": loss,
            "learning_rate": (
                torch.as_tensor(self.trainer.optimizers[0].param_groups[0]['lr'] if self._trainer is not None else 0)
            ),
            "text_loss": text_loss,
            "audio_loss": audio_loss,
            "batch_size": B,
            "sequence_length": T,
            "num_frames": num_frames.to(torch.float32),  # avoid warning
            "padding_ratio": num_frames / (B * T),
        }
        self.log_dict(ans, on_step=True)
        return ans

    def on_train_epoch_start(self) -> None:
        self.setup_audio_codec()  # potentially reloads the audio codec to make sure it's in fp32
        if hasattr(self.speech_generation, "use_speaker_encoder") and self.speech_generation.use_speaker_encoder:
            self.speech_generation.setup_speaker_encoder() # potentially reloads the speaker encoder to make sure it's in fp32

    def on_validation_epoch_start(self) -> None:
        self.on_train_epoch_start()
        self.asr_bleu = ASRBLEU(self.cfg.scoring_asr).reset()
        self.bleu = BLEU().reset()

    def on_validation_epoch_end(self, prefix="val") -> None:
        asr_bleu = self.asr_bleu.compute()
        for k, m in asr_bleu.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        bleu = self.bleu.compute()
        for k, m in bleu.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)

    def validation_step(self, batch: dict, batch_idx: int):

        # Update speaker embedding to reflect the one in the prompt during inference
        if self.speech_generation.inference_speaker_reference:
            self.speech_generation.update_inference_speaker_embedding(self.speech_generation.inference_speaker_reference)

        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted

            results = self.offline_inference(
                dataset_batch["source_audio"],
                dataset_batch["source_audio_lens"],
            )

            if self.cfg.get('audio_save_path', None) is not None and dist.get_rank() == 0:
                os.makedirs(self.cfg.audio_save_path, exist_ok=True)
                predicted_audios = results["audio"]
                for i in range(len(predicted_audios)):
                    pred_audio = predicted_audios[i].float()
                    user_audio = torchaudio.functional.resample(dataset_batch["source_audio"][i].float(), self.source_sample_rate, self.target_sample_rate)

                    T1, T2 = pred_audio.shape[0], user_audio.shape[0]
                    max_len = max(T1, T2)
                    pred_audio_padded = torch.nn.functional.pad(pred_audio, (0, max_len - T1), mode='constant', value=0)
                    user_audio_padded = torch.nn.functional.pad(user_audio, (0, max_len - T2), mode='constant', value=0)

                    # combine audio in a multichannel audio
                    combined_wav = torch.cat([user_audio_padded.squeeze().unsqueeze(0).detach().cpu(), pred_audio_padded.squeeze().unsqueeze(0).detach().cpu()], dim=0)

                    # save audio
                    out_audio_path = f"{self.cfg.audio_save_path}/{name}_{dataset_batch['sample_id'][i]}.wav"
                    torchaudio.save(out_audio_path, combined_wav.squeeze(), self.target_sample_rate)
                    print("Audio saved at:", out_audio_path)

            with fp32_precision():  # torchaudio resample is fragile to bfloat16 default dtype as well
                self.asr_bleu.update(
                    name=name,
                    refs=dataset_batch["target_texts"],
                    pred_audio=torchaudio.functional.resample(results["audio"], 22050, 16000),
                    pred_audio_lens=(results["audio_len"] / 22050 * 16000).to(torch.long),
                )

            self.bleu.update(name=name, refs=dataset_batch["target_texts"], hyps=results["text"])

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end(prefix="test")

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def _get_bos_embedding(self) -> torch.Tensor:
        """
        Remove the audio codec embedding for the beginning of AR decoding.
        """
        text_bos = torch.full((1,), fill_value=self.text_pad_id, device=self.device)
        input_embeds = self.embed_tokens(text_bos)
        return input_embeds

    @torch.no_grad()
    def offline_inference(
        self,
        input_signal: torch.Tensor,
        input_signal_lens: torch.Tensor,
        decode_audio: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Autoregressive prediction.

        Args:
            input_signal: a batch of waveforms with shape (B, T) with source sampling rate.
            input_signal_lens: example lengths as number of samples of shape (B,).
            decode_audio: bool, whether to decode audio codes to waveform.

        Returns:
            A dict with keys:
                * "text": generated text, de-tokenized to strings, properly skipping text_pad_id; list of length B.
                * "tokens_text": generated text tokens of shape (B, T2).
                * "tokens_audio": generated audio codes of shape (B, T2, K) where `K=num_codebooks`.
                * "tokens_len" output lengths as number of tokens of shape (B,).
                * "audio": generated waveform of shape (B, T3) (`decode_audio=True`).
                * "audio_len" output lengths as number of waveform samples of shape (B,) (when `decode_audio=True`).
        """
        source_encoded, lengths = self.perception(
            input_signal=input_signal,
            input_signal_length=input_signal_lens,
        )
        B, T_local, H = source_encoded.shape

        # Determine decoding length and pad if FSDP
        if self._use_fsdp:
            T_tensor = torch.tensor([T_local], device=source_encoded.device)
            dist.all_reduce(T_tensor, op=dist.ReduceOp.MAX)
            T = int(T_tensor.item())
            if T > T_local:
                last_frame = source_encoded[:, T_local - 1 : T_local, :]  # (B,1,H)
                pad = last_frame.repeat(1, T - T_local, 1)  # (B, T-T_local, H)
                source_encoded = torch.cat([source_encoded, pad], dim=1)
        else:
            T = T_local

        # Apply channel weight
        input_embeds = source_encoded.clone()
        input_embeds *= self.cfg.get("duplex_user_channel_weight", 1.0)

        # This cache is for self.llm
        cache = DynamicCache()
        # Call reset_input_and_kv_cache to enable cache for TransformerARSpeechDecoder
        self.speech_generation.reset_input_and_kv_cache(use_cache=True)
        gen_text = torch.empty(B, T, device=self.device, dtype=torch.long)
        gen_audio = torch.empty(B, T, self._num_codebooks, device=self.device, dtype=torch.long)

        # First step, use speech_delay token
        input_embeds[:, 0] += self._get_bos_embedding()
        first_audio = torch.full(
            [B, 1, self._num_codebooks],
            fill_value=self.speech_delay_id,
            device=self.device,
            dtype=torch.long,
        )
        ans = self(
            input_embeds[:, :1], 
            cache=cache,
            input_audio_tokens=first_audio,
            loss_mask=None,
            target_text_tokens=None, # text input will be sampled from llm backbone
            modality_adapter_emb=source_encoded[:, :1],
            speaker_encoder_emb=None, # for inference uses the cached inference_speaker_embedding
        )
        gen_text[:, 0] = ans["text_logits"][:, -1].argmax(dim=-1)
        gen_audio[:, 0] = ans["audio_logits"][:, -1].argmax(dim=-1)

        # Autoregressive loop
        for t in range(1, T):
            last_emb = self.embed_tokens(gen_text[:, t - 1])
            input_embeds[:, t] += last_emb
            current_audio = gen_audio[:, t - 1 : t, :]
            ans = self(
                input_embeds[:, t : t + 1],
                cache=ans["cache"],
                input_audio_tokens=current_audio,
                loss_mask=None,
                target_text_tokens=None, # text input will be sampled from llm backbone
                modality_adapter_emb=source_encoded[:, t : t + 1],
                speaker_encoder_emb=None, # for inference uses the cached inference_speaker_embedding
            )
            gen_text[:, t] = ans["text_logits"][:, -1].argmax(dim=-1)
            gen_audio[:, t] = ans["audio_logits"][:, -1].argmax(dim=-1)

        # Trim back to local length if padded
        if self._use_fsdp and T > T_local:
            gen_text = gen_text[:, :T_local]
            gen_audio = gen_audio[:, :T_local]

        ans = {
            "text": tokens_to_str(gen_text, lengths, tokenizer=self.tokenizer, pad_id=self.text_pad_id),
            "tokens_text": gen_text,
            "tokens_audio": gen_audio,
            "tokens_len": lengths,
        }

        if decode_audio:
            gen_audio_codes = replace_control_speech_codes(gen_audio, self._control_codes)
            with fp32_precision(), torch.no_grad():
                predicted_audio, predicted_audio_lens = self.audio_codec.decode(
                    tokens=gen_audio_codes.transpose(1, 2), tokens_len=lengths
                )
            ans["audio"] = predicted_audio
            ans["audio_len"] = predicted_audio_lens

        # Call reset_input_and_kv_cache to reset cache for TransformerARSpeechDecoder
        self.speech_generation.reset_input_and_kv_cache(use_cache=False)
        return ans

    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    def configure_optimizers(self):
        return configure_optimizers(self)

    def configure_model(self) -> None:
        # TODO(pzelasko): refactor into separate module re-usable across models
        device_mesh = self.device_mesh
        if device_mesh is None:
            return

        llm = self.llm
        if isinstance(llm, PeftModel):
            llm = llm.base_model.model

        if (tp_mesh := device_mesh["tensor_parallel"]).size() > 1:
            self._use_tp = True

            plan = {
                "layers.0": PrepareModuleInput(
                    input_layouts=(Replicate(),),  # , None)
                    desired_input_layouts=(Shard(1),),  # , None)
                    use_local_output=True,
                ),
                "norm": SequenceParallel(),
            }
            parallelize_module(llm, tp_mesh, plan)

            for transformer_block in llm.layers:
                plan = {
                    "input_layernorm": SequenceParallel(),
                    "self_attn.q_proj": ColwiseParallel(),
                    "self_attn.k_proj": ColwiseParallel(),
                    "self_attn.v_proj": ColwiseParallel(),
                    "self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
                    "post_attention_layernorm": SequenceParallel(),
                    "mlp": PrepareModuleInput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "mlp.gate_proj": ColwiseParallel(),
                    "mlp.up_proj": ColwiseParallel(),
                    "mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
                    # "pre_feedforward_layernorm": SequenceParallel(),
                    # "post_feedforward_layernorm": SequenceParallel(),
                }

                # Adjust attention module to use the local number of heads
                attn_layer = transformer_block.self_attn
                for attr in ("num_heads", "num_key_value_heads", "hidden_size"):
                    val = getattr(attn_layer, attr)
                    if val % tp_mesh.size() != 0:
                        logging.warning(
                            f"attn_layer.{attr}={val} is not divisible by {tp_mesh.size()=}: "
                            f"set a different tensor parallelism size to avoid errors."
                        )
                    setattr(attn_layer, attr, val // tp_mesh.size())

                parallelize_module(transformer_block, tp_mesh, plan)

            for m in (self.lm_head, self.audio_head):
                parallelize_module(
                    m,
                    tp_mesh,
                    ColwiseParallel(
                        input_layouts=Shard(1),
                        output_layouts=Shard(-1),
                        use_local_output=False,
                    ),
                )

        if (dp_mesh := device_mesh["data_parallel"]).size() > 1:
            assert dp_mesh.ndim == 1
            self._use_fsdp = True

            fsdp_config = {"mesh": dp_mesh}

            for idx, layer in enumerate(llm.layers):
                llm.layers[idx] = fully_shard(layer, **fsdp_config)
            self.embed_tokens = fully_shard(self.embed_tokens, **fsdp_config)
            self.llm = fully_shard(self.llm, **fsdp_config)
            self.lm_head = fully_shard(self.lm_head, **fsdp_config)
            self.perception = fully_shard(self.perception, **fsdp_config)
            self.speech_generation = fully_shard(self.speech_generation, **fsdp_config)

    def load_state_dict(self, state_dict, strict: bool = True):
        try:
            super().load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            logging.info(f"Error loading model state_dict !! Retrying with partial initialization!")
            model_dict = set_model_dict_for_partial_init(state_dict, self.state_dict())
            super().load_state_dict(model_dict, strict=False)
