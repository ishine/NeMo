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
from collections import defaultdict

import sacrebleu
import torch
from whisper_normalizer.english import EnglishTextNormalizer

from nemo.utils import logging


class BLEU:
    """
    Computes BLEU scores on text predictions.
    By default, uses Whisper's EnglishTextNormalizer on hypotheses and references.
    """

    def __init__(self, normalize: bool = True, normalizer=None, verbose: bool = False):
        self.verbose = verbose
        if normalize:
            if normalizer is None:
                self.normalizer = EnglishTextNormalizer()
            else:
                self.normalizer = normalizer
        else:
            self.normalizer = _identity

        self._refs = defaultdict(list)
        self._hyps = defaultdict(list)

    def reset(self):
        return self

    def update(self, name: str, refs: list[str], hyps: list[str]) -> None:
        for ref, hyp in zip(refs, hyps):
            self._refs[name].append(self.normalizer(ref))
            self._hyps[name].append(self.normalizer(hyp))
            if self.verbose:
                asrb = sacrebleu.sentence_bleu(hyp, [ref]).score
                logging.info(f"[REF]\t{ref}\n[HYP]\t{hyp} [{asrb:.2f}]")

    def compute(self) -> dict[str, torch.Tensor]:
        # Gather refs/hyps from all ranks so corpus BLEU is computed over the full
        # global dataset rather than per-rank shards.
        # All ranks participate in all_gather_object and then each rank computes BLEU
        # on the same merged data, producing an identical scalar everywhere.
        # sync_dist=True in self.log() then averages identical values → correct,
        # and no rank ever diverges from the collective call sequence → no NCCL timeout.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            local_data = {name: (list(self._refs[name]), list(self._hyps[name])) for name in self._refs}
            gathered = [None] * world_size
            torch.distributed.all_gather_object(gathered, local_data)
            merged_refs = defaultdict(list)
            merged_hyps = defaultdict(list)
            for rank_data in gathered:
                for name, (refs, hyps) in rank_data.items():
                    merged_refs[name].extend(refs)
                    merged_hyps[name].extend(hyps)
        else:
            merged_refs = self._refs
            merged_hyps = self._hyps

        corpus_metric = {}
        for name in merged_refs.keys():
            metric = torch.tensor(sacrebleu.corpus_bleu(merged_hyps[name], [merged_refs[name]]).score)
            corpus_metric[f"txt_bleu_{name}"] = metric
        if corpus_metric:
            corpus_metric["txt_bleu"] = torch.stack(list(corpus_metric.values())).mean()
        else:
            # No updates (e.g. no tool-response refs in any batch): return 0 so logging/sync_dist is valid
            corpus_metric["txt_bleu"] = torch.tensor(0.0, dtype=torch.float32)
        self._refs.clear()
        self._hyps.clear()
        return corpus_metric


def _identity(x):
    return x
