# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

mkdir tests/collections/llm/squad_download

coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/collections/llm/megatron_t5_finetuning.py \
    --devices=2 \
    --max-steps=250 \
    --data-dir=tests/collections/llm/squad_download \
    --experiment-dir=tests/collections/llm/t5_finetune_squad_redownload_results/$RUN_ID \
    --checkpoint-path=/home/TestData/nlp/megatron_t5/220m/nemo2.0_t5_220m_padding_attnmasktype_150steps
