#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#SBATCH -p batch -A coreai_dlalgo_llm -t 4:00:00 --nodes=16 --exclusive --mem=0 --overcommit --gpus-per-node 8 --ntasks-per-node=8 --dependency=singleton

export WANDB_RESUME=allow
export WANDB_NAME=train_vae

DIR=`pwd`

srun --signal=TERM@300 -l --container-image ${IMAGE} --container-mounts "/lustre:/lustre/,/home:/home" --no-container-mount-home --mpi=pmix bash -c "cd ${DIR} ; python -u nemo/collections/diffusion/vae/train_vae.py --yes $*"
