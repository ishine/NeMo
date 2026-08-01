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

echo "=== Acquiring MSMARCO dataset ==="
echo "---"

mkdir -p msmarco_dataset
cd msmarco_dataset

echo "- Downloading passages"
wget --quiet --continue https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz
tar -xzvf collection.tar.gz
rm collection.tar.gz

echo "- Downloading queries"
wget --quiet --continue https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz
tar -xzvf queries.tar.gz
rm queries.tar.gz
rm queries.eval.tsv

echo "- Downloading relevance labels"
wget --quiet --continue https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv
wget --quiet --continue https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv

echo "---"