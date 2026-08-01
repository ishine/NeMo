# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""A script to check that copyright headers exists"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

EXCLUSIONS = [
    "scripts/get_commonvoice_data.py",
    "nemo/utils/env_var_parsing.py",  # third-party MIT
]


def get_top_comments(_data):
    # Get all lines where comments should exist
    lines_to_extract = []
    for i, line in enumerate(_data):
        # If empty line, skip
        if line in ["", "\n", "", "\r", "\r\n"]:
            continue
        # If it is a comment line, we should get it
        if line.startswith("#"):
            lines_to_extract.append(i)
        # Assume all copyright headers occur before any import statements not enclosed in a comment block
        elif "import" in line:
            break

    comments = []
    for line in lines_to_extract:
        comments.append(_data[line])

    return comments


def main():
    parser = argparse.ArgumentParser(description="Usage for copyright header insertion script")
    parser.add_argument(
        '--dir',
        help='Path to source files to add copyright header to. Will recurse through subdirectories',
        required=True,
        type=str,
    )
    args = parser.parse_args()

    current_year = int(datetime.today().year)
    starting_year = 2020
    python_header_path = "tests/py_cprheader.txt"
    pyheader = Path(python_header_path).read_text(encoding='utf-8').splitlines()
    pyheader_lines = len(pyheader)

    problematic_files = []
    spdx_copyright_re = re.compile(
        r"^# SPDX-FileCopyrightText: Copyright \(c\) (?P<year>\d{4}) "
        r"NVIDIA CORPORATION & AFFILIATES\. All rights reserved\.\s*$"
    )
    spdx_license_re = re.compile(r"^# SPDX-License-Identifier: Apache-2\.0\s*$")

    for filename in Path(args.dir).rglob('*.py'):
        rel = str(filename)
        if rel in EXCLUSIONS or any(rel.endswith(e) for e in EXCLUSIONS):
            continue
        with open(str(filename), 'r', encoding='utf-8') as original:
            data = original.readlines()

        comments = get_top_comments(data)
        if len(comments) < pyheader_lines:
            print(f"{filename} has less header lines than the copyright template")
            problematic_files.append(filename)
            continue

        # Find NVIDIA SPDX copyright line (may follow third-party copyright lines)
        found = False
        for i, line in enumerate(comments):
            m = spdx_copyright_re.match(line.rstrip("\n"))
            if not m:
                continue
            found = True
            year = int(m.group("year"))
            if year < starting_year or year > current_year:
                problematic_files.append(filename)
                print(f"{filename} had an error with the year: {year}")
                break
            # Next non-third-party line should be SPDX-License-Identifier
            if i + 1 >= len(comments) or not spdx_license_re.match(comments[i + 1].rstrip("\n")):
                problematic_files.append(filename)
                print(f"{filename} missing SPDX-License-Identifier: Apache-2.0 after copyright")
                break
            # Remaining template lines (from index 2) must appear after SPDX license line
            base = i + 1
            ok = True
            for j in range(2, pyheader_lines):
                if base + (j - 1) >= len(comments) or pyheader[j] not in comments[base + (j - 1)]:
                    problematic_files.append(filename)
                    print(f"{filename} missed the line: {pyheader[j]}")
                    ok = False
                    break
            if not ok:
                break
            break
        if not found:
            print(f"{filename} did not match SPDX-FileCopyrightText NVIDIA header")
            problematic_files.append(filename)

    if len(problematic_files) > 0:
        print("check_copyright_headers.py found the following files that might not have a copyright header:")
        for _file in problematic_files:
            print(_file)
        sys.exit(1)


if __name__ == '__main__':
    main()
