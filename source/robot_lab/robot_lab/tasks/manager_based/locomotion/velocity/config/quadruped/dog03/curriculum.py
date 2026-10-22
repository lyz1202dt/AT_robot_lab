# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Resolve the 03dog stair stage independently of CSDog1's training stage."""

import json
import os
from pathlib import Path


def stairs_config():
    config = json.loads(Path(__file__).with_name("stairs_config.json").read_text())
    selected = os.environ.get("DOG03_STAIR_STAGE", str(config["stage"]))
    if selected not in config["stages"]:
        raise ValueError(f"Invalid DOG03_STAIR_STAGE {selected!r}; expected one of {', '.join(config['stages'])}")
    return {"stage": int(selected), **config["stages"][selected]}
