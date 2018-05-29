#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common definitions for Defect Detection 

TH: Thinning
SM: Scratch Mark
SD: Slot Defect
WR: Wrinkling
OKF: OK Front
OKB: Ok Back

"""

from util import one_hot

n_classes = 6
LBLS = [
    "TH",
    "SM",
    "SD",
    "WR",
    "OKF",
    "OKB"
    ]
NONE = "O"
LMAP = {i: one_hot(n_classes,i) for i, k in enumerate(LBLS)}
LID = {k: i for i, k in enumerate(LBLS)}
