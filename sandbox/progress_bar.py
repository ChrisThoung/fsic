# -*- coding: utf-8 -*-
"""
progress_bar
============
Experiments extending [`tqdm`](https://github.com/tqdm/tqdm)-based progress
bars, to eventually augment `ProgressBarMixin`.
"""

import fsic


SCRIPT = """
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
"""
SYMBOLS = fsic.parse_model(SCRIPT)
