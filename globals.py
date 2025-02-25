#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# globals.py
from collections import defaultdict
import time

daily_losses = defaultdict(float)
historical_losses = defaultdict(list)
open_orders = defaultdict(list)
last_day = time.strftime("%Y-%m-%d")
running = True
MAX_OPEN_ORDERS = 2  # Начальное значение, будет обновляться в limits.py
