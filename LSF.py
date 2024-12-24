#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author：Elan time:2020/2/25

import numpy as np


def GlobalLSF(instance, no_processor):
    laxity = []
    action = np.zeros(len(instance))
    for i in instance:
        laxity.append(i.laxity_time)
    executable = np.argsort(laxity)
    for i in range(no_processor):
        if i < len(executable):
            action[executable[i]] = 1
    return action
