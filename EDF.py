#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author：Elan time:2020/1/16

import numpy as np


def GlobalEDF(instance, no_processor):
    deadline = []
    action = np.zeros(len(instance))
    for i in instance:
        deadline.append(i.deadline)
    executable = np.argsort(deadline)
    for i in range(no_processor):
        if i < len(executable):
            action[executable[i]] = 1
    return action

