# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 14:29:44 2018

@author: USUARIO
"""

import time
import ctypes
from ctypes import byref, POINTER
import sys
import struct
import matplotlib.pyplot as plt

fileName = 'tttrmode.out'

with open(fileName, mode='rb') as file: # b is important -> binary
    fileContent = file.read()
    
data = struct.unpack("i" * ((len(fileContent) - 24) // 4), fileContent[20:-4])

plt.hist(data, bins=100)