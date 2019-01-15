# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 16:47:26 2018

@author: Luciano A. Masullo
"""

import ctypes
from ctypes import byref, POINTER
import time
import numpy as np

# From phdefin.h
LIB_VERSION = "3.0"
MAXDEVNUM = 8
MODE_T2 = 2
MODE_T3 = 3
TTREADMAX = 131072
FLAG_OVERFLOW = 0x0040
FLAG_FIFOFULL = 0x0003

# Measurement parameters, these are hardcoded since this is just a demo
mode = MODE_T3 # set T2 or T3 here, observe suitable Syncdivider and Range!
binning = 0 # you can change this, meaningful only in T3 mode
offset = 0 # you can change this, meaningful only in T3 mode
tacq = 1000 # Measurement time in millisec, you can change this
syncDivider = 1 # you can change this, observe mode! READ MANUAL!
CFDZeroCross0 = 10 # you can change this (in mV)
CFDLevel0 = 50 # you can change this (in mV)
CFDZeroCross1 = 10 # you can change this (in mV)
CFDLevel1 = 150 # you can change this (in mV)
dev = int(0)

# Variables to store information read from DLLs
buffer = (ctypes.c_uint * TTREADMAX)()
libVersion = ctypes.create_string_buffer(b"", 8)
hwSerial = ctypes.create_string_buffer(b"", 8)
hwPartno = ctypes.create_string_buffer(b"", 8)
hwVersion = ctypes.create_string_buffer(b"", 8)
hwModel = ctypes.create_string_buffer(b"", 16)
errorString = ctypes.create_string_buffer(b"", 40)
resolution = ctypes.c_double()
countRate0 = ctypes.c_int()
countRate1 = ctypes.c_int()
flags = ctypes.c_int()
nactual = ctypes.c_int()
ctcDone = ctypes.c_int()
warnings = ctypes.c_int()
warningstext = ctypes.create_string_buffer(b"", 16384)

phlib = ctypes.CDLL("phlib64.dll")

phlib.PH_OpenDevice(ctypes.c_int(dev), hwSerial)

serialNumber = hwSerial.value.decode("utf-8")

phlib.PH_Initialize(ctypes.c_int(dev), ctypes.c_int(mode))
phlib.PH_GetHardwareInfo(dev, hwModel, hwPartno, hwVersion)

device = hwModel.value.decode("utf-8")
version = hwVersion.value.decode("utf-8")

phlib.PH_StartMeas(ctypes.c_int(dev), ctypes.c_int(tacq))
time.sleep(2)
phlib.PH_StopMeas(ctypes.c_int(dev))

phlib.PH_CloseDevice(ctypes.c_int(dev))


phlib.PH_ReadFiFo(ctypes.c_int(dev), byref(buffer), TTREADMAX, byref(nactual))



print('{} {}, serial number {}'.format(device, version, serialNumber))