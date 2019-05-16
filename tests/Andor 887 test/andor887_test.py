# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 17:04:29 2018

@author: USUARIO
"""

#import lantz.drivers.legacy.andor.ccd as ccd
import lantz.drivers.andor.ccd as ccd
import ctypes as ct
import time
import numpy as np
import matplotlib.pyplot as plt

myAndor = ccd.CCD()

#myAndor.cameraIndex = 0
#print(myAndor.camera_handle(cam))
#a = myAndor.lib.GetAvailableCameras()
#print(a)

#cam = 0
#myAndor.current_camera = myAndor.camera_handle(cam)

myAndor.lib.Initialize()

print('idn:', myAndor.idn)
print('handler:', myAndor.current_camera)
print('detector shape:', myAndor.detector_shape)
#print(type(myAndor.detector_shape))
#print(type(myAndor.detector_shape[0]))
print('status:', myAndor.status)
myAndor.acquisition_mode = 'Run till abort'
print('acquisition mode:', myAndor.acquisition_mode)

print('has mechanical shutter:', myAndor.has_mechanical_shutter)

gain = ct.c_int()
myAndor.lib.GetEMCCDGain(ct.pointer(gain))

shape = myAndor.detector_shape
myAndor.set_exposure_time(0.100)
myAndor.set_image(shape=shape)

myAndor.shutter(0, 1 , 0, 0, 0)
myAndor.start_acquisition()
time.sleep(.2)
myAndor.abort_acquisition()
myAndor.shutter(0, 2, 0, 0, 0)

image = myAndor.most_recent_image16(shape)

myAndor.finalize()

plt.imshow(image, interpolation=None)

