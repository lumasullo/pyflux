# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 17:04:29 2018

@author: USUARIO
"""

import lantz.drivers.legacy.andor.ccd as ccd
import ctypes as ct
import time
import numpy as np
import matplotlib.pyplot as plt


myAndor = ccd.CCD()
#myAndor.cameraIndex = 0
#print(myAndor.camera_handle(cam))
#a = myAndor.lib.GetAvailableCameras()
#print(a)

cam = 0

myAndor.current_camera = myAndor.camera_handle(cam)
myAndor.lib.Initialize()

print(myAndor.idn)
print(myAndor.current_camera)

#cam = 1
#
#myAndor.current_camera = myAndor.camera_handle(cam)
#myAndor.lib.Initialize()
#
#hname = (ct.c_char * myAndor.camera_handle(cam))()
#myAndor.lib.GetHeadModel(hname)
#hname = str(hname.value)[2:-1]
#sn = ct.c_uint()
#myAndor.lib.GetCameraSerialNumber(ct.pointer(sn))
#print('Andor ' + hname + ', serial number ' + str(sn.value))
#print(myAndor.current_camera)
print(myAndor.detector_shape)
print(type(myAndor.detector_shape))
print(type(myAndor.detector_shape[0]))
print(myAndor.status)
myAndor.acquisition_mode = 'Run till abort'
print(myAndor.acquisition_mode)

shape = myAndor.detector_shape
myAndor.set_exposure_time(0.100)
myAndor.set_image(shape=shape)

myAndor.start_acquisition()
time.sleep(.5)
myAndor.abort_acquisition()

image = myAndor.most_recent_image16(shape)

myAndor.finalize()

plt.imshow(image)

