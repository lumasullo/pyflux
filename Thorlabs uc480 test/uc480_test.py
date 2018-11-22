# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:14:13 2018

@author: USUARIO
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from instrumental.drivers.cameras import uc480

sys.path.append('C:\Program Files\Thorlabs\Scientific Imaging\ThorCam')

myCamera = uc480.UC480_Camera()

print('Model {}'.format(myCamera.model))
print('Serial number {}'.format(myCamera.serial))

raw_image = myCamera.grab_image(exposure_time='50 ms')

r = raw_image[:, :, 0]
g = raw_image[:, :, 1]
b = raw_image[:, :, 2]

image = np.sum(raw_image, axis=2)

plt.imshow(image, interpolation='None', cmap='viridis')

#myCamera.close()
