# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 18:47:04 2018

@author: Cibion
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats



x, x_measured = np.array(np.load(os.path.join(os.getcwd(), 'data_x_selfc.npy')), 
                         dtype=np.float16)

y, y_measured = np.load(os.path.join(os.getcwd(), 'data_y_selfc.npy'))

z, z_measured = np.load(os.path.join(os.getcwd(), 'data_z_selfc.npy'))


#plt.figure('Piezo calibration, x axis (x vs x_measured)')
#plt.plot(x, x, '-')
#plt.plot(x, x_measured, 'ro')
#plt.xlabel('x setpoint (µm)')
#plt.ylabel('x measured (µm)')
#
#
#plt.figure('Piezo calibration,  y axis (y vs y_measured)')
#plt.plot(y, y, '-')
#plt.plot(y, y_measured,'ro')
#plt.xlabel('y setpoint (µm)')
#plt.ylabel('y measured (µm)')

#plt.figure('Piezo calibration, x axis (input V vs x_measured)')
#plt.plot(x/2, x, 'b-')
#plt.plot(x/2, x_measured, 'ro')
#plt.ylabel('x position (µm)')
#plt.xlabel('Input signal (V)')
#
#plt.ylim(-0.5, 20.5)
#
#plt.figure('Piezo calibration, y axis (input V vs y_measured)')
#plt.plot(y/2, y, 'b-')
#plt.plot(y/2, y_measured, 'ro')
#plt.ylabel('y position (µm)')
#plt.xlabel('Input signal (V)')
#
#plt.ylim(0, 20.5)
#
#x = np.append(np.array([0]), x)
#y = np.append(np.array([0]), y)
#
#
#plt.figure('x axis, signal vs amplified signal')
#plt.plot(x/2, x/2 * 7.5, 'b-')
#plt.plot(x/2, xV, 'ro')
#plt.ylabel('input signal (V)')
#plt.xlabel('amplified signal (V)')
#
#plt.figure('y axis, input signal vs amplified signal')
#plt.plot(y/2, y/2 * 7.5, 'b-')
#plt.plot(y/2, yV, 'ro')
#plt.ylabel('input signal (V)')
#plt.xlabel('amplified signal (V)')


plt.figure('built-in calibration')
data0, = plt.plot(x, x_measured, 'ro')
data1, = plt.plot(y, y_measured, 'go')
data2, = plt.plot(z, z_measured, 'bo')

plt.plot(np.linspace(0, 20, 1000), np.linspace(0, 75, 1000), 'k-')

plt.ylabel('volts signal (V)')
plt.xlabel('set position (µm)')






