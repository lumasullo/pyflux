# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 19:00:35 2018

@author: USUARIO
"""

import numpy as np
import time
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import Dock, DockArea
import pyqtgraph.ptime as ptime

from instrumental.drivers.cameras import uc480

import focus
import sys
sys.path.append('C:\Program Files\Thorlabs\Scientific Imaging\ThorCam')

from lantz import Q_

uc480camera = uc480.UC480_Camera()

print('Model {}'.format(uc480camera.model))
print('Serial number {}'.format(uc480camera.serial))

app = QtGui.QApplication([])

win = focus.FocusWidget(uc480camera)
win.show()
app.exec_()
