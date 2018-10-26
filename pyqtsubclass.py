# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 18:51:26 2017

@author: Federico Barabas
"""

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter, ParameterTree


class DevTree(ParameterTree):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Parameter tree for the camera configuration
        params = [{'name': 'Load image', 'type': 'group', 'children': [
                      {'name': 'STORM', 'type': 'group', 'children': [
                          {'name': 'Pixel size', 'type': 'float',
                           'value': 13.3e-9,  'siPrefix': True, 'suffix': 'm'},
                          {'name': 'Magnification', 'type': 'float',
                           'value': 10},
                          {'name': 'Load', 'type': 'action'}]},
                      {'name': 'STED', 'type': 'group', 'children': [
                          {'name': 'Pixel size', 'type': 'float',
                           'value': 20e-9,  'siPrefix': True, 'suffix': 'm'},
                          {'name': 'Load', 'type': 'action'}]}]},
                  {'name': 'Settings', 'type': 'group', 'children': [
                      {'name': 'ROI size', 'type': 'float', 'value': 1000e-9,
                       'siPrefix': True, 'suffix': 'm'},
                      {'name': 'Gaussian filter sigma', 'type': 'float',
                       'value': 100e-9,  'siPrefix': True, 'suffix': 'm'},
                      {'name': '#sigmas threshold', 'type': 'float',
                       'value': 0.5},
                      {'name': 'Neuron content discrimination',
                       'type': 'action'},
                      {'name': 'Filter image', 'type': 'action'},
                      {'name': 'Lines minimum length', 'type': 'float',
                       'value': 300e-9,  'siPrefix': True, 'suffix': 'm'},
                      {'name': 'Get direction', 'type': 'action'},
                      {'name': 'Ring periodicity', 'type': 'float',
                       'value': 180e-9,  'siPrefix': True, 'suffix': 'm'},
                      {'name': 'Sinusoidal pattern power', 'type': 'int',
                       'value': 6},
                      {'name': 'Angular step', 'type': 'float', 'value': 3,
                       'suffix': 'º'},
                      {'name': 'Angular range', 'type': 'float', 'value': 20,
                       'suffix': 'º'},
                      {'name': 'Pearson coefficient threshold',
                       'type': 'float', 'value': 0.2},
                      {'name': 'Advanced', 'type': 'action'},
                      {'name': 'Run analysis', 'type': 'action'}
                      ]}]

        self.p = Parameter.create(name='params', type='group', children=params)
        self.setParameters(self.p, showTop=False)

        self.advanced = True
        self.toggleAdvanced()
        advParam = self.p.param('Settings').param('Advanced')
        advParam.sigActivated.connect(self.toggleAdvanced)

    def toggleAdvanced(self):

        if self.advanced:
            self.p.param('Settings').param('Lines minimum length').hide()
            self.p.param('Settings').param('Sinusoidal pattern power').hide()
            self.p.param('Settings').param('Angular step').hide()
            self.p.param('Settings').param('Angular range').hide()
            self.advanced = False
        else:
            self.p.param('Settings').param('Lines minimum length').show()
            self.p.param('Settings').param('Sinusoidal pattern power').show()
            self.p.param('Settings').param('Angular step').show()
            self.p.param('Settings').param('Angular range').show()
            self.advanced = True


class Grid:

    def __init__(self, viewbox, shape, n=[10, 10]):
        self.vb = viewbox
        self.n = n
        self.lines = []

        pen = pg.mkPen(color=(255, 255, 255), width=1, style=QtCore.Qt.DotLine,
                       antialias=True)
        self.rect = QtGui.QGraphicsRectItem(0, 0, shape[0], shape[1])
        self.rect.setPen(pen)
        self.vb.addItem(self.rect)
        self.lines.append(self.rect)

        step = np.array(shape)/self.n

        for i in np.arange(0, self.n[0] - 1):
            cx = step[0]*(i + 1)
            line = QtGui.QGraphicsLineItem(cx, 0, cx, shape[1])
            line.setPen(pen)
            self.vb.addItem(line)
            self.lines.append(line)

        for i in np.arange(0, self.n[1] - 1):
            cy = step[1]*(i + 1)
            line = QtGui.QGraphicsLineItem(0, cy, shape[0], cy)
            line.setPen(pen)
            self.vb.addItem(line)
            self.lines.append(line)


class SubImgROI(pg.ROI):

    def __init__(self, step, *args, **kwargs):
        super().__init__([0, 0], [0, 0], translateSnap=True, scaleSnap=True,
                         *args, **kwargs)
        self.step = step
        self.keyPos = (0, 0)
        self.addScaleHandle([1, 1], [0, 0], lockAspect=True)

    def moveUp(self):
        self.moveRoi(0, self.step)

    def moveDown(self):
        self.moveRoi(0, -self.step)

    def moveRight(self):
        self.moveRoi(self.step, 0)

    def moveLeft(self):
        self.moveRoi(-self.step, 0)

    def moveRoi(self, dx, dy):
        self.keyPos = (self.keyPos[0] + dx, self.keyPos[1] + dy)
        self.setPos(self.keyPos)
