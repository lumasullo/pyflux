# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 14:18:19 2018

@author: Luciano A. Masullo
"""

import numpy as np
import time
import os

from threading import Thread

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import Dock, DockArea

from instrumental.drivers.cameras import uc480
import lantz.drivers.legacy.andor.ccd as ccd

import drivers
import focus
import scan
import tcspc
import xy



π = np.pi

class mainWindow(QtGui.QFrame):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.setWindowTitle('PyFLUX')
#        self.resize(1200, 2780)

        # GUI layout

        grid = QtGui.QGridLayout()
        self.setLayout(grid)

        # Dock Area
        
        dockArea = DockArea() 
        grid.addWidget(dockArea, 0, 0)
        
        # Scan
        
        scanDock = Dock('Confocal scan')
        
        DEVICENUMBER = 0x1
        self.adw = drivers.ADwin.ADwin(DEVICENUMBER, 1)
        scan.setupDevice(self.adw)
        
        self.scanWidget = scan.scanWidget(self.adw)
        scanDock.addWidget(self.scanWidget)
        dockArea.addDock(scanDock, 'left')
        
        # tcspc measurement
        
        tcspcDock = Dock("Time-correlated single-photon counting")
        
        ph = drivers.picoharp.PicoHarp300()
        
        self.tcspcWidget = tcspc.tcspcWidget(ph)
        tcspcDock.addWidget(self.tcspcWidget)
        dockArea.addDock(tcspcDock, 'bottom', scanDock)
        
        # focus lock
        
        focusDock = Dock("Focus Lock")
        
        uc480Camera = uc480.UC480_Camera()
        
        self.focusWidget = focus.focusWidget(uc480Camera, self.adw)
        focusDock.addWidget(self.focusWidget)
        dockArea.addDock(focusDock, 'right')
        
        # xy drift
        
        xyDock = Dock("xy drift control")
        
        andorCamera = ccd.CCD()
        
        self.xyWidget = xy.xyWidget(andorCamera)
        xyDock.addWidget(self.xyWidget)
        xyDock.addWidget(self.xyWidget)
        dockArea.addDock(xyDock, 'top', focusDock)
        
        # threads
        
        self.scanThread = QtCore.QThread(self)
        self.scanWidget.scworker.moveToThread(self.scanThread)
        self.scanThread.start()
        
        self.focusThread = QtCore.QThread(self)
        self.focusWidget.fworker.moveToThread(self.focusThread)
        self.focusThread.start()
        
        self.xyThread = QtCore.QThread(self)
        self.xyWidget.xyworker.moveToThread(self.xyThread)
        self.xyThread.start()
        
        self.tcspcThread = QtCore.QThread(self)
        self.tcspcWidget.moveToThread(self.tcspcThread)
        self.tcspcThread.start() 
        
         
    def closeEvent(self, *args, **kwargs):
        
#        self.xydriftWidget.andor.shutter(0, 2, 0, 0, 0)
#        self.xydriftWidget.andor.abort_acquisition()
#        self.xydriftWidget.andor.finalize()

        # TO DO: add every module close event function

        self.scanWidget.closeEvent()
        super().closeEvent(*args, **kwargs)
    

if __name__ == '__main__':

    app = QtGui.QApplication([])
    win = mainWindow()
    win.show()

    app.exec_()
