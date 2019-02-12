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

class mainWindow(QtGui.QMainWindow):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.setWindowTitle('PyFLUX')
        
        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)

        # Actions in menubar
        
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('Measurement')
        
        self.psfMeasAction = QtGui.QAction('PSF measurement', self)
        self.psfMeasAction.setStatusTip('Routine to measure one MINFLUX PSF')
        fileMenu.addAction(self.psfMeasAction)
        
        self.psfMeasAction.triggered.connect(self.psfMeasurement)
                
        self.minfluxMeasAction = QtGui.QAction('MINFLUX measurement', self)
        self.minfluxMeasAction.setStatusTip('Routine to perform a tcspc-MINFLUX measurement')
        fileMenu.addAction(self.minfluxMeasAction)

        # GUI layout

        grid = QtGui.QGridLayout()
        self.cwidget.setLayout(grid)

        # Dock Area
        
        dockArea = DockArea() 
        grid.addWidget(dockArea, 0, 0)
        
        # Scan
        
        scanDock = Dock('Confocal scan', size=(1,1))
        
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
        self.scanThread.start()
        self.scanWidget.scworker.moveToThread(self.scanThread)
        
        self.focusThread = QtCore.QThread(self)
        self.focusThread.start()
        self.focusWidget.fworker.moveToThread(self.focusThread)

        self.xyThread = QtCore.QThread(self)
        self.xyThread.start()
        self.xyWidget.xyworker.moveToThread(self.xyThread)

#        self.tcspcThread = QtCore.QThread(self)
#        self.tcspcThread.start() 
#        self.tcspcWidget.tcspcworker.moveToThread(self.tcspcThread)

        # sizes to fit my screen properly
        
        self.scanWidget.setMinimumSize(1000, 550)
        self.xyWidget.setMinimumSize(800, 300)
        self.move(1, 1)

    def psfMeasurement(self):
        
        self.psfWidget = psfMeasWidget()
        self.psfWidget.show()

    def closeEvent(self, *args, **kwargs):

        # TO DO: add every module close event function
        
        self.focusThread.terminate()
#        self.tcspcThread.terminate()
        self.xyThread.terminate()
        self.scanThread.terminate()
        self.scanWidget.closeEvent()
        super().closeEvent(*args, **kwargs)
        
class psfMeasWidget(QtGui.QWidget):
        
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        grid = QtGui.QGridLayout()
        
        self.setLayout(grid)
        self.paramWidget = QtGui.QFrame()
        self.paramWidget.setFrameStyle(QtGui.QFrame.Panel |
                                       QtGui.QFrame.Raised)
        
        grid.addWidget(self.paramWidget, 0, 0)
    

if __name__ == '__main__':

    app = QtGui.QApplication([])
    win = mainWindow()
    win.show()

    app.exec_()
