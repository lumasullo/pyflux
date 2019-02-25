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
import qdarkstyle

from instrumental.drivers.cameras import uc480
import lantz.drivers.andor.ccd as ccd
import drivers.picoharp as picoharp

import drivers
import drivers.ADwin as ADwin

import focus
import scan
import tcspc
import xy_tracking

π = np.pi


class Frontend(QtGui.QMainWindow):

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

        self.psfMeasAction.triggered.connect(self.psf_measurement)

        self.minfluxMeasAction = QtGui.QAction('MINFLUX measurement', self)
        self.minfluxMeasAction.setStatusTip('Routine to perform a tcspc-MINFLUX measurement')
        fileMenu.addAction(self.minfluxMeasAction)

        # GUI layout

        grid = QtGui.QGridLayout()
        self.cwidget.setLayout(grid)

        # Dock Area

        dockArea = DockArea()
        grid.addWidget(dockArea, 0, 0)

        ## scanner

        scanDock = Dock('Confocal scan', size=(1, 1))

        DEVICENUMBER = 0x1
        adw = ADwin.ADwin(DEVICENUMBER, 1)
        scan.setupDevice(adw)

        self.scanWidget = scan.Frontend()

        scanDock.addWidget(self.scanWidget)
        dockArea.addDock(scanDock, 'left')

        ## tcspc

        tcspcDock = Dock("Time-correlated single-photon counting")

        self.tcspcWidget = tcspc.Frontend()

        tcspcDock.addWidget(self.tcspcWidget)
        dockArea.addDock(tcspcDock, 'bottom', scanDock)

        ## focus lock

        focusDock = Dock("Focus Lock")

        self.focusWidget = focus.Frontend()

        focusDock.addWidget(self.focusWidget)
        dockArea.addDock(focusDock, 'right')

        ## xy tracking

        xyDock = Dock("xy drift control")

        self.xyWidget = xy_tracking.Frontend()

        xyDock.addWidget(self.xyWidget)
        dockArea.addDock(xyDock, 'top', focusDock)

        # sizes to fit my screen properly

        self.scanWidget.setMinimumSize(1000, 550)
        self.xyWidget.setMinimumSize(800, 300)
        self.move(1, 1)
        
    def make_connection(self, backend):
        
        backend.focusWorker.make_connection(self.focusWidget)
        backend.scanWorker.make_connection(self.scanWidget)
        backend.tcspcWorker.make_connection(self.tcspcWidget)
        backend.xyWorker.make_connection(self.xyWidget)
        

    def psf_measurement(self):

        self.psfWidget = psf_meas_widget()
        self.psfWidget.show()

    def closeEvent(self, *args, **kwargs):

        # TO DO: add every module close event function

#        self.focusThread.terminate()
##        self.tcspcThread.terminate()
#        self.xyThread.terminate()
#        self.scanThread.terminate()
#        self.scanWidget.closeEvent()
        super().closeEvent(*args, **kwargs)
        
        
class Backend(QtCore.QObject):
    
    def __init__(self, adw, ph, ccd, scmos, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.scanWorker = scan.Backend(adw)
        self.focusWorker = focus.Backend(scmos, adw)
        self.tcspcWorker = tcspc.Backend(ph)
        self.xyWorker = xy_tracking.Backend(ccd)

    def make_connection(self, frontend):
        
        frontend.focusWidget.make_connection(self.focusWorker)
        frontend.scanWidget.make_connection(self.scanWorker)
        frontend.tcspcWidget.make_connection(self.tcspcWorker)
        frontend.xyWidget.make_connection(self.xyWorker)


class psf_meas_widget(QtGui.QWidget):
     
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        self.setWindowTitle('PSF measurement')

        grid = QtGui.QGridLayout()

        self.setLayout(grid)
        self.paramWidget = QtGui.QFrame()
        self.paramWidget.setFrameStyle(QtGui.QFrame.Panel |
                                       QtGui.QFrame.Raised)

        grid.addWidget(self.paramWidget, 0, 0)
        
        subgrid = QtGui.QGridLayout()
        self.paramWidget.setLayout(subgrid)
        
        self.NframesLabel = QtGui.QLabel('Number of frames')
        self.NframesEdit = QtGui.QLineEdit('10')
        self.doughnutLabel = QtGui.QLabel('Doughnut label')
        self.doughnutEdit = QtGui.QLineEdit('')
        self.startButton = QtGui.QPushButton('Start PSF measurement')
        self.progress = QtGui.QProgressBar(self)
        
        subgrid.addWidget(self.NframesLabel, 0, 0)
        subgrid.addWidget(self.NframesEdit, 1, 0)
        subgrid.addWidget(self.doughnutLabel, 2, 0)
        subgrid.addWidget(self.doughnutEdit, 3, 0)
        subgrid.addWidget(self.startButton, 4, 0)
        subgrid.addWidget(self.progress, 5, 0)
        


if __name__ == '__main__':

    app = QtGui.QApplication([])
    app.setStyle(QtGui.QStyleFactory.create('fusion'))
#    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    
    gui = Frontend()
    
    cam = uc480.UC480_Camera()
    andor = ccd.CCD()
    ph = picoharp.PicoHarp300()
    
    DEVICENUMBER = 0x1
    adw = ADwin.ADwin(DEVICENUMBER, 1)
    scan.setupDevice(adw)
    
    worker = Backend(adw, ph, andor, cam)
    
    gui.make_connection(worker)
    worker.make_connection(gui)
    
    gui.scanWidget.emit_param()
    worker.scanWorker.emit_param()

    focusThread = QtCore.QThread()
    worker.focusWorker.moveToThread(focusThread)
    worker.focusWorker.focusTimer.moveToThread(focusThread)
    worker.focusWorker.focusTimer.timeout.connect(worker.focusWorker.update)

    focusThread.start()
    
    tcspcWorkerThread = QtCore.QThread()
    worker.tcspcWorker.moveToThread(tcspcWorkerThread)
    
    tcspcWorkerThread.start()
    
    scanThread = QtCore.QThread()
    worker.scanWorker.moveToThread(scanThread)
    worker.scanWorker.viewtimer.moveToThread(scanThread)
    worker.scanWorker.viewtimer.timeout.connect(worker.scanWorker.update_view)

    scanThread.start()
    
    gui.show()
    app.exec_()
