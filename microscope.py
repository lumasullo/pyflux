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

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot

import drivers
import drivers.ADwin as ADwin

import focus
import scan
import tcspc
import xy_tracking

import tools.tools as tools

π = np.pi


class Frontend(QtGui.QMainWindow):
    
    closeSignal = pyqtSignal()

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.setWindowTitle('PyFLUX')

        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)

        # Actions in menubar

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('Measurement')
        
        self.psfWidget = psf_meas_widget()

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
#        dockArea.addDock(xyDock, 'top')

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

        self.psfWidget.show()

    def closeEvent(self, *args, **kwargs):
        
        self.closeSignal.emit()

        super().closeEvent(*args, **kwargs)
        
        
class Backend(QtCore.QObject):
    
    measurePSFSignal = pyqtSignal()
    
    def __init__(self, adw, ph, ccd, scmos, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.scanWorker = scan.Backend(adw)
        self.focusWorker = focus.Backend(scmos, adw)
        self.tcspcWorker = tcspc.Backend(ph)
        self.xyWorker = xy_tracking.Backend(ccd, adw)

    def make_connection(self, frontend):
        
        frontend.focusWidget.make_connection(self.focusWorker)
        frontend.scanWidget.make_connection(self.scanWorker)
        frontend.tcspcWidget.make_connection(self.tcspcWorker)
        frontend.xyWidget.make_connection(self.xyWorker)
        frontend.closeSignal.connect(self.stop)
            
    def stop(self):
        
        self.scanWorker.stop()
        self.focusWorker.stop()
        self.tcspcWorker.stop()
        self.xyWorker.stop()
        
        
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
#    app.setStyle(QtGui.QStyleFactory.create('fusion'))
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    
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
    
    # drift correction connections
    
    worker.scanWorker.xyDriftSignal.connect(worker.xyWorker.discrete_drift_correction)
    worker.scanWorker.paramSignal.connect(worker.xyWorker.get_scan_parameters)
    gui.scanWidget.feedbackLoopBox.stateChanged.connect(gui.xyWidget.emit_roi_info)
    worker.xyWorker.paramSignal.connect(worker.scanWorker.get_drift_corrected_param)

    # initial parameters
    
    gui.scanWidget.emit_param()
    worker.scanWorker.emit_param()
      
    # focus thread

    focusThread = QtCore.QThread()
    worker.focusWorker.moveToThread(focusThread)
    worker.focusWorker.focusTimer.moveToThread(focusThread)
    worker.focusWorker.focusTimer.timeout.connect(worker.focusWorker.update) # TO DO: this will probably call the update twice, fix!!
#
    focusThread.start()
    
    # focus GUI thread
    
    focusGUIThread = QtCore.QThread()
    gui.focusWidget.moveToThread(focusGUIThread)
    
    focusGUIThread.start()
#    
#    # xy worker thread
#    
    xyThread = QtCore.QThread()
    worker.xyWorker.moveToThread(xyThread)
    worker.xyWorker.viewtimer.moveToThread(xyThread)
#    worker.xyWorker.viewtimer.timeout.connect(worker.xyWorker.update_view)
#    
    xyThread.start()
    
    # xy GUI thread
    
    xyGUIThread = QtCore.QThread()
    gui.xyWidget.moveToThread(xyGUIThread)
    
    xyGUIThread.start()

    # tcspc thread
    
    tcspcWorkerThread = QtCore.QThread()
    worker.tcspcWorker.moveToThread(tcspcWorkerThread)
    
    tcspcWorkerThread.start()
    
    # scan thread
    
    scanThread = QtCore.QThread()
    
    worker.scanWorker.moveToThread(scanThread)
    worker.scanWorker.viewtimer.moveToThread(scanThread)
    worker.scanWorker.viewtimer.timeout.connect(worker.scanWorker.update_view)

    scanThread.start()
    
    gui.show()
    app.exec_()
