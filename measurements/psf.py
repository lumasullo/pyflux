# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:38:16 2019

@author: Luciano A. Masullo
"""

import numpy as np
import time
import os
from datetime import date, datetime

from threading import Thread

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import Dock, DockArea
import qdarkstyle

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from tkinter import Tk, filedialog

import tools.tools as tools

π = np.pi

DEBUG = True

class Frontend(QtGui.QFrame):
    
    paramSignal = pyqtSignal(dict)
     
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        self.setup_gui()
        
    def emit_param(self):
        
        params = dict()
        params['label'] = self.doughnutLabel.text()
        params['nframes'] = int(self.NframesEdit.text())
        
        self.paramSignal.emit(params)
    
    @pyqtSlot(float)
    def get_progress_signal(self, completed):
        
        self.progressBar.setValue(completed)
        
    def setup_gui(self):
        
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
        self.doughnutEdit = QtGui.QLineEdit('Black, Blue, Yellow, Orange')
        self.startButton = QtGui.QPushButton('Start')
        self.stopButton = QtGui.QPushButton('Stop')
        self.progressBar = QtGui.QProgressBar(self)
        
        subgrid.addWidget(self.NframesLabel, 0, 0)
        subgrid.addWidget(self.NframesEdit, 1, 0)
        subgrid.addWidget(self.doughnutLabel, 2, 0)
        subgrid.addWidget(self.doughnutEdit, 3, 0)
        subgrid.addWidget(self.startButton, 4, 0)
        subgrid.addWidget(self.progressBar, 5, 0)
        subgrid.addWidget(self.stopButton, 6, 0)
        
    def make_connection(self, backend):
    
        backend.progressSignal.connect(self.get_progress_signal)
        
            
class Backend(QtCore.QObject):
    
    xySignal = pyqtSignal(bool, bool) # bool 1: whether you feedback ON or OFF, bool 2: initial position
    xyStopSignal = pyqtSignal()
    
    zSignal = pyqtSignal(bool, bool)
    zStopSignal = pyqtSignal()
    
    confocalSignal = pyqtSignal(bool, str)

    progressSignal = pyqtSignal(float)

    def __init__(self, *args, **kwargs):
    
        super().__init__(*args, **kwargs)
        
        self.i = 0
        self.nframes = 30
        
        self.xyIsDone = False
        self.zIsDone = False
        self.confocalIsDone = False
        
        self.measTimer = QtCore.QTimer()
        self.measTimer.timeout.connect(self.measurement_loop)

    def start_measurement(self):
        
        self.i = 0
        
        print(datetime.now(), '[psf] measurement started')
    
        self.xyStopSignal.emit()
        self.zStopSignal.emit()
        
        self.data = np.zeros((1, 1))  # data array (size, size, nframes)
        self.xyz_flag = True
        self.scan_flag = True
    
        self.measTimer.start(0)
        
    def stop_measurement(self):
        
        self.measTimer.stop()
        self.progressSignal.emit(0)
        
        self.xyStopSignal.emit()
        self.zStopSignal.emit()
        
        print(datetime.now(), '[psf] measurement ended')
        
        self.export_data()
        
    def measurement_loop(self):
                
        if self.xyz_flag:
            
            if self.i is 0:
                initial = True
            else:
                initial = False
                
            self.xySignal.emit(True, initial)
            self.zSignal.emit(True, initial)
            
            if DEBUG:
                print(datetime.now(), '[psf] xy and z signals emitted ({})'.format(self.i))
            
            self.xyz_flag = False
            
        if self.xyIsDone and self.zIsDone:
                
            if self.scan_flag:
                    
                self.confocalSignal.emit(True, 'frame')
                
                if DEBUG:
                    print(datetime.now(), '[psf] scan signal emitted ({})'.format(self.i))
                    
                self.scan_flag = False
                                        
            if self.confocalIsDone:
                
                completed = ((self.i+1)/self.nframes) * 100
                self.progressSignal.emit(completed)
                                
                self.xyz_flag = True
                self.scan_flag = True
                self.xyIsDone = False
                self.zIsDone = False
                self.confocalIsDone = False
                
                print(datetime.now(), '[psf] PSF {} of {}'.format(self.i+1, 
                                                                  self.nframes))
                
                if self.i < self.nframes-1:
                
                    self.i += 1
                
                else:
                    
                    self.stop_measurement()
                    
    def export_data(self):
        
        pass
    
    @pyqtSlot(dict)
    def get_frontend_param(self, params):
        
        self.label = params['label']
        self.nframes = params['nframes']
                
    @pyqtSlot(bool) 
    def get_xy_is_done(self, val):
        
        self.xyIsDone = True
        
    @pyqtSlot(bool) 
    def get_z_is_done(self, val):
        
        self.zIsDone = True       
        
    @pyqtSlot(bool) 
    def get_confocal_is_done(self, val):
        
        self.confocalIsDone = True
        
    def ask_scan_parameters(self):
        
        pass
        
    @pyqtSlot(dict) 
    def get_scan_parameters(self, params):
        
        self.Npixels = int(params['NofPixels'])
        
        # TO DO: build config file
        
    def make_connection(self, frontend):
        
        frontend.startButton.clicked.connect(self.start_measurement)
        frontend.stopButton.clicked.connect(self.stop_measurement)
            
            
            
            
            
        