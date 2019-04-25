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

from instrumental.drivers.cameras import uc480
import lantz.drivers.andor.ccd as ccd
import drivers.picoharp as picoharp

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from tkinter import Tk, filedialog

import drivers
import drivers.ADwin as ADwin

import tools.tools as tools

π = np.pi

class Frontend(QtGui.QFrame):
     
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
        
class Backend(QtCore.QObject):
    
        def __init__(self, *args, **kwargs):
        
            super().__init__(*args, **kwargs)
            
            self.i = 0
            self.n = 0 # TO DO: get from GUI
         
        @pyqtSlot() 
        def get_scan_parameters(self):
            
            self.Npixels = 50
            
            # TO DO: build config file
            
        def start_psf_measurement(self):
            
            self.data = np.zeros()
            
            self.xySignal.emit()
            self.zSignal.emit()
            
            time.sleep(1)
            
            self.confocalSignal.emit()
            
        @pyqtSlot()  
        def psf_measurement(self):
            
            if self.i < self.n:
            
                self.xySignal.emit()
                self.zSignal.emit()
                
                time.sleep(1)
                
                self.confocalSignal.emit()
            
            
        