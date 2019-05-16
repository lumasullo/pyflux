# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:35:15 2019

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


π = np.pi

class Frontend(QtGui.QFrame):
    
    filenameSignal = pyqtSignal(str)
    paramSignal = pyqtSignal(np.ndarray, np.ndarray, int)
    
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        self.setup_gui()
        

    def emit_filename(self):  
        
        filename = os.path.join(self.folderEdit.text(),
                                self.filenameEdit.text())
        
        self.filenameSignal.emit(filename)
        
        print('[minflux] filename', filename)
    
    @pyqtSlot(np.ndarray, np.ndarray, int)     
    def get_backend_param(self, r0, _r, acqt):
        
        positions = str(_r)
        acqtime = str(acqt)
        r0 = str(r0)
        
#        print(datetime.now(), '[minflux] positions', positions)
#        print(datetime.now(), '[minflux] acqtime', acqtime)
        
        self.absolutePosEdit.setText(r0)
        self.positionsEdit.setText(positions)
        self.acqtimeEdit.setText(acqtime)
        
    def emit_param_to_backend(self):
        
        try:
            r0 = np.array(self.absolutePosEdit.text(), dtype=np.float)
            _r = np.array(self.positionsEdit.text(), dtype=np.float)
            acqtime = int(self.acqtimeEdit.text())
            
            self.paramSignal.emit(r0, _r, acqtime)
            
        except(ValueError):
            
            pass
        
    def load_folder(self):

        try:
            root = Tk()
            root.withdraw()
            folder = filedialog.askdirectory(parent=root,
                                             initialdir=self.initialDir)
            root.destroy()
            if folder != '':
                self.folderEdit.setText(folder)
        except OSError:
            pass
        
    def setup_gui(self):
        
        self.setWindowTitle('MINFLUX measurement')

        grid = QtGui.QGridLayout()

        self.setLayout(grid)
        self.paramWidget = QtGui.QFrame()
        self.paramWidget.setFrameStyle(QtGui.QFrame.Panel |
                                       QtGui.QFrame.Raised)

        grid.addWidget(self.paramWidget, 0, 0)
        
        subgrid = QtGui.QGridLayout()
        self.paramWidget.setLayout(subgrid)
        
        self.absolutePosLabel = QtGui.QLabel('Starting position (µm)')
        self.absolutePosEdit = QtGui.QLineEdit('')
    
        self.positionsLabel = QtGui.QLabel('Realtive positions (µm)')
        self.positionsEdit = QtGui.QLineEdit('')
        
        self.acqtimeLabel = QtGui.QLabel('Acq time per position (s)')
        self.acqtimeEdit = QtGui.QLineEdit('')
        
        self.startButton = QtGui.QPushButton('Start')
        self.progress = QtGui.QProgressBar(self)
        
        subgrid.addWidget(self.absolutePosLabel, 0, 0)
        subgrid.addWidget(self.absolutePosEdit, 1, 0)
        
        subgrid.addWidget(self.positionsLabel, 2, 0)
        subgrid.addWidget(self.positionsEdit, 3, 0)
        subgrid.addWidget(self.acqtimeLabel, 4, 0)
        subgrid.addWidget(self.acqtimeEdit, 5, 0)
        subgrid.addWidget(self.startButton, 6, 0)
        subgrid.addWidget(self.progress, 7, 0)
        
        # file/folder widget
        
        self.fileWidget = QtGui.QFrame()
        self.fileWidget.setFrameStyle(QtGui.QFrame.Panel |
                                      QtGui.QFrame.Raised)
        
        self.fileWidget.setFixedHeight(120)
        self.fileWidget.setFixedWidth(150)
        
        # folder
        
        # TO DO: move this to backend
        
        today = str(date.today()).replace('-', '')
        root = r'C:\\Data\\'
        folder = root + today
        
        try:  
            os.mkdir(folder)
        except OSError:  
            print(datetime.now(), '[tcspc] Directory {} already exists'.format(folder))
        else:  
            print(datetime.now(), '[tcspc] Successfully created the directory {}'.format(folder))

        self.folderLabel = QtGui.QLabel('Folder')
        self.folderEdit = QtGui.QLineEdit(folder)
        self.browseFolderButton = QtGui.QPushButton('Browse')
        self.browseFolderButton.setCheckable(True)
        self.filenameLabel = QtGui.QLabel('File name')
        self.filenameEdit = QtGui.QLineEdit('filename')
        
        grid.addWidget(self.fileWidget, 0, 1)
        
        file_subgrid = QtGui.QGridLayout()
        self.fileWidget.setLayout(file_subgrid)
        
        file_subgrid.addWidget(self.filenameLabel, 0, 0, 1, 2)
        file_subgrid.addWidget(self.filenameEdit, 1, 0, 1, 2)
        file_subgrid.addWidget(self.folderLabel, 2, 0, 1, 2)
        file_subgrid.addWidget(self.folderEdit, 3, 0, 1, 2)
        file_subgrid.addWidget(self.browseFolderButton, 4, 0)
        
        self.folderEdit.textChanged.connect(self.emit_filename)
        self.positionsEdit.textChanged.connect(self.emit_param_to_backend)
        self.acqtimeEdit.textChanged.connect(self.emit_param_to_backend)
        
    def make_connection(self, backend):
        
        backend.paramSignal.connect(self.get_backend_param)

class Backend(QtCore.QObject):
    
    askROIcenterSignal = pyqtSignal()
#    moveToSignal = pyqtSignal(np.ndarray)
    tcspcPrepareSignal = pyqtSignal(str, int, int)
    tcspcStartSignal = pyqtSignal()
    xyzStartSignal = pyqtSignal()
    xyzEndSignal = pyqtSignal(str)
    xyMoveAndLockSignal = pyqtSignal(np.ndarray, np.ndarray)
    
    paramSignal = pyqtSignal(np.ndarray, np.ndarray, int)
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        self.i = 1 # counter
        self.n = 6 # number of movements after initial movement
        self.r0 = np.array([3.0, 3.0]) # TO DO: get this from scan or ADwin
        self._r = np.array([[0.035, 0.035], [0.00, 0.00], [0.07, 0.00], [0.07, 0.07], [0.00, 0.07], [0.035, 0.035]])
        self.acqtime = 5 # in s
        
        self.update_param()
        self.emit_param_to_frontend()
        # TO DO: get parameters from GUI
        
#    def ask_ROI_center(self):
#        
#        self.askROIcenterSignal.emit()
    
    @pyqtSlot(np.ndarray)    
    def get_ROI_center(self, center):
        
        self.r0 = center
        time.sleep(0.4)
        self.xyzStartSignal.emit()
        
        
    @pyqtSlot(np.ndarray, np.ndarray, int)
    def get_frontend_param(self, r0, _r, acqt):
        
        self._r = np.array(_r)
        self.r0 = np.array(r0)
        self.acqtime = acqt
        
        self.update_param()
        
    def emit_param_to_frontend(self):
        
        self.paramSignal.emit(self.r0, self._r, self.acqtime)
     
    @pyqtSlot(str) 
    def get_minflux_filename(self, val):
        
        self.currentfname = val
        
    def update_param(self):
        
        self.r = self.r0 + self._r
        
    def start_minflux_meas(self):
        
        """ Starts minflux measurement
        
        n positions 
        with acqtime in each position
        
        """
        
#        self.ask_ROI_center()
        self.update_param()
#        self.moveToSignal.emit(self.r[0]) # signal emitted to xy for smooth, long movement
        self.tcspcPrepareSignal.emit(self.currentfname, self.acqtime, self.n) # signal emitted to tcspc module to start the measurement
        print(datetime.now(), '[minflux] movement', 0)

        phtime = 4  # in s, it takes 4 s for the PH to start the measurement, TO DO: check if this can be reduced (send email to Picoquant, etc)
        time.sleep(phtime)
        
        self.tcspcStartSignal.emit()
        self.xyzStartSignal.emit()
        
        self.xyMoveAndLockSignal.emit(self.r[0], self._r[0]) # singal emitted to xy and z modules to start the feedback and wait for acqtime, then move to next position
       
        print(datetime.now(), '[minflux] movement', 0)
        
    @pyqtSlot()    
    def partial_measurement(self):
        
        if self.i < self.n:
        
            time.sleep(self.acqtime)
            print(datetime.now(), '[minflux] movement', self.i)
            self.xyMoveAndLockSignal.emit(self.r[self.i], self._r[self.i])
            self.i += 1
            
        else:
            
            print(datetime.now(), '[minflux] last movement done')
            self.i = 2
        
    @pyqtSlot()  
    def get_tcspc_done_signal(self):
        
        self.xyzEndSignal.emit(self.currentfname)
        
    def make_connection(self, frontend):
        
        frontend.paramSignal.connect(self.get_frontend_param)
        frontend.filenameSignal.connect(self.get_minflux_filename)
        frontend.startButton.clicked.connect(self.start_minflux_meas)
     
        
