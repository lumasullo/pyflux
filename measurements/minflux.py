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
        self.stopButton = QtGui.QPushButton('Stop')
#        self.progress = QtGui.QProgressBar(self)
        
        subgrid.addWidget(self.absolutePosLabel, 0, 0)
        subgrid.addWidget(self.absolutePosEdit, 1, 0)
        
        subgrid.addWidget(self.positionsLabel, 2, 0)
        subgrid.addWidget(self.positionsEdit, 3, 0)
        subgrid.addWidget(self.acqtimeLabel, 4, 0)
        subgrid.addWidget(self.acqtimeEdit, 5, 0)
#        subgrid.addWidget(self.progressBar, 6, 0)
        subgrid.addWidget(self.startButton, 7, 0)
        subgrid.addWidget(self.stopButton, 8, 0)
        
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
            print(datetime.now(), '[minflux] Directory {} already exists'.format(folder))
        else:  
            print(datetime.now(), '[minflux] Successfully created the directory {}'.format(folder))

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
    
    tcspcPrepareSignal = pyqtSignal(str, int, int)
    tcspcStartSignal = pyqtSignal()
    
    xyzStartSignal = pyqtSignal()
    xyzEndSignal = pyqtSignal(str)
    
    moveToSignal = pyqtSignal(np.ndarray, np.ndarray)
    
    paramSignal = pyqtSignal(np.ndarray, np.ndarray, int)
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        self.i = 0 # counter
        self.r0 = np.array([3.0, 3.0]) # TO DO: get this from scan or ADwin
        self._r = np.array([[0.035, 0.035], [0.00, 0.00], [0.07, 0.00], [0.07, 0.07], [0.00, 0.07], [0.035, 0.035]])
        self.acqtime = 5 # in s
#        self.dt = 5
        
        self.update_param()
        self.emit_param_to_frontend()
        
        self.measTimer = QtCore.QTimer()
        self.measTimer.timeout.connect(self.loop)
            
    @pyqtSlot(np.ndarray)    
    def get_ROI_center(self, center):
        
        ''' 
        Connection: [scan] ROIcenterSignal
        Description: gets the position selected by the user in [scan]
        '''
        
        self.r0 = center[0:2]
        time.sleep(0.4)
        self.xyzStartSignal.emit()
        
    @pyqtSlot(np.ndarray, np.ndarray, int)
    def get_frontend_param(self, r0, _r, acqt):
        
        """
        Connection: [frontend] paramSignal
        """
        
        self._r = np.array(_r) # relative positions
        self.r0 = np.array(r0) # offset position
        self.acqtime = acqt
                
        self.update_param()
        
    def emit_param_to_frontend(self):
        
        self.paramSignal.emit(self.r0, self._r, self.acqtime)
     
    @pyqtSlot(str) 
    def get_filename(self, val):
        
        """
        Connection: [frontend] filenameSignal
        """
        
        self.currentfname = val
        
    def update_param(self):
        
        self.r = self.r0 + self._r # absolute position
        self.n = np.shape(self._r)[0]
                
    def start(self):
        
        self.i = 0
        
        self.update_param()
        
        self.tcspcPrepareSignal.emit(self.currentfname, self.acqtime, self.n) # signal emitted to tcspc module to start the measurement

        phtime = 4  # in s, it takes 4 s for the PH to start the measurement, TO DO: check if this can be reduced (send email to Picoquant, etc)
        time.sleep(phtime)
        
        self.t0 = time.time()
        self.measTimer.start(0)
    
    def loop(self):
        
        now = time.time()
        
        if (now - (self.t0 + self.i * self.acqtime)) > self.acqtime:
            
            print(datetime.now(), '[minflux] loop', self.i)
                        
            self.moveToSignal.emit(self.r[self.i], self._r[self.i])
        
            self.i += 1
            
            if self.i == self.n:
                                
                self.stop()
                
                print(datetime.now(), '[minflux] measurement ended')

                
    def stop(self):
        
        self.measTimer.stop()
        
    @pyqtSlot()  
    def get_tcspc_done_signal(self):
        
        """
        Connection: [tcspc] tcspcDoneSignal
        """
        
        self.xyzEndSignal.emit(self.currentfname)
        
    def make_connection(self, frontend):
        
        frontend.paramSignal.connect(self.get_frontend_param)
        frontend.filenameSignal.connect(self.get_filename)
        frontend.startButton.clicked.connect(self.start)
        frontend.stopButton.clicked.connect(self.stop)
     
        
