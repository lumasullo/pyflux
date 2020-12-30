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

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QGroupBox
from tkinter import Tk, filedialog

import tools.tools as tools


π = np.pi

class Frontend(QtGui.QFrame):
    
    filenameSignal = pyqtSignal(str)
    paramSignal = pyqtSignal(dict)

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        self.setup_gui()
        
    def emit_filename(self):  
        
        filename = os.path.join(self.folderEdit.text(),
                                self.filenameEdit.text())
        
        today = str(date.today()).replace('-', '')
        filename = tools.getUniqueName(filename + '_' + today)
        
        self.filenameSignal.emit(filename)
             
    def emit_param(self):
        
        params = dict()
        
        filename = os.path.join(self.folderEdit.text(),
                                self.filenameEdit.text())
        
        params['measType'] = self.measType.currentText()
        params['acqtime'] = int(self.acqtimeEdit.text())
        params['filename'] = filename
        params['patternType'] = self.patternType.currentText()
        params['patternLength'] = float(self.lengthEdit.text())
        
        self.paramSignal.emit(params)
        
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
        
    def toggle_parameters(self):
        
        if self.measType.currentText() == 'Predefined positions':
            
            self.patternType.show()
            self.lengthLabel.show()
            self.lengthEdit.show()
      
        else:
            
            self.patternType.hide()
            self.lengthLabel.hide()
            self.lengthEdit.hide()
                    
    def setup_gui(self):
        
        self.setWindowTitle('MINFLUX measurement')

        grid = QtGui.QGridLayout()

        self.setLayout(grid)
        self.paramWidget = QGroupBox('Parameter')

        grid.addWidget(self.paramWidget, 0, 0)
        
        subgrid = QtGui.QGridLayout()
        self.paramWidget.setLayout(subgrid)
        
        self.measLabel = QtGui.QLabel('Measurement type')
        
        self.measType = QtGui.QComboBox()
        self.measTypes = ['Standard', 'Predefined positions']
        self.measType.addItems(self.measTypes)
        
        self.patternType = QtGui.QComboBox()
        self.patternTypes = ['Row', 'Square', 'Triangle']
        self.patternType.addItems(self.patternTypes)
        
        self.lengthLabel = QtGui.QLabel('L [nm]')
        self.lengthEdit = QtGui.QLineEdit('30')
        
        self.patternType.hide()
        self.lengthLabel.hide()
        self.lengthEdit.hide()
        
        self.acqtimeLabel = QtGui.QLabel('Acq time [s]')
        self.acqtimeEdit = QtGui.QLineEdit('5')
        
        self.startButton = QtGui.QPushButton('Start')
        self.stopButton = QtGui.QPushButton('Stop')

        subgrid.addWidget(self.measLabel, 0, 0, 1, 2)
        subgrid.addWidget(self.measType, 1, 0, 1, 2)
        
        subgrid.addWidget(self.patternType, 2, 0, 1, 2)
#        subgrid.addWidget(self.patternTypes, 3, 0)
        
        subgrid.addWidget(self.lengthLabel, 4, 0, 1, 1)
        subgrid.addWidget(self.lengthEdit, 4, 1, 1, 1)
        
        subgrid.addWidget(self.acqtimeLabel, 6, 0, 1, 1)
        subgrid.addWidget(self.acqtimeEdit, 6, 1, 1, 1)
        subgrid.addWidget(self.startButton, 7, 0, 1, 2)
        subgrid.addWidget(self.stopButton, 8, 0, 1, 2)
        
        # file/folder widget
        
        self.fileWidget = QGroupBox('Save options') 
        self.fileWidget.setFixedHeight(155)
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
        self.filenameEdit = QtGui.QLineEdit('minflux')
        
        grid.addWidget(self.fileWidget, 0, 1)
        
        file_subgrid = QtGui.QGridLayout()
        self.fileWidget.setLayout(file_subgrid)
        
        file_subgrid.addWidget(self.filenameLabel, 0, 0, 1, 2)
        file_subgrid.addWidget(self.filenameEdit, 1, 0, 1, 2)
        file_subgrid.addWidget(self.folderLabel, 2, 0, 1, 2)
        file_subgrid.addWidget(self.folderEdit, 3, 0, 1, 2)
        file_subgrid.addWidget(self.browseFolderButton, 4, 0)
        
        self.measType.currentIndexChanged.connect(self.toggle_parameters)
        
        self.folderEdit.textChanged.connect(self.emit_param)
        self.acqtimeEdit.textChanged.connect(self.emit_param)
        self.lengthEdit.textChanged.connect(self.emit_param)
        self.patternType.activated.connect(self.emit_param)
        
    def make_connection(self, backend):
        
#        backend.paramSignal.connect(self.get_backend_param)
        
        pass

class Backend(QtCore.QObject):
    
    tcspcPrepareSignal = pyqtSignal(str, int, int)
    tcspcStartSignal = pyqtSignal()
    
    xyzStartSignal = pyqtSignal()
    xyzEndSignal = pyqtSignal(str)
    xyStopSignal = pyqtSignal(bool)

    
    moveToSignal = pyqtSignal(np.ndarray, np.ndarray)
    
    paramSignal = pyqtSignal(np.ndarray, np.ndarray, int)
    shutterSignal = pyqtSignal(int, bool)
    
    saveConfigSignal = pyqtSignal(str)

    
    def __init__(self, adwin, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        self.i = 0 # counter
        self.n = 1
        self.adw = adwin
  
        self.pattern = np.array([0, 0])
        
        self.measTimer = QtCore.QTimer()
        self.measTimer.timeout.connect(self.loop)
            
    @pyqtSlot(np.ndarray)    
    def get_ROI_center(self, center):
        
        ''' 
        Connection: [scan] ROIcenterSignal
        Description: gets the position selected by the user in [scan]
        '''
        
        self.r0 = center[0:2]
        self.update_param()
        
        time.sleep(0.4)
        
        print(datetime.now(), '[minflux] got ROI center')
        
        #TODO delete eventually
        self.xyzStartSignal.emit()
        
    @pyqtSlot(dict)
    def get_frontend_param(self, params):
        
        """
        Connection: [frontend] paramSignal
        """
        
        self.acqtime = params['acqtime']
        self.measType = params['measType']
        
        today = str(date.today()).replace('-', '')
        self.filename = params['filename'] + '_' + today
        
        self.patternType = params['patternType']
        self.patternLength = float(params['patternLength'])/1000 # in micrometer
        
        self.update_param()
        
    def update_param(self):
        
        l = self.patternLength
        h = np.sqrt(3/2)*l
        
        currentXposition = tools.convert(self.adw.Get_FPar(70), 'UtoX')
        currentYposition = tools.convert(self.adw.Get_FPar(71), 'UtoX')
                
        self.r0 = np.array([currentXposition, currentYposition])
        
        if self.measType == 'Predefined positions':
        
            if self.patternType == 'Row':
                
                self.pattern = np.array([[0, -l], [0, 0], [0, l]])
                
                print('ROW')
                
            if self.patternType == 'Square':
                
                self.pattern = np.array([[0, 0], [l/2, l/2], [l/2, -l/2],
                                        [-l/2, -l/2], [-l/2, l/2]])
        
                print('SQUARE')
        
            if self.patternType == 'Triangle':
                
                self.pattern = np.array([[0, 0], [0, (2/3)*h], [l/2, -(1/3)*h],
                                        [-l/2, -(1/3)*h]])
        
                print('TRIANGLE')
                
            self.r = self.r0 + self.pattern
            self.n = np.shape(self.r)[0]
                
        else:
            
            self.r = self.r0
            self.n = 1
    
#        print('[minflux] self.pattern', self.pattern)
#        print(datetime.now(), '[minflux] self.r', self.r)
#        print(datetime.now(), '[minflux] self.r.shape', self.r.shape)
                
    def start(self):
        
        self.i = 0
        self.shutterSignal.emit(8, True)
        
        if self.measType == 'Standard':
            print('[minflux] self.n, self.acqtime', self.n, self.acqtime)
            self.tcspcPrepareSignal.emit(self.filename, self.acqtime, self.n) # signal emitted to tcspc module to start the measurement
#            phtime = 4.0  # in s, it takes 4 s for the PH to start the measurement, TO DO: check if this can be reduced (send email to Picoquant, etc)
#            time.sleep(phtime)
            self.tcspcStartSignal.emit()
            self.t0 = time.time()
            
        if self.measType == 'Predefined positions':
            print(datetime.now(), '[minflux] Predefined positions')
            self.update_param()
            time.sleep(0.2)
            self.tcspcPrepareSignal.emit(self.filename, self.acqtime, self.n) # signal emitted to tcspc module to start the measurement
#            phtime = 4.0  # in s, it takes 4 s for the PH to start the measurement, TO DO: check if this can be reduced (send email to Picoquant, etc)
#            time.sleep(phtime)
            self.tcspcStartSignal.emit()
            self.t0 = time.time()
            self.measTimer.start(0)
    
    def loop(self):
        
        now = time.time()
        
        if (now - (self.t0 + self.i * self.acqtime) + self.acqtime) > self.acqtime:
            
            print(datetime.now(), '[minflux] loop', self.i)
                        
            self.moveToSignal.emit(self.r[self.i], self.pattern[self.i])
        
            self.i += 1
            
            if self.i == self.n:
                                
                self.stop()
                
                print(datetime.now(), '[minflux] measurement ended')
                
    def stop(self):
        
        self.shutterSignal.emit(8, False)
        self.measTimer.stop()
        
    @pyqtSlot()  
    def get_tcspc_done_signal(self):
        
        """
        Connection: [tcspc] tcspcDoneSignal
        """
        #make scan saving config file
        self.saveConfigSignal.emit(self.filename)

        self.xyzEndSignal.emit(self.filename)
        
        self.stop()
        
        print(datetime.now(), '[minflux] measurement ended')
        
    def make_connection(self, frontend):
        
        frontend.paramSignal.connect(self.get_frontend_param)
#        frontend.filenameSignal.connect(self.get_filename)
        frontend.startButton.clicked.connect(self.start)
        frontend.stopButton.clicked.connect(self.stop)

        
