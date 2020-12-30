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
import tifffile

π = np.pi

DEBUG = False

class Frontend(QtGui.QFrame):
    
    paramSignal = pyqtSignal(dict)
    
    """
    Signals
    
    """
     
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        self.setup_gui()
        
    def emit_param(self):
        
        filename = os.path.join(self.folderEdit.text(),
                                self.filenameEdit.text())
        
        params = dict()
#        params['label'] = self.doughnutLabel.text()
        params['nframes'] = int(self.NframesEdit.text())
        params['filename'] = filename
        params['folder'] = self.folderEdit.text()
        params['acqtime'] = int(self.tcspcTimeEdit.text())
        
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
    
    @pyqtSlot(float)
    def get_progress_signal(self, completed):
        
        self.progressBar.setValue(completed)
        
    def setup_gui(self):
        
        self.setWindowTitle('CHECHU measurement')
        
        self.resize(230, 250)

        grid = QtGui.QGridLayout()

        self.setLayout(grid)
        self.paramWidget = QtGui.QFrame()
        self.paramWidget.setFrameStyle(QtGui.QFrame.Panel |
                                       QtGui.QFrame.Raised)
        
        self.paramWidget.setFixedHeight(180)
        self.paramWidget.setFixedWidth(170)

        grid.addWidget(self.paramWidget, 0, 0)
        
        subgrid = QtGui.QGridLayout()
        self.paramWidget.setLayout(subgrid)
        
        self.NframesLabel = QtGui.QLabel('Number of frames')
        self.NframesEdit = QtGui.QLineEdit('20')
        self.tcspcTimeLabel = QtGui.QLabel('tcspc acq time [s]')
        self.tcspcTimeEdit = QtGui.QLineEdit('20')
#        self.doughnutLabel = QtGui.QLabel('Doughnut label')
#        self.doughnutEdit = QtGui.QLineEdit('Black, Blue, Yellow, Orange')
        self.filenameLabel = QtGui.QLabel('File name')
        self.filenameEdit = QtGui.QLineEdit('chechu')
        self.startButton = QtGui.QPushButton('Start')
        self.stopButton = QtGui.QPushButton('Stop')
        self.progressBar = QtGui.QProgressBar(self)
        
        subgrid.addWidget(self.NframesLabel, 0, 0)
        subgrid.addWidget(self.NframesEdit, 1, 0)
        subgrid.addWidget(self.tcspcTimeLabel, 2, 0)
        subgrid.addWidget(self.tcspcTimeEdit, 3, 0)
        subgrid.addWidget(self.filenameLabel, 4, 0)
        subgrid.addWidget(self.filenameEdit, 5, 0)
        subgrid.addWidget(self.progressBar, 6, 0)
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
            print(datetime.now(), '[tcspc] Directory {} already exists'.format(folder))
        else:  
            print(datetime.now(), '[tcspc] Successfully created the directory {}'.format(folder))

        self.folderLabel = QtGui.QLabel('Folder')
        self.folderEdit = QtGui.QLineEdit(folder)
        self.browseFolderButton = QtGui.QPushButton('Browse')
        self.browseFolderButton.setCheckable(True)
        
        grid.addWidget(self.fileWidget, 0, 1)
        
        file_subgrid = QtGui.QGridLayout()
        self.fileWidget.setLayout(file_subgrid)
        
        file_subgrid.addWidget(self.filenameLabel, 0, 0, 1, 2)
        file_subgrid.addWidget(self.filenameEdit, 1, 0, 1, 2)
        file_subgrid.addWidget(self.folderLabel, 2, 0, 1, 2)
        file_subgrid.addWidget(self.folderEdit, 3, 0, 1, 2)
        file_subgrid.addWidget(self.browseFolderButton, 4, 0)
        
        # connections
        
        self.filenameEdit.textChanged.connect(self.emit_param)
#        self.doughnutEdit.textChanged.connect(self.emit_param)
        self.NframesEdit.textChanged.connect(self.emit_param)
        self.browseFolderButton.clicked.connect(self.load_folder)

    def make_connection(self, backend):
    
        backend.progressSignal.connect(self.get_progress_signal)
        
            
class Backend(QtCore.QObject):
    
    xySignal = pyqtSignal(bool, bool) # bool 1: whether you feedback ON or OFF, bool 2: initial position
    xyStopSignal = pyqtSignal()
    
    zSignal = pyqtSignal(bool, bool)
    zStopSignal = pyqtSignal()
    
    endSignal = pyqtSignal(str)
    
#    scanSignal = pyqtSignal(bool, str, np.ndarray)
    scanSignal = pyqtSignal(bool, str, float)
    moveToInitialSignal = pyqtSignal()
    tcspcPrepareSignal = pyqtSignal(str, int, int)
    tcspcStartSignal = pyqtSignal()

    progressSignal = pyqtSignal(float)
    
    """
    Signals
    
    """

    def __init__(self, *args, **kwargs):
    
        super().__init__(*args, **kwargs)
        
        self.i = 0
        self.n = 1
        
#        self.xyIsDone = False
        self.zIsDone = False
        self.scanIsDone = False
        
        self.measTimer = QtCore.QTimer()
        self.measTimer.timeout.connect(self.loop)

    def start(self):
        
        name = tools.getUniqueName(self.filename)
        self.timefile = open(name + '_ref_time_scan', "w+")
        
        self.tcspcPrepareSignal.emit(self.filename, self.tcspcTime, self.n)
        self.tcspcStartSignal.emit()

        self.i = 0
        
        print(datetime.now(), '[chechu] measurement started')
    
#        self.xyStopSignal.emit()
        self.zStopSignal.emit()
        
        self.moveToInitialSignal.emit()
        
        self.dataF = np.zeros((self.nFrames, self.nPixels, self.nPixels))
        self.dataB = np.zeros((self.nFrames, self.nPixels, self.nPixels))
        print(datetime.now(), '[chechu] data shape is', np.shape(self.dataF))
#        self.xy_flag = True
        self.z_flag = True
        self.scan_flag = True
    
        self.measTimer.start(0)
        
    def stop(self):
        
        self.measTimer.stop()
        self.progressSignal.emit(0)
        
        self.endSignal.emit(self.filename)
        
#        self.xyStopSignal.emit()
        self.zStopSignal.emit()
        
        print(datetime.now(), '[chechu] measurement ended')
        self.timefile.close()
        
        self.export_data()
        
    def loop(self):
        
        if self.i == 0:
            initial = True
        else:
            initial = False
                
#        if self.xy_flag:
#            
#            self.xySignal.emit(True, initial)
#            self.xy_flag = False
#            
#            if DEBUG:
#                print(datetime.now(), '[chechu] xy signal emitted ({})'.format(self.i))
            
#        if self.xyIsDone:
            
        if self.z_flag:
        
            self.zSignal.emit(True, initial)
            self.z_flag = False
            
            if DEBUG:
                print(datetime.now(), '[chechu] z signal emitted ({})'.format(self.i))

        if self.zIsDone:

            if self.scan_flag:
                    
#                initialPos = np.array([self.target_x, self.target_y, 
#                                       self.target_z], dtype=np.float64)
                
                self.timefile.write(str(datetime.now()) + ' ' + str(self.i) + '\n')
                self.timefile.write(str(time.time()) + ' ' + str(self.i) +  '\n')
                self.scanSignal.emit(True, 'chechu', self.target_z)
                self.scan_flag = False
                
                if DEBUG:
                    print(datetime.now(), 
                          '[chechu] scan signal emitted ({})'.format(self.i))
                    
            if self.scanIsDone:
                
                completed = ((self.i+1)/self.nFrames) * 100
                self.progressSignal.emit(completed)
                                
#                self.xy_flag = True
                self.z_flag = True
                self.scan_flag = True
#                self.xyIsDone = False
                self.zIsDone = False
                self.scanIsDone = False
                
                self.dataF[self.i, :, :] = self.currentFrameF
                self.dataB[self.i, :, :] = self.currentFrameB
                
                print(datetime.now(), 
                      '[chechu] frame {} of {}'.format(self.i+1, 
                                                       self.nFrames))
                                    
                if self.i < self.nFrames-1:
                
                    self.i += 1
                
                else:
                    
                    self.stop()
                    
    def export_data(self):
    
        # TO DO: export config file
        
#        fname = self.filename
#        np.savetxt(fname + '.txt', [])
        
        fname = self.filename
        filename = tools.getUniqueName(fname)

        np.savetxt(filename + '.txt', [])
        
        self.dataF = np.array(self.dataF, dtype=np.float32)
        tifffile.imsave(filename + 'F.tiff', self.dataF)
        
        self.dataB = np.array(self.dataB, dtype=np.float32)
        tifffile.imsave(filename + 'B.tiff', self.dataB)
    
    @pyqtSlot(dict)
    def get_frontend_param(self, params):
        
#        self.label = params['label']
        self.nFrames = params['nframes']
        self.tcspcTime = params['acqtime']
        
        today = str(date.today()).replace('-', '')
        self.filename = tools.getUniqueName(params['filename'] + '_' + today)
        
        print(datetime.now(), '[chechu] file name', self.filename)
                
#    @pyqtSlot(bool, float, float) 
#    def get_xy_is_done(self, val, x, y):
#        
#        """
#        Connection: [xy_tracking] xyIsDone
#        
#        """
#        self.xyIsDone = True
#        self.target_x = x
#        self.target_y = y
        
    @pyqtSlot(bool, float) 
    def get_z_is_done(self, val, z):
        
        """
        Connection: [focus] zIsDone
        
        """
        
        self.zIsDone = True     
        self.target_z = z
        
    @pyqtSlot(bool, np.ndarray, np.ndarray) 
    def get_scan_is_done(self, val, imageF, imageB):
        
        """
        Connection: [scan] scanIsDone
        
        """
        
        self.scanIsDone = True
        self.currentFrameF = imageF
        self.currentFrameB = imageB
               
    @pyqtSlot(dict) 
    def get_scan_parameters(self, params):
        
        # TO DO: this function is connected to the scan frontend, it should
        # be connected to a proper funciton in the scan backend
        
        self.nPixels = int(params['NofPixels'])
                    
        # TO DO: build config file
        
    def make_connection(self, frontend):
        
        frontend.startButton.clicked.connect(self.start)
        frontend.stopButton.clicked.connect(self.stop)
        frontend.paramSignal.connect(self.get_frontend_param)
            
            
            
            
            
        