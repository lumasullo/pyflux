# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:38:16 2019

@author: Luciano A. Masullo
"""

import numpy as np
import os
from datetime import date, datetime
import time

from pyqtgraph.Qt import QtCore, QtGui

from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QGroupBox
from tkinter import Tk, filedialog

import tools.tools as tools
import imageio as iio 

π = np.pi

DEBUG = True

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
        params['label'] = self.doughnutLabel.text()
        params['nframes'] = self.NframesEdit.value() #!!! now making up number of frames per doughnut position,
                                                     #no longer total number of images!!!
        params['filename'] = filename
        params['folder'] = self.folderEdit.text()
        params['nDonuts'] = self.donutSpinBox.value()
        params['alignMode'] = self.activateModeCheckbox.isChecked()
        
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
        
        if completed == 100:
            self.stopButton.setEnabled(False)
            self.startButton.setEnabled(True)
            
    def activate_alignmentmode(self, on):
        if on:
            self.shutter1Checkbox.setEnabled(True)
            self.shutter2Checkbox.setEnabled(True)
            self.shutter3Checkbox.setEnabled(True)
            self.shutter4Checkbox.setEnabled(True)
            print(datetime.now(), '[psf] Alignment mode activated')
        else:
            self.shutter1Checkbox.setEnabled(False)
            self.shutter2Checkbox.setEnabled(False)
            self.shutter3Checkbox.setEnabled(False)
            self.shutter4Checkbox.setEnabled(False)
            print(datetime.now(), '[psf] Alignment mode deactivated')
            
        
    
    def setup_gui(self):
        
        self.setWindowTitle('PSF measurement')
        
        self.resize(230, 300)

        grid = QtGui.QGridLayout()

        self.setLayout(grid)
        self.paramWidget = QGroupBox('Parameter')
        self.paramWidget.setMinimumHeight(250)
        self.paramWidget.setFixedWidth(175)

        grid.addWidget(self.paramWidget, 0, 0, 2, 1)
        
        subgrid = QtGui.QGridLayout()
        self.paramWidget.setLayout(subgrid)
        
        self.NframesLabel = QtGui.QLabel('Frames per doughnut')
        self.NframesEdit = QtGui.QSpinBox()
        self.DonutNumLabel = QtGui.QLabel('Number of doughnuts')
        self.donutSpinBox = QtGui.QSpinBox()
        self.doughnutLabel = QtGui.QLabel('Doughnut label')
        self.doughnutEdit = QtGui.QLineEdit('Black, Blue, Yellow, Orange')
        self.filenameLabel = QtGui.QLabel('File name')
        self.filenameEdit = QtGui.QLineEdit('psf')
        self.startButton = QtGui.QPushButton('Start')
        self.stopButton = QtGui.QPushButton('Stop')
        self.stopButton.setEnabled(False)
        self.progressBar = QtGui.QProgressBar(self)
        
        self.NframesEdit.setValue(5)
        self.NframesEdit.setRange(1,99)
        self.donutSpinBox.setValue(4)
        self.donutSpinBox.setMaximum(10)
        
        subgrid.addWidget(self.DonutNumLabel, 0, 0)
        subgrid.addWidget(self.donutSpinBox, 1, 0)       
        subgrid.addWidget(self.NframesLabel, 2, 0)
        subgrid.addWidget(self.NframesEdit, 3, 0)
        subgrid.addWidget(self.doughnutLabel, 4, 0)
        subgrid.addWidget(self.doughnutEdit, 5, 0)
        subgrid.addWidget(self.filenameLabel, 6, 0)
        subgrid.addWidget(self.filenameEdit, 7, 0)
        subgrid.addWidget(self.progressBar, 8, 0)
        subgrid.addWidget(self.startButton, 9, 0)
        subgrid.addWidget(self.stopButton, 10, 0)
        
        # file/folder widget
        
        self.fileWidget = QGroupBox('Save options')  
        self.fileWidget.setFixedHeight(155)
        self.fileWidget.setFixedWidth(150)
        
        # folder
        
        # TO DO: move this to backend
        
        today = str(date.today()).replace('-', '')
        root = r'C:\\Data\\'
        folder = root + today
        self.initialDir = folder
        
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
        
        grid.addWidget(self.fileWidget, 0, 1, 1, 1)
        
        file_subgrid = QtGui.QGridLayout()
        self.fileWidget.setLayout(file_subgrid)
        
        file_subgrid.addWidget(self.filenameLabel, 0, 0, 1, 2)
        file_subgrid.addWidget(self.filenameEdit, 1, 0, 1, 2)
        file_subgrid.addWidget(self.folderLabel, 2, 0, 1, 2)
        file_subgrid.addWidget(self.folderEdit, 3, 0, 1, 2)
        file_subgrid.addWidget(self.browseFolderButton, 4, 0)
        
        #setup alignment mode widget
        self.alignWidget = QGroupBox('Alignment mode')  
        self.alignWidget.setFixedHeight(110)
        self.alignWidget.setFixedWidth(150)
        
        grid.addWidget(self.alignWidget, 1, 1, 1, 1)
        
        align_subgrid = QtGui.QGridLayout()
        self.alignWidget.setLayout(align_subgrid)
        
        self.activateModeCheckbox = QtGui.QCheckBox('Mode Activated')
        self.shutter1Checkbox = QtGui.QCheckBox('1')
        self.shutter2Checkbox = QtGui.QCheckBox('2')
        self.shutter3Checkbox = QtGui.QCheckBox('3')
        self.shutter4Checkbox = QtGui.QCheckBox('4')
        
        self.checkboxGroup = QtGui.QButtonGroup(self)
        self.checkboxGroup.addButton(self.shutter1Checkbox)
        self.checkboxGroup.addButton(self.shutter2Checkbox)
        self.checkboxGroup.addButton(self.shutter3Checkbox)
        self.checkboxGroup.addButton(self.shutter4Checkbox)
        
        self.shutter1Checkbox.setEnabled(False)
        self.shutter2Checkbox.setEnabled(False)
        self.shutter3Checkbox.setEnabled(False)
        self.shutter4Checkbox.setEnabled(False)
        
        align_subgrid.addWidget(self.activateModeCheckbox, 0, 0, 1, 2)
        align_subgrid.addWidget(self.shutter1Checkbox, 1, 0)
        align_subgrid.addWidget(self.shutter2Checkbox, 2, 0)
        align_subgrid.addWidget(self.shutter3Checkbox, 1, 1)
        align_subgrid.addWidget(self.shutter4Checkbox, 2, 1)
        
        # connections
        
        self.startButton.clicked.connect(self.emit_param)
        self.startButton.clicked.connect(lambda: self.stopButton.setEnabled(True))
        self.startButton.clicked.connect(lambda: self.startButton.setEnabled(False))
        self.stopButton.clicked.connect(lambda: self.startButton.setEnabled(True))
        self.browseFolderButton.clicked.connect(self.load_folder)
        self.activateModeCheckbox.clicked.connect(lambda: self.activate_alignmentmode(self.activateModeCheckbox.isChecked()))

    def make_connection(self, backend):
    
        backend.progressSignal.connect(self.get_progress_signal)
      
    def closeEvent(self, *args, **kwargs):
        self.progressBar.setValue(0)
        super().closeEvent(*args, **kwargs)
            
class Backend(QtCore.QObject):
    
    xySignal = pyqtSignal(bool, bool) # bool 1: whether you feedback ON or OFF, bool 2: initial position
    xyStopSignal = pyqtSignal(bool)
    
    zSignal = pyqtSignal(bool, bool)
    zStopSignal = pyqtSignal()
    
    endSignal = pyqtSignal(str)
    
    scanSignal = pyqtSignal(bool, str, np.ndarray)
    moveToInitialSignal = pyqtSignal()

    progressSignal = pyqtSignal(float)
    
    shutterSignal = pyqtSignal(int, bool)
    
    saveConfigSignal = pyqtSignal(str)
    
    """
    Signals
    
    """

    def __init__(self, *args, **kwargs):
    
        super().__init__(*args, **kwargs)
        
        self.i = 0
        
        self.xyIsDone = False
        self.zIsDone = False
        self.scanIsDone = False
        
        self.measTimer = QtCore.QTimer()
        self.measTimer.timeout.connect(self.loop)
        
        self.checkboxID_old = 7
        self.alignMode = False

    def start(self):
        
        self.i = 0
        
        self.xyIsDone = False
        self.zIsDone = False
        self.scanIsDone = False
        
        self.progressSignal.emit(0)
        
        self.shutterSignal.emit(7, False)
        self.shutterSignal.emit(11, False)
        
        print(datetime.now(), '[psf] PSF measurement started')
          
        self.xyStopSignal.emit(True)
        self.zStopSignal.emit()
        
        #open IR and tracking shutter
        self.shutterSignal.emit(5, True)
        self.shutterSignal.emit(6, True)
        
        self.moveToInitialSignal.emit()

        
        self.data = np.zeros((self.totalFrameNum, self.nPixels, self.nPixels))
        print(datetime.now(), '[psf] Data shape is', np.shape(self.data))
        self.xy_flag = True
        self.z_flag = True
        self.scan_flag = True
    
        self.measTimer.start(0)
        
    def stop(self):
        
        self.measTimer.stop()
        self.progressSignal.emit(100) #changed from 0
        self.shutterSignal.emit(8, False)
        
        #new filename indicating that getUniqueName() has already found filename
        #rerunning would only cause errors in files being saved by focus and xy_tracking
        attention_filename = '!' + self.filename
        self.endSignal.emit(attention_filename)
        
        self.xyStopSignal.emit(False)
        self.zStopSignal.emit()
        
        self.export_data()
        
        print(datetime.now(), '[psf] PSF measurement ended')
        
    def loop(self):
        
        if self.i == 0:
            initial = True
        else:
            initial = False
      
        if self.xy_flag:
            
            self.xySignal.emit(True, initial)
            self.xy_flag = False
            
            if DEBUG:
                print(datetime.now(), '[psf] xy signal emitted ({})'.format(self.i))
            
        if self.xyIsDone:
            
            if self.z_flag:
                      
                self.zSignal.emit(True, initial)
                self.z_flag = False
                
                if DEBUG:
                    print(datetime.now(), '[psf] z signal emitted ({})'.format(self.i))

            if self.zIsDone:
                
                shutternum = self.i // self.nFrames + 1
    
                if self.scan_flag:
                    
                    if not self.alignMode:
                        self.shutterSignal.emit(shutternum, True)
                        
                    initialPos = np.array([self.target_x, self.target_y, 
                                           self.target_z], dtype=np.float64)
                      
                    self.scanSignal.emit(True, 'frame', initialPos)
                    self.scan_flag = False
                    
                    if DEBUG:
                        print(datetime.now(), 
                              '[psf] scan signal emitted ({})'.format(self.i))
                        
                if self.scanIsDone:
                   
                    if not self.alignMode:
                        self.shutterSignal.emit(shutternum, False)
                    
                    completed = ((self.i+1)/self.totalFrameNum) * 100
                    self.progressSignal.emit(completed)
                                    
                    self.xy_flag = True
                    self.z_flag = True
                    self.scan_flag = True
                    self.xyIsDone = False
                    self.zIsDone = False
                    self.scanIsDone = False
                    
                    self.data[self.i, :, :] = self.currentFrame
                    
                    print(datetime.now(), 
                          '[psf] PSF {} of {}'.format(self.i+1, 
                                                      self.totalFrameNum))
                                        
                    if self.i < self.totalFrameNum-1:
                    
                        self.i += 1
                    
                    else:
                        
                        self.stop()
                                            
    def export_data(self):
  
        fname = self.filename
        filename = tools.getUniqueName(fname)

        self.data = np.array(self.data, dtype=np.float32)
        
        iio.mimwrite(filename + '.tiff', self.data)
        
        #make scan saving config file
        self.saveConfigSignal.emit(filename)
    
    @pyqtSlot(dict)
    def get_frontend_param(self, params):
        
        self.label = params['label']
        self.nFrames = params['nframes']
        self.k = params['nDonuts']
        
        today = str(date.today()).replace('-', '')
        self.filename = tools.getUniqueName(params['filename'] + '_' + today)
         
        self.totalFrameNum = self.nFrames * self.k
        
        self.alignMode = params['alignMode']
                
    @pyqtSlot(bool, float, float) 
    def get_xy_is_done(self, val, x, y):
        
        """
        Connection: [xy_tracking] xyIsDone
        
        """
        self.xyIsDone = True
        self.target_x = x
        self.target_y = y
        
    @pyqtSlot(bool, float) 
    def get_z_is_done(self, val, z):
        
        """
        Connection: [focus] zIsDone
        
        """
        
        self.zIsDone = True     
        self.target_z = z
        
    @pyqtSlot(bool, np.ndarray) 
    def get_scan_is_done(self, val, image):
        
        """
        Connection: [scan] scanIsDone
        
        """
        
        self.scanIsDone = True
        self.currentFrame = image
               
    @pyqtSlot(dict) 
    def get_scan_parameters(self, params):
        
        # TO DO: this function is connected to the scan frontend, it should
        # be connected to a proper funciton in the scan backend
        
        self.nPixels = int(params['NofPixels'])
                    
        # TO DO: build config file
        
    # button_clicked slot
    @pyqtSlot(QtGui.QAbstractButton)
    #@pyqtSlot(int)
    def checkboxGroup_selection(self, button_or_id):
        
        self.shutterSignal.emit(self.checkboxID_old, False)
        
        #if isinstance(button_or_id, QtGui.QAbstractButton):
        checkboxID = int(button_or_id.text())
        #print('Checkbox {} was selected'.format(checkboxID))
        self.shutterSignal.emit(checkboxID, True)
        
        self.checkboxID_old = checkboxID
        #elif isinstance(button_or_id, int):
#        print('"Id {}" was clicked'.format(button_or_id))

        
    def make_connection(self, frontend):
        
        frontend.startButton.clicked.connect(self.start)
        frontend.stopButton.clicked.connect(self.stop)
        frontend.paramSignal.connect(self.get_frontend_param)
        frontend.checkboxGroup.buttonClicked['QAbstractButton *'].connect(self.checkboxGroup_selection)    
            
            
            
        