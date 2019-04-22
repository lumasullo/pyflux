# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 14:14:14 2019

@author: USUARIO
"""

import numpy as np
import time
from datetime import date, datetime
import os
import matplotlib.pyplot as plt
import tools.tools as tools
from tkinter import Tk, filedialog
import tifffile as tiff
import scipy.optimize as opt

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from pyqtgraph.dockarea import Dock, DockArea

import tools.viewbox_tools as viewbox_tools
import drivers.picoharp as picoharp
import PicoHarp.Read_PTU as Read_PTU
import drivers.ADwin as ADwin
import scan


import tools.pyqtsubclass as pyqtsc
import tools.colormaps as cmaps

import qdarkstyle
import ctypes

class Frontend(QtGui.QFrame):
    
    paramSignal = pyqtSignal(list)
    measureSignal = pyqtSignal()
        
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
                
        # initial directory

        self.initialDir = r'C:\Data'
        self.setup_gui()
        
    def start_measurement(self):
        
        self.measureSignal.emit()
        
        self.measureButton.setChecked(False)
    
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
        
    def emit_param(self):
        
        filename = os.path.join(self.folderEdit.text(),
                                self.filenameEdit.text())
        
        name = filename
        res = int(self.resolutionEdit.text())
        tacq = int(self.acqtimeEdit.text())
        folder = self.folderEdit.text()
        paramlist = [name, res, tacq, folder]
        
        self.paramSignal.emit(paramlist)
        
    @pyqtSlot(float, float)
    def get_backend_parameters(self, cts0, cts1):
        
        self.channel0Label.setText(('Input0 (sync) = {} c/s'.format(cts0)))
        self.channel1Label.setText(('Input1 (APD) = {} c/s'.format(cts1)))
    
    @pyqtSlot(np.ndarray, np.ndarray)    
    def plot_data(self, relTime, absTime):
        
        counts, bins = np.histogram(relTime, bins=50) # TO DO: choose proper binning
        self.histPlot.plot(bins[0:-1], counts)

#        plt.hist(relTime, bins=300)
#        plt.xlabel('time (ns)')
#        plt.ylabel('ocurrences')

        timetrace, time = np.histogram(absTime, bins=50) # timetrace with 50 bins

        self.tracePlot.plot(time[0:-1], timetrace)

#        plt.plot(time[0:-1], timetrace)
#        plt.xlabel('time (ms)')
#        plt.ylabel('counts')
                
    def clear_data(self):
        
        self.histPlot.clear()
        self.tracePlot.clear()
    
    def make_connection(self, backend):
        
        backend.ctRatesSignal.connect(self.get_backend_parameters)
        backend.plotDataSignal.connect(self.plot_data)
        
    def  setup_gui(self):
        
        # widget with tcspc parameters

        self.paramWidget = QtGui.QFrame()
        self.paramWidget.setFrameStyle(QtGui.QFrame.Panel |
                                       QtGui.QFrame.Raised)
        
        self.paramWidget.setFixedHeight(400)
        self.paramWidget.setFixedWidth(250)
        
#        phParamTitle = QtGui.QLabel('<h2><strong>TCSPC settings</strong></h2>')
        phParamTitle = QtGui.QLabel('<h2>TCSPC settings</h2>')
        phParamTitle.setTextFormat(QtCore.Qt.RichText)
        
        # widget to display data
        
        self.dataWidget = pg.GraphicsLayoutWidget()
        
        # Shutter button
        
        self.shutterButton = QtGui.QPushButton('Shutter open/close')
        self.shutterButton.setCheckable(True)
        
        # Measure button

        self.measureButton = QtGui.QPushButton('Measure')
        self.measureButton.setCheckable(True)
        
        # forced stop measurement
        
        self.stopButton = QtGui.QPushButton('Stop')
        
        # exportData button
        
        self.exportDataButton = QtGui.QPushButton('Export data')
#        self.exportDataButton.setCheckable(True)
        
        # Clear data
        
        self.clearButton = QtGui.QPushButton('Clear data')
#        self.clearButton.setCheckable(True)
        
        # TCSPC parameters

        self.acqtimeLabel = QtGui.QLabel('Acquisition time (s)')
        self.acqtimeEdit = QtGui.QLineEdit('1')
        self.resolutionLabel = QtGui.QLabel('Resolution (ps)')
        self.resolutionEdit = QtGui.QLineEdit('8')
        self.offsetLabel = QtGui.QLabel('Offset (ns)')
        self.offsetEdit = QtGui.QLineEdit('0')
        self.channel0Label = QtGui.QLabel('Input0 (sync) = --- c/s')
        self.channel1Label = QtGui.QLabel('Input1 (APD) = --- c/s')
        
        self.filenameLabel = QtGui.QLabel('File name')
        self.filenameEdit = QtGui.QLineEdit('filename')
        
        # microTime histogram and timetrace
        
        self.histPlot = self.dataWidget.addPlot(row=1, col=0, title="microTime histogram")
        self.histPlot.setLabels(bottom=('ns'),
                                left=('counts'))
        
        self.tracePlot = self.dataWidget.addPlot(row=2, col=0, title="Time trace")
        self.tracePlot.setLabels(bottom=('ms'),
                                left=('counts'))
        
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
        
        # GUI connections
        
        self.measureButton.clicked.connect(self.start_measurement)
        self.browseFolderButton.clicked.connect(self.load_folder)
        self.clearButton.clicked.connect(self.clear_data)    
        
        self.acqtimeEdit.textChanged.connect(self.emit_param)
        self.resolutionEdit.textChanged.connect(self.emit_param)

        # GUI layout

        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.paramWidget, 0, 0)
        grid.addWidget(self.dataWidget, 0, 1)
        
        subgrid = QtGui.QGridLayout()
        self.paramWidget.setLayout(subgrid)
        subgrid.addWidget(phParamTitle, 0, 0, 2, 3)

        subgrid.addWidget(self.acqtimeLabel, 2, 0)
        subgrid.addWidget(self.acqtimeEdit, 2, 1)
        subgrid.addWidget(self.resolutionLabel, 4, 0)
        subgrid.addWidget(self.resolutionEdit, 4, 1)
        subgrid.addWidget(self.offsetLabel, 6, 0)
        subgrid.addWidget(self.offsetEdit, 6, 1)
        subgrid.addWidget(self.channel0Label, 8, 0)
        subgrid.addWidget(self.channel1Label, 9, 0)
        
        subgrid.addWidget(self.filenameLabel, 11, 0)
        subgrid.addWidget(self.filenameEdit, 11, 1)
        subgrid.addWidget(self.folderLabel, 13, 0)
        subgrid.addWidget(self.folderEdit, 13, 1)
        subgrid.addWidget(self.browseFolderButton, 15, 0)
        
        subgrid.addWidget(self.shutterButton, 17, 0)
        subgrid.addWidget(self.measureButton, 18, 0)
        subgrid.addWidget(self.stopButton, 19, 0)
#        subgrid.addWidget(self.exportDataButton, 20, 0)
        subgrid.addWidget(self.clearButton, 21, 0)
        
class Backend(QtCore.QObject):

    ctRatesSignal = pyqtSignal(float, float)
    plotDataSignal = pyqtSignal(np.ndarray, np.ndarray)
    
#    xyzSignal = pyqtSignal(bool, str)
    tcspcDoneSignal = pyqtSignal()
    
    def __init__(self, ph_device, adwin, *args, **kwargs): 
        
        super().__init__(*args, **kwargs)
          
        self.ph = ph_device 
        self.adw = adwin
        self.shutter_state = False
        
    def measure_count_rate(self):
        
        # TO DO: fix this method to update cts0 and cts1 automatically
        
        pass
          
    def prepare_ph(self):
        
        self.ph.open()
        self.ph.initialize()
        self.ph.setup()
        
        self.ph.syncDivider = 4 # this parameter must be set such that the count rate at channel 0 (sync) is equal or lower than 10MHz
        self.ph.resolution = self.resolution # desired resolution in ps
        self.ph.offset = 0
        self.ph.tacq = self.tacq * 1000 # time in ms
        
        self.cts0 = self.ph.countrate(0)
        self.cts1 = self.ph.countrate(1)

        self.ctRatesSignal.emit(self.cts0, self.cts1)
  
        print(datetime.now(), '[tcspc] Resolution = {} ps'.format(self.ph.resolution))
        print(datetime.now(), '[tcspc] Acquisition time = {} s'.format(self.ph.tacq))
    
    @pyqtSlot()           
    def measure(self):
        
        t0 = time.time()

        self.currentfname = tools.getUniqueName(self.fname)
        
        delay = 4.0 # 4.0 s is the typical time that the PH takes to start a measurement
        
#        self.xyzSignal.emit(True, self.currentfname)
        
        self.prepare_ph()
        self.ph.lib.PH_SetBinning(ctypes.c_int(0), 
                                  ctypes.c_int(1)) # TO DO: fix this in a clean way (1 = 8 ps resolution)

        t1 = time.time()
        
        print(datetime.now(), '[tcspc] starting the PH measurement took {} s'.format(t1-t0))

        self.ph.startTTTR(self.currentfname)
        np.savetxt(self.currentfname + '.txt', [])
        
        while self.ph.measure_state is not 'done':
            pass
        
#        self.xyzSignal.emit(False, self.currentfname)
        self.export_data()
        
    @pyqtSlot(str, int, int)
    def prepare_minflux(self, fname, acqtime, n):
        
        print(datetime.now(), ' [tcspc] preparing minflux measurement')
        
        t0 = time.time()

        self.currentfname = tools.getUniqueName(fname)
                
#        self.xyzSignal.emit(True, self.currentfname)
        
        self.prepare_ph()
        
        self.ph.tacq = acqtime * n * 1000 # TO DO: correspond to GUI !!!
        
        print(' [tcspc] self.ph.tacq', self.ph.tacq)
        
        self.ph.lib.PH_SetBinning(ctypes.c_int(0), 
                                  ctypes.c_int(1)) # TO DO: fix this in a clean way (1 = 8 ps resolution)
   
        t1 = time.time()
        
        print(datetime.now(), '[tcspc] preparing the PH measurement took {} s'.format(t1-t0))
        
        
    @pyqtSlot()
    def measure_minflux(self):

        self.ph.startTTTR(self.currentfname)
        
        np.savetxt(self.currentfname + '.txt', [])
        
        while self.ph.measure_state is not 'done':
            pass
        
        self.tcspcDoneSignal.emit()
        self.export_data()
        
    def stop_measure(self):
        
        # TO DO: make this function
        
        print(datetime.now(), '[tcspc] stop measure function')

    def export_data(self):
        
#        self.reset_data()
        
        inputfile = open(self.currentfname, "rb") # TO DO: fix file selection
        print(datetime.now(), '[tcspc] opened {} file'.format(self.currentfname))
        
        numRecords = self.ph.numRecords # number of records
        globRes = 2.5e-8  # in ns, corresponds to sync @40 MHz
        timeRes = self.ph.resolution * 1e-12 # time resolution in s
#        print(timeRes)

        relTime, absTime = Read_PTU.readPT3(inputfile, numRecords)

        inputfile.close()
        
        relTime = relTime * timeRes # in real time units (s)
        self.relTime = relTime * 1e9  # in (ns)

        absTime = absTime * globRes * 1e9  # true time in (ns), 4 comes from syncDivider, 10 Mhz = 40 MHz / syncDivider 
        self.absTime = absTime / 1e6 # in ms

        filename = self.currentfname + '_arrays.txt'
        
        datasize = np.size(self.absTime[self.absTime != 0])
        data = np.zeros((2, datasize))
        
        data[0, :] = self.relTime[self.absTime != 0]
        data[1, :] = self.absTime[self.absTime != 0]
        
        self.plotDataSignal.emit(data[0, :], data[1, :])
        
        np.savetxt(filename, data.T) # transpose for easier loading
        
        print(datetime.now(), '[tcspc] tcspc data exported')
        
    @pyqtSlot(list)
    def get_frontend_parameters(self, paramlist):
        
        print(datetime.now(), '[tcspc] got frontend parameters')

        self.fname = paramlist[0]
        self.resolution = paramlist[1]
        self.tacq = paramlist[2]
        self.folder = paramlist[3]      
        
    @pyqtSlot(bool)
    def toggle_shutter(self, val):
        
        if val is True:
            
            self.shutter_state = True
            
            self.adw.Set_Par(55, 0)
            self.adw.Set_Par(50, 1)
            self.adw.Set_Par(57, 1)
            self.adw.Start_Process(5)
            
            print(datetime.now(), '[tcspc] Shutter opened')
            
        if val is False:
            
            self.shutte_state = False
            
            self.adw.Set_Par(55, 0)
            self.adw.Set_Par(50, 0)
            self.adw.Set_Par(57, 1)
            self.adw.Start_Process(5)

            print(datetime.now(), '[tcspc] Shutter closed')

    def make_connection(self, frontend):

        frontend.paramSignal.connect(self.get_frontend_parameters)
        frontend.measureSignal.connect(self.measure)
        frontend.stopButton.clicked.connect(self.stop_measure)
        frontend.shutterButton.clicked.connect(lambda: self.toggle_shutter(frontend.shutterButton.isChecked()))

        frontend.emit_param() # TO DO: change such that backend has parameters defined from the start

    def stop(self):
  
        pass


if __name__ == '__main__':

    app = QtGui.QApplication([])
    app.setStyle(QtGui.QStyleFactory.create('fusion'))
#    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    DEVICENUMBER = 0x1
    adw = ADwin.ADwin(DEVICENUMBER, 1)
    scan.setupDevice(adw)

    ph = picoharp.PicoHarp300()
    worker = Backend(ph, adw)
    gui = Frontend()
    
    workerThread = QtCore.QThread()
    workerThread.start()
    worker.moveToThread(workerThread)

    worker.make_connection(gui)
    gui.make_connection(worker)

    gui.setWindowTitle('Time-correlated single-photon counting')
    gui.show()

    app.exec_()
    