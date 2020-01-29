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
#        self.measureButton.setChecked(True) TO DO: signal from backend that toggles button
    
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
        
        # TO DO: change for dictionary
        
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
        
        # conversion to kHz
        
        cts0_khz = cts0/1000 
        cts1_khz = cts1/1000
        
        self.channel0Value.setText(('{}'.format(cts0_khz)))
        self.channel1Value.setText(('{}'.format(cts1_khz)))
    
    @pyqtSlot(np.ndarray, np.ndarray)    
    def plot_data(self, relTime, absTime):
        
        counts, bins = np.histogram(relTime, bins=50) # TO DO: choose proper binning
        self.histPlot.plot(bins[0:-1], counts)

#        plt.hist(relTime, bins=300)
#        plt.xlabel('time (ns)')
#        plt.ylabel('ocurrences')

        counts, time = np.histogram(absTime, bins=50) # timetrace with 50 bins
        
        binwidth = time[-1]/50
        timetrace_khz = counts/binwidth 

        self.tracePlot.plot(time[0:-1], timetrace_khz)

#        plt.plot(time[0:-1], timetrace)
        self.tracePlot.setLabels(bottom=('Time', 'ms'),
                                        left=('Count rate', 'kHz'))
                
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
        
        self.paramWidget.setFixedHeight(245)
        self.paramWidget.setFixedWidth(230)
        
#        phParamTitle = QtGui.QLabel('<h2><strong>TCSPC settings</strong></h2>')
        phParamTitle = QtGui.QLabel('<h2>TCSPC settings</h2>')
        phParamTitle.setTextFormat(QtCore.Qt.RichText)
        
        # widget to display data
        
        self.dataWidget = pg.GraphicsLayoutWidget()
        
        # file/folder widget
        
        self.fileWidget = QtGui.QFrame()
        self.fileWidget.setFrameStyle(QtGui.QFrame.Panel |
                                      QtGui.QFrame.Raised)
        
        self.fileWidget.setFixedHeight(125)
        self.fileWidget.setFixedWidth(230)
        
        # Shutter button
        
        self.shutterButton = QtGui.QPushButton('Shutters open/close')
        self.shutterButton.setCheckable(True)
        
        # Prepare button
        
        self.prepareButton = QtGui.QPushButton('Prepare TTTR')
        
        # Measure button

        self.measureButton = QtGui.QPushButton('Measure TTTR')
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

        self.acqtimeLabel = QtGui.QLabel('Acquisition time [s]')
        self.acqtimeEdit = QtGui.QLineEdit('1')
        self.resolutionLabel = QtGui.QLabel('Resolution [ps]')
        self.resolutionEdit = QtGui.QLineEdit('8')
        self.offsetLabel = QtGui.QLabel('Offset [ns]')
        self.offsetEdit = QtGui.QLineEdit('0')
        
        self.channel0Label = QtGui.QLabel('Input 0 (sync) [kHz]')
        self.channel0Value = QtGui.QLineEdit('')
        self.channel0Value.setReadOnly(True)
        
        self.channel1Label = QtGui.QLabel('Input 1 (APD) [kHz]')
        self.channel1Value = QtGui.QLineEdit('')
        self.channel1Value.setReadOnly(True)
        
        self.filenameLabel = QtGui.QLabel('File name')
        self.filenameEdit = QtGui.QLineEdit('minfluxfile')
        
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

        # general GUI layout

        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.paramWidget, 0, 0)
        grid.addWidget(self.fileWidget, 1, 0)
        grid.addWidget(self.dataWidget, 0, 1, 2, 2)
        
        # param Widget layout
        
        subgrid = QtGui.QGridLayout()
        self.paramWidget.setLayout(subgrid)
        #subgrid.addWidget(phParamTitle, 0, 0, 2, 3)

        subgrid.addWidget(self.acqtimeLabel, 2, 0)
        subgrid.addWidget(self.acqtimeEdit, 2, 1)
        subgrid.addWidget(self.resolutionLabel, 4, 0)
        subgrid.addWidget(self.resolutionEdit, 4, 1)
        subgrid.addWidget(self.offsetLabel, 6, 0)
        subgrid.addWidget(self.offsetEdit, 6, 1)
        subgrid.addWidget(self.channel0Label, 8, 0)
        subgrid.addWidget(self.channel0Value, 8, 1)
        subgrid.addWidget(self.channel1Label, 9, 0)
        subgrid.addWidget(self.channel1Value, 9, 1)
        
        subgrid.addWidget(self.measureButton, 17, 0)
        subgrid.addWidget(self.prepareButton, 18, 0)
        #subgrid.addWidget(self.shutterButton, 19, 0)
        subgrid.addWidget(self.stopButton, 17, 1)
        subgrid.addWidget(self.clearButton, 18, 1)
        
        file_subgrid = QtGui.QGridLayout()
        self.fileWidget.setLayout(file_subgrid)
        
        file_subgrid.addWidget(self.filenameLabel, 0, 0, 1, 2)
        file_subgrid.addWidget(self.filenameEdit, 1, 0, 1, 2)
        file_subgrid.addWidget(self.folderLabel, 2, 0, 1, 2)
        file_subgrid.addWidget(self.folderEdit, 3, 0, 1, 2)
        file_subgrid.addWidget(self.browseFolderButton, 4, 0)

    def closeEvent(self, *args, **kwargs):
        
        workerThread.exit()
        super().closeEvent(*args, **kwargs)
        app.quit()
        
        
class Backend(QtCore.QObject):

    ctRatesSignal = pyqtSignal(float, float)
    plotDataSignal = pyqtSignal(np.ndarray, np.ndarray)
    
    tcspcDoneSignal = pyqtSignal()
    
    def __init__(self, ph_device, adwin, *args, **kwargs): 
        
        super().__init__(*args, **kwargs)
          
        self.ph = ph_device 
        self.adw = adwin
        
        self.tcspcTimer = QtCore.QTimer()
       
    def update(self):
        #TODO: fix this to make countrate being properly displayed apparently it 
        #is an buffer issue because its working in the tttstart loop but not 
        #when called from here
        pass
        
    def measure_count_rate(self):
        
        # TO DO: fix this method to update cts0 and cts1 automatically
        self.cts0 = self.ph.countrate(0)
        self.cts1 = self.ph.countrate(1)

        self.ctRatesSignal.emit(self.cts0, self.cts1)
          
    def prepare_ph(self):
        
        self.ph.open()
        self.ph.initialize()
        self.ph.setup_ph300()
        self.ph.setup_phr800()
        
        self.ph.syncDivider = 4 # this parameter must be set such that the count rate at channel 0 (sync) is equal or lower than 10MHz
        self.ph.resolution = self.resolution # desired resolution in ps
        
        self.ph.lib.PH_SetBinning(ctypes.c_int(0), 
                                  ctypes.c_int(1)) # TO DO: fix this in a clean way (1 = 8 ps resolution)

        self.ph.offset = 0
        self.ph.tacq = self.tacq * 1000 # time in ms
        
        self.cts0 = self.ph.countrate(0)
        self.cts1 = self.ph.countrate(1)

        self.ctRatesSignal.emit(self.cts0, self.cts1)
  
        print(datetime.now(), '[tcspc] Resolution = {} ps'.format(self.ph.resolution))
        print(datetime.now(), '[tcspc] Acquisition time = {} s'.format(self.ph.tacq))
    
        print(datetime.now(), '[tcspc] Picoharp 300 prepared for TTTR measurement')
    
    @pyqtSlot()           
    def measure(self):
        
        # WARNING: run first once with commercial software to make it work
        # TO DO: fix above
        
        t0 = time.time()
        
        self.prepare_ph()

        self.currentfname = tools.getUniqueName(self.fname)

        t1 = time.time()
        
        print(datetime.now(), '[tcspc] starting the PH measurement took {} s'.format(t1-t0))

        #TODO check timer value
        self.tcspcTimer.start(3000)
        
        self.ph.startTTTR(self.currentfname)
        np.savetxt(self.currentfname + '.txt', [])
        
        while self.ph.measure_state != 'done':
            pass
        
        self.export_data()
                
    @pyqtSlot(str, int, int)
    def prepare_minflux(self, fname, acqtime, n):
        
        print(datetime.now(), ' [tcspc] preparing minflux measurement')
        
        t0 = time.time()
        
        self.currentfname = tools.getUniqueName(fname)
                        
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
        
        print(datetime.now(), '[tcspc] minflux measurement started')
                
        while self.ph.measure_state != 'done':
            pass
        
        self.tcspcDoneSignal.emit()
        self.export_data()
        
    def stop_measure(self):
        
        # TO DO: make this function, not so easy because the while loop in the driver
        # HINT: maybe use a timer loop to be able to access variables within the loop
        
        self.tcspcTimer.stop()
        print(datetime.now(), '[tcspc] stop measure function (empty)')

    def export_data(self):
        
#        self.reset_data()
        
        inputfile = open(self.currentfname, "rb") # TO DO: fix file selection
        print(datetime.now(), '[tcspc] opened {} file'.format(self.currentfname))
        
        numRecords = self.ph.numRecords # number of records
        globRes = 2.5e-8  # in ns, corresponds to sync @40 MHz
        timeRes = self.ph.resolution * 1e-12 # time resolution in s

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
        
        np.savetxt(self.currentfname + '.txt', []) # TO DO: build config file, now empty file

    @pyqtSlot(list)
    def get_frontend_parameters(self, paramlist):
        
        print(datetime.now(), '[tcspc] got frontend parameters')

        self.fname = paramlist[0]
        self.resolution = paramlist[1]
        self.tacq = paramlist[2]
        self.folder = paramlist[3]      
            
    @pyqtSlot(bool)
    def toggle_minflux_shutters(self, val):
        if val is True:
            tools.toggle_shutter(self.adw, 7, True)
            print(datetime.now(), '[tcspc] Minflux shutters opened')
            
        if val is False:
            tools.toggle_shutter(self.adw, 7, False)
            print(datetime.now(), '[tcspc] Minflux shutters closed')

    def make_connection(self, frontend):

        frontend.paramSignal.connect(self.get_frontend_parameters)
        frontend.measureSignal.connect(self.measure)
        frontend.prepareButton.clicked.connect(self.prepare_ph)
        frontend.stopButton.clicked.connect(self.stop_measure)
        #frontend.shutterButton.clicked.connect(lambda: self.toggle_minflux_shutters(frontend.shutterButton.isChecked()))

        frontend.emit_param() # TO DO: change such that backend has parameters defined from the start

    def stop(self):
  
        self.tcspcTimer.stop()


if __name__ == '__main__':

    if not QtGui.QApplication.instance():
        app = QtGui.QApplication([])
    else:
        app = QtGui.QApplication.instance()
        
    #app.setStyle(QtGui.QStyleFactory.create('fusion'))
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    DEVICENUMBER = 0x1
    adw = ADwin.ADwin(DEVICENUMBER, 1)
    scan.setupDevice(adw)

    ph = picoharp.PicoHarp300()
    worker = Backend(ph, adw)
    gui = Frontend()
    
    workerThread = QtCore.QThread()
    workerThread.start()
    worker.moveToThread(workerThread)
    
    worker.tcspcTimer.moveToThread(workerThread)
    worker.tcspcTimer.timeout.connect(worker.update)

    worker.make_connection(gui)
    gui.make_connection(worker)

    gui.setWindowTitle('Time-correlated single-photon counting')
    gui.show()

    app.exec_()
    