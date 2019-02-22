# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 14:14:14 2019

@author: USUARIO
"""

import numpy as np
import time
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
import tools.pyqtsubclass as pyqtsc
import tools.colormaps as cmaps

import qdarkstyle

class Frontend(QtGui.QFrame):
    
    paramSignal = pyqtSignal(list)
    measureSignal = pyqtSignal()
        
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
                
        # initial directory

        self.initialDir = r'C:\Data'
        self.setup_gui()
        
    def start_measurement(self):
        
        filename = os.path.join(self.folderEdit.text(),
                                self.filenameEdit.text())
        
        name = tools.getUniqueName(filename)
        res = int(self.resolutionEdit.text())
        tacq = int(self.acqtimeEdit.text())
        folder = self.folderEdit.text()
        
        paramlist = [name, res, tacq, folder]

        self.paramSignal.emit(paramlist)
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
        
    @pyqtSlot(float, float)
    def get_backend_parameters(self, cts0, cts1):
        
        self.channel0Label.setText(('Input0 (sync) = {} c/s'.format(cts0)))
        self.channel1Label.setText(('Input0 (sync) = {} c/s'.format(cts1)))
    
    @pyqtSlot(np.ndarray, np.ndarray)    
    def plot_data(self, relTime, absTime):
        
        counts, bins = np.histogram(relTime, bins=100) # TO DO: choose proper binning
        self.histPlot.plot(bins[0:-1], counts)

#        plt.hist(relTime, bins=300)
#        plt.xlabel('time (ns)')
#        plt.ylabel('ocurrences')

        timetrace, time = np.histogram(absTime, bins=50) # timetrace with 10 ms bins

        self.tracePlot.plot(time[0:-1], timetrace)

#        plt.plot(time[0:-1], timetrace)
#        plt.xlabel('time (ms)')
#        plt.ylabel('counts')
        
        self.readButton.setChecked(False)
        
    def make_connection(self, backend):
        
        backend.ctRatesSignal.connect(self.get_backend_parameters)
        backend.readDataSignal.connect(self.plot_data)
        
    def  setup_gui(self):
        
        # widget with tcspc parameters

        self.paramWidget = QtGui.QFrame()
        self.paramWidget.setFrameStyle(QtGui.QFrame.Panel |
                                       QtGui.QFrame.Raised)
        
        self.paramWidget.setFixedHeight(420)
        self.paramWidget.setFixedWidth(200)
        
#        phParamTitle = QtGui.QLabel('<h2><strong>TCSPC settings</strong></h2>')
        phParamTitle = QtGui.QLabel('<h2>TCSPC settings</h2>')
        phParamTitle.setTextFormat(QtCore.Qt.RichText)
        
        # widget to display data
        
        self.dataWidget = pg.GraphicsLayoutWidget()
        
        # Measure button

        self.measureButton = QtGui.QPushButton('Measure')
        self.measureButton.setCheckable(True)
        self.measureButton.clicked.connect(self.start_measurement)
        
        # Read button
        
        self.readButton = QtGui.QPushButton('Read')
        self.readButton.setCheckable(True)
#        self.readButton.clicked.connect(self.read)
        
        # TCSPC parameters

        self.acqtimeLabel = QtGui.QLabel('Acquisition time (s)')
        self.acqtimeEdit = QtGui.QLineEdit('1')
        self.resolutionLabel = QtGui.QLabel('Resolution (ps)')
        self.resolutionEdit = QtGui.QLineEdit('16')
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

        self.folderLabel = QtGui.QLabel('Folder')
        self.folderEdit = QtGui.QLineEdit(self.initialDir)
        self.browseFolderButton = QtGui.QPushButton('Browse')
        self.browseFolderButton.setCheckable(True)
        self.browseFolderButton.clicked.connect(self.load_folder)
        
        # GUI layout

        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.paramWidget, 0, 0)
        grid.addWidget(self.dataWidget, 0, 1)
        
        subgrid = QtGui.QGridLayout()
        self.paramWidget.setLayout(subgrid)
        subgrid.addWidget(phParamTitle, 0, 0, 2, 3)

        subgrid.addWidget(self.acqtimeLabel, 2, 0)
        subgrid.addWidget(self.acqtimeEdit, 3, 0)
        subgrid.addWidget(self.resolutionLabel, 4, 0)
        subgrid.addWidget(self.resolutionEdit, 5, 0)
        subgrid.addWidget(self.offsetLabel, 6, 0)
        subgrid.addWidget(self.offsetEdit, 7, 0)
        subgrid.addWidget(self.channel0Label, 8, 0)
        subgrid.addWidget(self.channel1Label, 9, 0)
        
        subgrid.addWidget(self.filenameLabel, 11, 0)
        subgrid.addWidget(self.filenameEdit, 12, 0)
        subgrid.addWidget(self.folderLabel, 13, 0)
        subgrid.addWidget(self.folderEdit, 14, 0)
        subgrid.addWidget(self.browseFolderButton, 15, 0)
        
        subgrid.addWidget(self.measureButton, 17, 0)
        subgrid.addWidget(self.readButton, 19, 0)
        
class Backend(QtCore.QObject):

    ctRatesSignal = pyqtSignal(float, float)
    readDataSignal = pyqtSignal(np.ndarray, np.ndarray)
    
    def __init__(self, ph_device, *args, **kwargs): 
        
        super().__init__(*args, **kwargs)
          
        self.ph = ph_device 
        
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
        
#        self.channel0Label.setText(('Input0 (sync) = {} c/s'.format(self.cts0)))
#        self.channel1Label.setText(('Input0 (sync) = {} c/s'.format(self.cts1)))
        
        self.ctRatesSignal.emit(self.cts0, self.cts1)
  
        print('Resolution = {} ps'.format(self.ph.resolution))
        print('Acquisition time = {} s'.format(self.ph.tacq))
     
    @pyqtSlot()           
    def measure(self):
         
        self.prepare_ph()
        print('Measurement started')
        self.ph.startTTTR(self.fname)

    def read_data(self):
        
        inputfile = open(self.fname, "rb")
        
        numRecords = self.ph.numRecords # number of records
        globRes = 2.5e-8  # in ns, corresponds to sync @40 MHz
        timeRes = self.ph.resolution * 1e-12 # time resolution in s

        relTime, absTime = Read_PTU.readPT3(inputfile, numRecords)
        
        inputfile.close()
        
        relTime = relTime * timeRes # in real time units (s)
        relTime = relTime * 1e9  # in (ns)

        absTime = absTime * globRes * 1e9  # true time in (ns), 4 comes from syncDivider, 10 Mhz = 40 MHz / syncDivider 
        absTime = absTime / 1e6 # in ms
        
        self.readDataSignal.emit(relTime, absTime)

    @pyqtSlot(list)
    def get_frontend_parameters(self, paramlist):

        self.fname = paramlist[0]
        self.resolution = paramlist[1]
        self.tacq = paramlist[2]
        self.folder = paramlist[3]


    def make_connection(self, frontend):

        frontend.paramSignal.connect(self.get_frontend_parameters)
        frontend.measureSignal.connect(self.measure)
        frontend.readButton.clicked.connect(self.read_data)


    def stop(self):
  
        pass


if __name__ == '__main__':

    app = QtGui.QApplication([])
    app.setStyle(QtGui.QStyleFactory.create('fusion'))
#    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    ph = picoharp.PicoHarp300()
    worker = Backend(ph)
    gui = Frontend()
    
    workerThread = QtCore.QThread()
    workerThread.start()
    worker.moveToThread(workerThread)

    worker.make_connection(gui)
    gui.make_connection(worker)

    gui.setWindowTitle('Time-correlated single-photon counting')
    gui.show()

    app.exec_()
    