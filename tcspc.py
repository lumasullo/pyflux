# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 14:14:14 2019

@author: USUARIO
"""

import numpy as np
import time
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import tools.tools as tools
import ctypes as ct
from PIL import Image
from tkinter import Tk, filedialog
import tifffile as tiff
import scipy.optimize as opt


from threading import Thread

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import Dock, DockArea

import drivers.ADwin as ADwin
from instrumental.drivers.cameras import uc480
import tools.viewbox_tools as viewbox_tools
import drivers.picoharp as picoharp
import PicoHarp.Read_PTU as Read_PTU
import tools.pyqtsubclass as pyqtsc
import tools.colormaps as cmaps

class tcspcWidget(QtGui.QFrame):
        
    def __init__(self, ph_device, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        # create ph object
        self.ph = ph_device
        
        # widget with tcspc parameters

        self.paramWidget = QtGui.QFrame()
        self.paramWidget.setFrameStyle(QtGui.QFrame.Panel |
                                       QtGui.QFrame.Raised)
        
        self.paramWidget.setFixedHeight(420)
        self.paramWidget.setFixedWidth(200)
        
        phParamTitle = QtGui.QLabel('<h2><strong>TCSPC settings</strong></h2>')
        phParamTitle.setTextFormat(QtCore.Qt.RichText)
        
        # widget to display data
        
        self.dataWidget = pg.GraphicsLayoutWidget()
        
        # Measure button

        self.measureButton = QtGui.QPushButton('Measure')
        self.measureButton.setCheckable(True)
        self.measureButton.clicked.connect(self.measure)
        
        # Read button
        
        self.readButton = QtGui.QPushButton('Read')
        self.readButton.setCheckable(True)
        self.readButton.clicked.connect(self.read)
        
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
        self.folderEdit = QtGui.QLineEdit(r'C:\Data')
        self.browseFolderButton = QtGui.QPushButton('Browse')
        self.browseFolderButton.setCheckable(True)
        self.browseFolderButton.clicked.connect(self.loadFolder)
        
        # initial directory

        self.initialDir = r'C:\Data'
        
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
        
    def preparePH(self):
        
        self.ph.open()
        self.ph.initialize()
        self.ph.setup()
        
        self.ph.syncDivider = 4 # this parameter must be set such that the count rate at channel 0 (sync) is equal or lower than 10MHz
        self.ph.resolution = int(self.resolutionEdit.text()) # desired resolution in ps
        self.ph.offset = 0
        self.ph.tacq = int(self.acqtimeEdit.text()) * 1000 # time in ms
        
        self.cts0 = self.ph.countrate(0)
        self.cts1 = self.ph.countrate(1)
        
        self.channel0Label.setText(('Input0 (sync) = {} c/s'.format(self.cts0)))
        self.channel1Label.setText(('Input0 (sync) = {} c/s'.format(self.cts1)))
  
        print('Resolution = {} ps'.format(self.ph.resolution))
        print('Acquisition time = {} s'.format(self.ph.tacq))
        
    def measure(self):
        
        self.preparePH()
        self.filename = os.path.join(self.folderEdit.text(),
                                     self.filenameEdit.text())
        self.name = tools.getUniqueName(self.filename)
        self.ph.startTTTR(self.name)
        
        self.measureButton.setChecked(False)
        
    def read(self):
        
        inputfile = open(self.name, "rb")
        
        numRecords = self.ph.numRecords # number of records
        globRes = 2.5e-8  # in ns, corresponds to sync @40 MHz
        timeRes = self.ph.resolution * 1e-12 # time resolution in s

        relTime, absTime = Read_PTU.readPT3(inputfile, numRecords)
        
        inputfile.close()
        
        relTime = relTime * timeRes # in real time units (s)
        relTime = relTime * 1e9  # in (ns)
        
        counts, bins = np.histogram(relTime, bins=100) # TO DO: choose proper binning
        self.histPlot.plot(bins[0:-1], counts)
        
        self.readButton.setChecked(False)

#        plt.hist(relTime, bins=300)
#        plt.xlabel('time (ns)')
#        plt.ylabel('ocurrences')

        absTime = absTime * globRes * 1e9  # true time in (ns), 4 comes from syncDivider, 10 Mhz = 40 MHz / syncDivider 
        absTime = absTime / 1e6 # in ms

#        plt.figure()
        timetrace, time = np.histogram(absTime, bins=50) # timetrace with 10 ms bins
        
        self.tracePlot.plot(time[0:-1], timetrace)

#        plt.plot(time[0:-1], timetrace)
#        plt.xlabel('time (ms)')
#        plt.ylabel('counts')
#        
#        inputfile.close()
        
    def loadFolder(self):

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
        
        
if __name__ == '__main__':
    
    ph = picoharp.PicoHarp300()

    app = QtGui.QApplication([])
    win = tcspcWidget(ph)
    win.setWindowTitle('Time-correlated single-photon counting')
    win.show()

    app.exec_()