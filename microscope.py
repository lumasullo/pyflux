# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 14:18:19 2018

@author: Cibion
"""

import numpy as np
import time
import os
import matplotlib.pyplot as plt
import tools
from PIL import Image
from tkinter import Tk, filedialog
import tifffile as tiff

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import Dock, DockArea

import ADwin
from instrumental.drivers.cameras import uc480
import viewbox_tools


class scanWidget(QtGui.QFrame):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

#        self.device = 'simulated'
#        self.device = 'nidaq'
        self.device = 'ADwin'
        self.edited_scan = True
        self.roi = None
        self.traceROI = None
        self.lineROI = None
        self.image = np.zeros((128, 128))
        self.resize(1800, 800)
        
        # setup device parameters

        self.setupDevice()

        # Create widget where the liveview image will be displayed

        imageWidget = pg.GraphicsLayoutWidget()
        self.vb = imageWidget.addViewBox(row=0, col=0)
        self.linePlot = imageWidget.addPlot(row=1, col=0)
        self.linePlot.setLabels(bottom=('nm'),
                                left=('counts'))


        # LiveView Button

        self.liveviewButton = QtGui.QPushButton('confocal LIVEVIEW')
        self.liveviewButton.setCheckable(True)
        self.liveviewButton.clicked.connect(self.liveview)

        # ROI buttons

        self.ROIButton = QtGui.QPushButton('ROI')
        self.ROIButton.setCheckable(True)
        self.ROIButton.clicked.connect(self.ROImethod)

        self.selectROIButton = QtGui.QPushButton('select ROI')
        self.selectROIButton.clicked.connect(self.selectROI)

        # Acquire frame button

        self.acquireFrameButton = QtGui.QPushButton('Acquire frame')
        self.acquireFrameButton.setCheckable(True)
        self.acquireFrameButton.clicked.connect(self.frameAcquisition)

        # preview scan button

        self.previewScanButton = QtGui.QPushButton('Scan preview')
        self.previewScanButton.setCheckable(True)
        self.previewScanButton.clicked.connect(self.previewScan)
        
        # line profile button
        
        self.lineProfButton = QtGui.QPushButton('Line profile')
        self.lineProfButton.setCheckable(True)
        self.lineProfButton.clicked.connect(self.lineProfile)

        # Scanning parameters

        self.initialPosLabel = QtGui.QLabel('Initial Pos'
                                            ' [x0, y0, z0] (µm)')
        self.initialPosEdit = QtGui.QLineEdit('3 3 10')
        self.scanRangeLabel = QtGui.QLabel('Scan range (µm)')
        self.scanRangeEdit = QtGui.QLineEdit('8')
        self.pxTimeLabel = QtGui.QLabel('Pixel time (µs)')
        self.pxTimeEdit = QtGui.QLineEdit('500')
        self.NofPixelsLabel = QtGui.QLabel('Number of pixels')
        self.NofPixelsEdit = QtGui.QLineEdit('128')
        self.pxSizeLabel = QtGui.QLabel('Pixel size (nm)')
        self.pxSizeValue = QtGui.QLabel('')
        self.frameTimeLabel = QtGui.QLabel('Frame time (s)')
        self.frameTimeValue = QtGui.QLabel('')

        self.auxAccelerationLabel = QtGui.QLabel('Aux acc'
                                                 ' (% of a_max)')
        self.auxAccelerationEdit = QtGui.QLineEdit('1 1 1 1')

        self.waitingTimeLabel = QtGui.QLabel('Scan waiting time (µs)')
        self.waitingTimeEdit = QtGui.QLineEdit('0')

        # Connections between changes in parameters and paramChanged function

        self.NofPixelsEdit.textChanged.connect(self.paramChanged)
        self.scanRangeEdit.textChanged.connect(self.paramChanged)
        self.pxTimeEdit.textChanged.connect(self.paramChanged)
        self.initialPosEdit.textChanged.connect(self.paramChanged)
        self.auxAccelerationEdit.textChanged.connect(self.paramChanged)
        self.waitingTimeEdit.textChanged.connect(self.paramChanged)

        # filename

        self.filenameLabel = QtGui.QLabel('File name')
        self.filenameEdit = QtGui.QLineEdit('filename')

        # folder

        self.folderLabel = QtGui.QLabel('Folder')
        self.folderEdit = QtGui.QLineEdit(r'C:\Data')
        self.browseFolderButton = QtGui.QPushButton('Browse')
        self.browseFolderButton.setCheckable(True)
        self.browseFolderButton.clicked.connect(self.loadFolder)

        # move to button

        self.moveToButton = QtGui.QPushButton('Move to')
        self.moveToLabel = QtGui.QLabel('Move to [x0, y0, z0] (µm)')
        self.moveToEdit = QtGui.QLineEdit('0 0 10')

        self.moveToButton.clicked.connect(self.moveToAction)
        
        # drift precision measurement
        
        self.driftPresMeasButton = QtGui.QPushButton('Drift precision measurement')
        self.driftPresMeasButton.clicked.connect(self.driftPrecisionMeas)

        # scan selection

        self.scanModeLabel = QtGui.QLabel('Scan type')

        self.scanMode = QtGui.QComboBox()
        self.scanModes = ['xy', 'xz', 'yz']
        self.scanMode.addItems(self.scanModes)
        
        self.detectorType = QtGui.QComboBox()
        self.dettypes = ['APD','photodiode']
        self.detectorType.addItems(self.dettypes)
        
        self.detectorType.activated.connect(self.paramChanged)

        self.scanMode.activated.connect(self.paramChanged)

        # widget with scanning parameters

        self.paramWidget = QtGui.QFrame()
        self.paramWidget.setFrameStyle(QtGui.QFrame.Panel |
                                       QtGui.QFrame.Raised)
        
        scanParamTitle = QtGui.QLabel('<h2><strong>Scan settings</strong></h2>')
        scanParamTitle.setTextFormat(QtCore.Qt.RichText)

        # GUI layout

        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(imageWidget, 0, 1)
        grid.addWidget(self.paramWidget, 0, 0)

        subgrid = QtGui.QGridLayout()
        self.paramWidget.setLayout(subgrid)

        subgrid.addWidget(self.liveviewButton, 18, 0)
        subgrid.addWidget(self.ROIButton, 19, 0)
        subgrid.addWidget(self.selectROIButton, 20, 0)
        subgrid.addWidget(self.acquireFrameButton, 21, 0)
        subgrid.addWidget(self.previewScanButton, 22, 0)
        subgrid.addWidget(self.lineProfButton, 23, 0)

        subgrid.addWidget(scanParamTitle, 0, 0, 2, 3)
        
        subgrid.addWidget(self.initialPosLabel, 2, 0)
        subgrid.addWidget(self.initialPosEdit, 3, 0)
        subgrid.addWidget(self.scanRangeLabel, 4, 0)
        subgrid.addWidget(self.scanRangeEdit, 5, 0)
        subgrid.addWidget(self.pxTimeLabel, 6, 0)
        subgrid.addWidget(self.pxTimeEdit, 7, 0)
        subgrid.addWidget(self.NofPixelsLabel, 8, 0)
        subgrid.addWidget(self.NofPixelsEdit, 9, 0)
        subgrid.addWidget(self.pxSizeLabel, 10, 0)
        subgrid.addWidget(self.pxSizeValue, 11, 0)
        subgrid.addWidget(self.frameTimeLabel, 12, 0)
        subgrid.addWidget(self.frameTimeValue, 13, 0)
        subgrid.addWidget(self.auxAccelerationLabel, 14, 0)
        subgrid.addWidget(self.auxAccelerationEdit, 15, 0)
        subgrid.addWidget(self.waitingTimeLabel, 16, 0)
        subgrid.addWidget(self.waitingTimeEdit, 17, 0)

        subgrid.addWidget(self.filenameLabel, 2, 2)
        subgrid.addWidget(self.filenameEdit, 3, 2)
        subgrid.addWidget(self.folderLabel, 4, 2)
        subgrid.addWidget(self.folderEdit, 5, 2)
        subgrid.addWidget(self.browseFolderButton, 6, 2)
        
        subgrid.addWidget(self.scanModeLabel, 2, 1)
        subgrid.addWidget(self.scanMode, 3, 1)
        subgrid.addWidget(self.detectorType, 4, 1)

        subgrid.addWidget(self.moveToLabel, 5, 1)
        subgrid.addWidget(self.moveToEdit, 6, 1)
        subgrid.addWidget(self.moveToButton, 7, 1)
        subgrid.addWidget(self.driftPresMeasButton, 8, 1)

        self.paramWidget.setFixedHeight(500)
        self.paramWidget.setFixedWidth(400)
        
        # Dock Area
        
        dockArea = DockArea()
        
        grid.addWidget(dockArea, 0, 2)
        
        # pulsed-MINFLUX measurement
        
        minfluxDock = Dock("MINFLUX measurement")
        minfluxWidget = QtGui.QWidget()
        minfluxDock.addWidget(minfluxWidget)
        dockArea.addDock(minfluxDock)
        
        # drift correction
        
        driftDock = Dock("Drift correction")

        self.driftWidget = driftWidget(self.adw)
        driftDock.addWidget(self.driftWidget)
        driftDock.addWidget(self.driftWidget)
        dockArea.addDock(driftDock)
        
        # trace measurement
        
        traceDock = Dock("Trace measurement")

        traceWidget = QtGui.QWidget()
        traceGrid = QtGui.QGridLayout()
        traceWidget.setLayout(traceGrid)
        self.traceGraph = pg.GraphicsLayoutWidget()
        self.tracePlot = self.traceGraph.addPlot(0, 0)
        
        self.linePlot.setLabels(bottom=('s'),
                                left=('counts'))
        
        self.traceROIButton = QtGui.QPushButton('Trace ROI')
        self.traceROIButton.setCheckable(True)
        self.traceROIButton.clicked.connect(self.traceROImethod)
        
        self.measureTracesButton = QtGui.QPushButton('Measure traces')
        self.measureTracesButton.setCheckable(True)
        self.measureTracesButton.clicked.connect(self.measureTraces)
        
        self.binTimeLabel = QtGui.QLabel('Time bin (µs)')
        self.binTimeLabel.setFixedWidth(65)
        self.binTimeEdit = QtGui.QLineEdit('100')
        self.binTimeEdit.setFixedWidth(45)
        
        self.totTimeLabel = QtGui.QLabel('Total time (s)')
        self.totTimeLabel.setFixedWidth(65)
        self.totTimeEdit = QtGui.QLineEdit('0.01')
        self.totTimeEdit.setFixedWidth(45)
        
        self.binTimeEdit.textChanged.connect(self.paramChanged)
        self.totTimeEdit.textChanged.connect(self.paramChanged)
        
        traceGrid.addWidget(self.traceGraph, 0, 0, 1, 5)
        traceGrid.addWidget(self.measureTracesButton, 1, 4, 1, 1)
        traceGrid.addWidget(self.traceROIButton, 1, 2, 1, 1)
        
        traceGrid.addWidget(self.binTimeLabel, 1, 0, 1, 1)
        traceGrid.addWidget(self.binTimeEdit, 1, 1, 1, 1)
        
        traceGrid.addWidget(self.totTimeLabel, 2, 0, 1, 1)
        traceGrid.addWidget(self.totTimeEdit, 2, 1, 1, 1)
        
        
        traceDock.addWidget(traceWidget)
        dockArea.addDock(traceDock, 'above', minfluxDock)
        dockArea.addDock(driftDock, 'above', traceDock)
        
        # Viewbox and image item where the liveview will be displayed

        self.vb.setMouseMode(pg.ViewBox.RectMode)
        self.img = pg.ImageItem()
        self.img.translate(-0.5, -0.5)
        self.vb.addItem(self.img)
        self.vb.setAspectLocked(True)
        imageWidget.setAspectLocked(True)

        # set up histogram for the liveview image

        self.hist = pg.HistogramLUTItem(image=self.img)
        self.hist.gradient.loadPreset('thermal')
        self.hist.vb.setLimits(yMin=0, yMax=66000)

        for tick in self.hist.gradient.ticks:
            tick.hide()
        imageWidget.addItem(self.hist, row=0, col=1)

        # Create a timer for the update of the liveview

        self.viewtimer = QtCore.QTimer()
        self.viewtimer.timeout.connect(self.updateView)

        # Counter for the saved images

        self.imageNumber = 0

        # initialize flag for the ADwin process

        self.flag = 0

        # update parameters

        self.paramChanged()
        
        # initialize fpar_50, fpar_51, fpar_52 ADwin position parameters
        
        pos_zero = tools.convert(0, 'XtoU')
        
        self.adw.Set_FPar(50, pos_zero)
        self.adw.Set_FPar(51, pos_zero)
        self.adw.Set_FPar(52, pos_zero)
        
        # move to z = 10 µm

        self.moveTo(0, 0, 10)

        # initial directory

        self.initialDir = r'C:\Data'

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

    def setupDevice(self):

        DEVICENUMBER = 0x1

        BTL = "ADwin11.btl"
        PROCESS_1 = "linescan_signal_data.TB1"
        PROCESS_2 = "moveto_xyz.TB2"
        PROCESS_6 = "trace_measurement.TB6"

        self.adw = ADwin.ADwin(DEVICENUMBER, 1)

        btl = self.adw.ADwindir + BTL

        self.currdir = os.getcwd()
        process_folder = os.path.join(self.currdir, "processes")

        process_1 = os.path.join(process_folder, PROCESS_1)
        process_2 = os.path.join(process_folder, PROCESS_2)
        process_6 = os.path.join(process_folder, PROCESS_6)

        self.adw.Boot(btl)
        self.adw.Load_Process(process_1)
        self.adw.Load_Process(process_2)
        self.adw.Load_Process(process_6)

    def previewScan(self):

        plt.figure('Preview scan plot x vs t')
        plt.plot(self.data_t_adwin[0:-1], self.data_x_adwin, 'go')
        plt.xlabel('t (ADwin time)')
        plt.ylabel('V (DAC units)')

        if np.max(self.data_x_adwin) > 2**16:

            plt.plot(self.data_t_adwin[0:-1],
                     2**16 * np.ones(np.size(self.data_t_adwin[0:-1])), 'r-')
    

    def paramChanged(self):

        # updates parameters according to what is input in the GUI
    
        self.detector = self.detectorType.currentText()
        self.scantype = self.scanMode.currentText()

        self.scanRange = float(self.scanRangeEdit.text())
        self.NofPixels = int(self.NofPixelsEdit.text())
        self.pxTime = float(self.pxTimeEdit.text())
        self.a_aux_coeff = np.array(self.auxAccelerationEdit.text().split(' '),
                                    dtype=np.float32)/100
        self.initialPos = np.array(self.initialPosEdit.text().split(' '),
                                   dtype=np.float16)
        self.pxSize = self.scanRange/self.NofPixels   # in µm
        self.frameTime = self.NofPixels**2 * self.pxTime / 10**6

        self.waitingtime = float(self.waitingTimeEdit.text())  # in µs

        self.frameTimeValue.setText('{}'.format(np.around(self.frameTime, 2)))
        self.pxSizeValue.setText('{}'.format(1000 * np.around(self.pxSize,
                                                              5)))  # in nm

        self.linetime = (1/1000)*self.pxTime*self.NofPixels  # in ms

        #  aux scan parameters

        self.a_max = 4 * 10**-6  # in µm/µs^2

        if np.all(self.a_aux_coeff) <= 1:
            self.a_aux = self.a_aux_coeff * self.a_max
        else:
            self.a_aux[self.a_aux > 1] = self.a_max

        self.NofAuxPixels = 100

        self.waiting_pixels = int(self.waitingtime/self.pxTime)
        self.tot_pixels = (2 * self.NofPixels + 4 * self.NofAuxPixels +
                           self.waiting_pixels)

        # create scan signal

        self.dy = self.pxSize
        self.dz = self.pxSize

        (self.data_t, self.data_x,
         self.data_y) = tools.ScanSignal(self.scanRange,
                                         self.NofPixels,
                                         self.NofAuxPixels,
                                         self.pxTime,
                                         self.a_aux,
                                         self.dy,
                                         self.initialPos[0],
                                         self.initialPos[1],
                                         self.initialPos[2],
                                         self.scantype,
                                         self.waitingtime)

#        self.viewtimer_time = (1/1000) * self.data_t[-1]    # in ms
        
        # TO DO: entender bien esto del timer = 0, leer documentación
        
        self.viewtimer_time = 0  # largar el timer lo más rápido posible

        # Create blank image
        # edited_scan = True --> size of the useful part of the scan
        # edited_scan = False --> size of the full scan including aux parts

        if self.edited_scan is True:

#            size = (2 * self.NofPixels, self.NofPixels)
            size = (self.NofPixels, self.NofPixels)

        else:

            size = (self.tot_pixels, self.NofPixels)

        self.blankImage = np.zeros(size)
        self.image = self.blankImage
        self.i = 0
        
        # parameters for the time trace measurements
        
        self.binTime = float(self.binTimeEdit.text())
        self.totTime = float(self.totTimeEdit.text()) * 10**6 # in µs
        
        self.tot_bins = np.int(self.totTime/self.binTime)

        # load the new parameters into the ADwin system

        self.updateDeviceParameters()

    def updateDeviceParameters(self):
        
        if self.detector == 'APD':
            self.adw.Set_Par(30, 0)  # Digital input (APD)

        if self.detector == 'photodiode':
            self.adw.Set_Par(30, 1)  # Analog input (photodiode)

        # select scan type

        if self.scantype == 'xy':

            self.adw.Set_FPar(10, 1)
            self.adw.Set_FPar(11, 2)

        if self.scantype == 'xz':

            self.adw.Set_FPar(10, 1)
            self.adw.Set_FPar(11, 6)

        if self.scantype == 'yz':

            self.adw.Set_FPar(10, 2)
            self.adw.Set_FPar(11, 6)

        #  initial positions x and y

        self.x_i = self.initialPos[0]
        self.y_i = self.initialPos[1]
        self.z_i = self.initialPos[2]

        self.x_offset = 0
        self.y_offset = 0
        self.z_offset = 0

        #  load ADwin parameters

        self.adw.Set_Par(1, self.tot_pixels)

        self.data_t_adwin = tools.timeToADwin(self.data_t)
        self.data_x_adwin = tools.convert(self.data_x, 'XtoU')
        self.data_y_adwin = tools.convert(self.data_y, 'XtoU')

        # repeat last element because time array has to have one more
        # element than position array

        dt = self.data_t_adwin[-1] - self.data_t_adwin[-2]

        self.data_t_adwin = np.append(self.data_t_adwin,
                                      (self.data_t_adwin[-1] + dt))

        # prepare arrays for conversion into ADwin-readable data

        self.time_range = np.size(self.data_t_adwin)
        self.space_range = np.size(self.data_x_adwin)

        self.data_t_adwin = np.array(self.data_t_adwin, dtype='int')
        self.data_x_adwin = np.array(self.data_x_adwin, dtype='int')
        self.data_y_adwin = np.array(self.data_y_adwin, dtype='int')

        self.data_t_adwin = list(self.data_t_adwin)
        self.data_x_adwin = list(self.data_x_adwin)
        self.data_y_adwin = list(self.data_y_adwin)

        self.adw.SetData_Long(self.data_t_adwin, 2, 1, self.time_range)
        self.adw.SetData_Long(self.data_x_adwin, 3, 1, self.space_range)
        self.adw.SetData_Long(self.data_y_adwin, 4, 1, self.space_range)
        
        # prepare and load time trace parameters
        
        self.binTime_adwin = tools.timeToADwin(self.binTime)
        self.totTime_adwin = tools.timeToADwin(self.totTime)
        
        self.trace_data = list(np.array(np.zeros(self.tot_bins), 
                                                 dtype='int'))
        
        self.adw.SetData_Long(self.trace_data, 6, 1, self.tot_bins)

        self.adw.Set_FPar(60, self.tot_bins)
        self.adw.Set_FPar(65, self.binTime_adwin)
        self.adw.Set_FPar(67, self.totTime_adwin)

    def moveToParameters(self, x_f, y_f, z_f, n_pixels_x=128, n_pixels_y=128,
                         n_pixels_z=128, pixeltime=2000):

        x_f = tools.convert(x_f, 'XtoU')
        y_f = tools.convert(y_f, 'XtoU')
        z_f = tools.convert(z_f, 'XtoU')
        
        print(x_f, y_f, z_f)

        self.adw.Set_Par(21, n_pixels_x)
        self.adw.Set_Par(22, n_pixels_y)
        self.adw.Set_Par(23, n_pixels_z)

        self.adw.Set_FPar(23, x_f)
        self.adw.Set_FPar(24, y_f)
        self.adw.Set_FPar(25, z_f)

        self.adw.Set_FPar(26, tools.timeToADwin(pixeltime))

    def moveTo(self, x_f, y_f, z_f):

        self.moveToParameters(x_f, y_f, z_f)
        self.adw.Start_Process(2)

    def moveToAction(self):

        final_position = np.array(self.moveToEdit.text().split(' '),
                                  dtype=np.float16)

        self.moveTo(final_position[0], final_position[1], final_position[2])
        
    def getPosition(self):
        
        xPos = self.adw.Get_FPar(50)
        yPos = self.adw.Get_FPar(51)
        zPos = self.adw.Get_FPar(52)
        
        self.xPos = tools.convert(xPos, 'UtoX')
        self.yPos = tools.convert(yPos, 'UtoX')
        self.zPos = tools.convert(zPos, 'UtoX') 
            
    def frameAcquisition(self):

        if self.acquireFrameButton.isChecked():
            self.liveviewStop()
            self.liveviewButton.setChecked(False)
            self.frameAcquisitionStart()

        else:
            self.frameAcquisitionStop()

    def frameAcquisitionStart(self):

        self.acquisitionMode = 'frame'

        # save scan plot (x vs t)

        self.filename = os.path.join(self.folderEdit.text(),
                                     self.filenameEdit.text())

        plt.figure('Scan plot x vs t')
        plt.plot(self.data_t_adwin[0:-1], self.data_x_adwin, 'go')
        plt.xlabel('t (ADwin time)')
        plt.ylabel('V (DAC units)')

        fname = tools.getUniqueName(self.filename)
        fname = fname + '_scan_plot'
        plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None)
        
        # restar the slow axis
        
        self.y_offset = 0

        # move to initial position smoothly

        if self.scantype == 'xy':

            self.moveTo(self.x_i, self.y_i, self.z_i)

        if self.scantype == 'xz':

            self.moveTo(self.x_i, self.y_i + self.scanRange/2,
                        self.z_i - self.scanRange/2)

        if self.scantype == 'yz':

            self.moveTo(self.x_i + self.scanRange/2, self.y_i,
                        self.z_i - self.scanRange/2)

        self.i = 0

        # start updateView timer

        self.viewtimer.start(self.viewtimer_time)

    def frameAcquisitionStop(self):

        self.viewtimer.stop()

        # experiment parameters

        name = tools.getUniqueName(self.filename)
        now = time.strftime("%c")
        tools.saveConfig(self, now, name)

        # save image

        data = self.image
        result = Image.fromarray(data.astype('uint16'))

        result.save(r'{}.tif'.format(name))

        self.imageNumber += 1
        self.acquireFrameButton.setChecked(False)
    
    def lineAcquisition(self):

        # TO DO: fix problem of waiting time

        self.adw.Start_Process(1)

#        # flag changes when process is finished

        if (((1/1000) * self.data_t[-1]) < 240):   # in ms

            while self.flag == 0:
                self.flag = self.adw.Get_Par(2)
        
        else:
            
            line_time = (1/1000) * self.data_t[-1]  # in ms
            wait_time = line_time * 1.05

            time.sleep(wait_time/1000)

#        print(self.viewtimer_time)
#        time0 = time.time()
        line_data = self.adw.GetData_Long(1, 0, self.tot_pixels)
#        time1 = time.time()
#        print(time1 - time0)
#        time.sleep(0.2)

        line_data[0] = 0  # TO DO: fix the high count error on first element

        return line_data

    # This is the function triggered by pressing the liveview button
    def liveview(self):

        if self.liveviewButton.isChecked():
            self.liveviewStart()

        else:
            self.liveviewStop()

    def liveviewStart(self):

        self.acquisitionMode = 'liveview'

        if self.scantype == 'xy':

            self.moveTo(self.x_i, self.y_i, self.z_i)

        if self.scantype == 'xz':

            self.moveTo(self.x_i, self.y_i + self.scanRange/2,
                        self.z_i - self.scanRange/2)

        if self.scantype == 'yz':

            self.moveTo(self.x_i + self.scanRange/2, self.y_i,
                        self.z_i - self.scanRange/2)

        if self.roi is not None:

            self.vb.removeItem(self.roi)
            self.roi.hide()

            self.ROIButton.setChecked(False)
            
        if self.lineROI is not None:

            self.vb.removeItem(self.lineROI)

            self.lineProfButton.setChecked(False)

        else:

            pass

        self.viewtimer.start(self.viewtimer_time)

    def liveviewStop(self):

        self.viewtimer.stop()

    def updateView(self):

        if self.scantype == 'xy':

            dy = tools.convert(self.dy, 'ΔXtoU')
            self.y_offset = int(self.y_offset + dy)
            self.adw.Set_FPar(2, self.y_offset)

        if self.scantype == 'xz' or self.scantype == 'yz':

            dz = tools.convert(self.dz, 'ΔXtoU')
            self.z_offset = int(self.z_offset + dz)
            self.adw.Set_FPar(2, self.z_offset)

        self.lineData = self.lineAcquisition()

        if self.edited_scan is True:

            c0 = self.NofAuxPixels
            c1 = self.NofAuxPixels + self.NofPixels

            self.lineData_edited = self.lineData[c0:c1]
            self.image[:, self.NofPixels-1-self.i] = self.lineData_edited

        else:

            self.image[:, self.NofPixels-1-self.i] = self.lineData

        self.img.setImage(self.image, autoLevels=False)

        if self.i < self.NofPixels-1:

            self.i = self.i + 1

        else:

            print('Frame ended')

            if self.acquisitionMode == 'frame':
                self.frameAcquisitionStop()

            self.i = 0
            self.y_offset = 0
            self.z_offset = 0

            if self.scantype == 'xy':

                self.moveTo(self.x_i, self.y_i, self.z_i)

            if self.scantype == 'xz':

                self.moveTo(self.x_i, self.y_i + self.scanRange/2,
                            self.z_i - self.scanRange/2)

            if self.scantype == 'yz':

                self.moveTo(self.x_i + self.scanRange/2, self.y_i,
                            self.z_i - self.scanRange/2)

            self.updateDeviceParameters()
            
    def lineProfile(self):
        
        if self.lineROI is None:
            
            self.lineROI = pg.LineSegmentROI([[10, 64], [120,64]], pen='b')
            self.vb.addItem(self.lineROI)
            
        else:

            self.vb.removeItem(self.lineROI)
            
            self.lineROI = pg.LineSegmentROI([[10, 64], [120,64]], pen='b')
            self.vb.addItem(self.lineROI)
            
        self.lineROI.sigRegionChanged.connect(self.updateLineProfile)
        
    def measureTraces(self):
        
        roi = self.traceROI.getArrayRegion(self.image, self.img)
        
        self.adw.Start_Process(6)
        time.sleep((self.totTime * 1.5) / 10**6) # in s
        self.trace_data = self.adw.GetData_Long(6, 1, self.tot_bins)
        self.trace_data = np.array(self.trace_data)

        self.updateTraceGraph()
        
    def updateTraceGraph(self):
        
        t = np.arange(0, self.tot_bins) * self.binTime
        self.tracePlot.clear()
        self.tracePlot.plot(t, self.trace_data)
        
    def updateLineProfile(self):
        
#        data = self.line.getArrayRegion(self.image, self.vb.imageItem, axes=(1,2))
        data = self.lineROI.getArrayRegion(self.image, self.img)
        self.linePlot.clear()
        x = self.pxSize * np.arange(np.size(data))*1000
        self.linePlot.plot(x, data)
        
    def traceROImethod(self):
        
#        if self.traceRoi_list is None:
#            self.traceRoi_list = []

        trace_roi_size = 6
        traceROIpen = pg.mkPen(color='r')
        
        if self.traceROI is None:

            
            ROIpos = (0.5 * self.NofPixels - 64, 0.5 * self.NofPixels - 64)
            self.traceROI = viewbox_tools.ROI(trace_roi_size, self.vb, ROIpos,
                                              handlePos=(1, 0),
                                              handleCenter=(0, 1),
                                              scaleSnap=True,
                                              translateSnap=True,
                                              pen=traceROIpen)
        else:
            
            self.vb.removeItem(self.traceROI)
            self.traceROI.hide()

            ROIpos = (0.5 * self.NofPixels - 64, 0.5 * self.NofPixels - 64)
            self.traceROI = viewbox_tools.ROI(trace_roi_size, self.vb, ROIpos,
                                         handlePos=(1, 0),
                                         handleCenter=(0, 1),
                                         scaleSnap=True,
                                         translateSnap=True,
                                         pen=traceROIpen)
            
#        self.traceRoi_list.append(self.roi)
        
    def ROImethod(self):
        
        ROIpen = pg.mkPen(color='y')

        if self.roi is None:

            ROIpos = (0.5 * self.NofPixels - 64, 0.5 * self.NofPixels - 64)
            self.roi = viewbox_tools.ROI(self.NofPixels/2, self.vb, ROIpos,
                                         handlePos=(1, 0),
                                         handleCenter=(0, 1),
                                         scaleSnap=True,
                                         translateSnap=True,
                                         pen=ROIpen)

        else:

            self.vb.removeItem(self.roi)
            self.roi.hide()

            ROIpos = (0.5 * self.NofPixels - 64, 0.5 * self.NofPixels - 64)
            self.roi = viewbox_tools.ROI(self.NofPixels/2, self.vb, ROIpos,
                                         handlePos=(1, 0),
                                         handleCenter=(0, 1),
                                         scaleSnap=True,
                                         translateSnap=True,
                                         pen=ROIpen)
            
    def selectROI(self):

        self.liveviewStop()

        array = self.roi.getArrayRegion(self.image, self.img)
        ROIpos = np.array(self.roi.pos())

        newPos_px = tools.ROIscanRelativePOS(ROIpos,
                                             self.NofPixels,
                                             np.shape(array)[1])

        newPos_µm = newPos_px * self.pxSize + self.initialPos[0:2]

        newPos_µm = np.around(newPos_µm, 2)

        self.initialPosEdit.setText('{} {} {}'.format(newPos_µm[0],
                                                      newPos_µm[1],
                                                      self.initialPos[2]))

        newRange_px = np.shape(array)[0]
        newRange_µm = self.pxSize * newRange_px
        newRange_µm = np.around(newRange_µm, 2)
        self.scanRangeEdit.setText('{}'.format(newRange_µm))
        
    def driftPrecisionMeas(self):
        
        n = 50 # number of images saved
        dx = 0.010 # step in µm
        exptime = '50 ms' # exposure time in ms
        
        width = self.driftWidget.uc480.width
        height = self.driftWidget.uc480.height
        
        self.driftPrecisionData = np.zeros((n, height, width), dtype=np.int16)
        self.targetPos = self.initialPos
        
        for i in range(0, n):
            
            self.targetPos = self.targetPos + np.array([dx, 0, 0])
            print(self.targetPos)
            self.moveTo(*self.targetPos)
            rawimage = self.driftWidget.uc480.grab_image(exposure_time=exptime)
            image = np.sum(rawimage, axis=2)
            self.driftPrecisionData[i, :, :] = image
            
            time.sleep(0.050) # wait 50 ms just in case
            
        
            
        np.save(self.currdir + '\driftdata.npy', self.driftPrecisionData)
        tiff.imsave(self.currdir + '\driftdata.tif', self.driftPrecisionData)
            
        # TO DO: change position in display to actual new position for safety
        # TO DO: display actual position somwhere in the GUI
        # TO DO: save driftPrecisionData
            
    def closeEvent(self, *args, **kwargs):

        # Stop running threads

        self.viewtimer.stop()

        # Go back to 0 position

        x_0 = 0
        y_0 = 0
        z_0 = 0

        self.moveTo(x_0, y_0, z_0)

        super().closeEvent(*args, **kwargs)
        
class driftWidget(QtGui.QFrame):
    
    def __init__(self, ADwin, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        self.uc480 = uc480.UC480_Camera()
        
        imageWidget = pg.GraphicsLayoutWidget()
        self.vb = imageWidget.addViewBox(row=0, col=0)
        self.vb.setMouseMode(pg.ViewBox.RectMode)
        self.img = pg.ImageItem()
        self.img.translate(-0.5, -0.5)
        self.vb.addItem(self.img)
        self.vb.setAspectLocked(True)
        imageWidget.setAspectLocked(True)
        
        self.hist = pg.HistogramLUTItem(image=self.img)
        self.hist.gradient.loadPreset('grey')
        self.hist.vb.setLimits(yMin=0, yMax=66000)

        for tick in self.hist.gradient.ticks:
            tick.hide()
        imageWidget.addItem(self.hist, row=0, col=1)
        
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(imageWidget, 0, 0)
        
        self.paramWidget = QtGui.QFrame()
        self.paramWidget.setFrameStyle(QtGui.QFrame.Panel |
                                       QtGui.QFrame.Raised)
        
        grid.addWidget(self.paramWidget, 0, 1)
        
        # IR LiveView Button

        self.liveviewButton = QtGui.QPushButton('IR camera LIVEVIEW')
        self.liveviewButton.setCheckable(True)
        self.liveviewButton.clicked.connect(self.liveview)
        
        self.viewtimer = QtCore.QTimer()
        self.viewtimer.timeout.connect(self.updateView)
        
        subgrid = QtGui.QGridLayout()
        self.paramWidget.setLayout(subgrid)

        subgrid.addWidget(self.liveviewButton, 0, 1)
        
#        self.andorLuca = ccd.CCD()
#        cam = 1
#        self.andorLuca.current_camera = self.andorLuca.camera_handle(cam)
#        self.andorLuca.lib.Initialize()
#        self.shape = self.andorLuca.detector_shape
#        self.andorLuca.set_image(shape=self.shape)
#        self.andorLuca.set_exposure_time(0.100)
#
#        print(self.andorLuca.idn)
        
    def liveview(self):

        if self.liveviewButton.isChecked():
            self.liveviewStart()

        else:
            self.liveviewStop()

        
    def liveviewStart(self):
        
#        idle = 'Camera is idle, waiting for instructions.'
#        if self.andorLuca.status != idle:
#            self.andorLuca.abort_acquisition()
#
#        self.andorLuca.acquisition_mode = 'Run till abort'
##            self.andor.shutter(0, 1, 0, 0, 0)
#
#        self.andorLuca.start_acquisition()
#        time.sleep(0.5)
#
#        # Initial image
        self.rawimage = self.uc480.grab_image(exposure_time='20 ms')
        self.image = np.sum(self.rawimage, axis=2) # sum r, g, b images
#
        self.img.setImage(np.transpose(self.image), autoLevels=False)
        self.hist.setHistogramRange(np.min(self.image), np.max(self.image))
#
##        self.vb.scene().sigMouseMoved.connect(self.mouseMoved)
#
        self.viewtimer.start(30)

        pass
    
    def liveviewStop(self):
        
        self.viewtimer.stop()
            
    def updateView(self):
        """ Image update while in Liveview mode
        """
        self.rawimage = self.uc480.grab_image(exposure_time='20 ms')
        self.image = np.sum(self.rawimage, axis=2) # sum r, g, b images

        self.img.setImage(np.transpose(self.image), autoLevels=False)
#            self.hist.setHistogramRange(np.min(self.image), np.max(self.image))

            
class minfluxWidget(QtGui.QFrame):
        
    def __init__(self, ADwin, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.PH = picoharp.PicoHarp300()
        


if __name__ == '__main__':

    app = QtGui.QApplication([])
    win = scanWidget()
    win.show()

    app.exec_()
