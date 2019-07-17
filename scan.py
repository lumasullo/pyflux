# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 14:18:19 2018

@author: Luciano A. Masullo
"""

import numpy as np
import time
from datetime import date, datetime
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
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5 import QtTest
import qdarkstyle

import tools.PSF as PSF

import drivers.ADwin as ADwin
import tools.viewbox_tools as viewbox_tools
import tools.colormaps as cmaps
from tools.lineprofile import linePlotWidget

π = np.pi


def setupDevice(adw):

    BTL = "ADwin11.btl"
    PROCESS_1 = "line_scan.TB1"
    PROCESS_2 = "moveto_xyz.TB2"
    PROCESS_3 = "actuator_z.TB3"
    PROCESS_4 = "actuator_xy.TB4"
    PROCESS_5 = "shutter.TB5"
    PROCESS_6 = "trace.TB6"
    
    btl = adw.ADwindir + BTL
    adw.Boot(btl)

    currdir = os.getcwd()
    process_folder = os.path.join(currdir, "processes")

    process_1 = os.path.join(process_folder, PROCESS_1)
    process_2 = os.path.join(process_folder, PROCESS_2)
    process_3 = os.path.join(process_folder, PROCESS_3)
    process_4 = os.path.join(process_folder, PROCESS_4)
    process_5 = os.path.join(process_folder, PROCESS_5)
    process_6 = os.path.join(process_folder, PROCESS_6)
    
    adw.Load_Process(process_1)
    adw.Load_Process(process_2)
    adw.Load_Process(process_3)
    adw.Load_Process(process_4)
    adw.Load_Process(process_5)
    adw.Load_Process(process_6)
    
    
class Frontend(QtGui.QFrame):
    
    paramSignal = pyqtSignal(dict)
    closeSignal = pyqtSignal()
    liveviewSignal = pyqtSignal(bool, str)
    frameacqSignal = pyqtSignal(bool)
    fitPSFSignal = pyqtSignal(np.ndarray)

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

#        self.device = 'simulated'
#        self.device = 'nidaq'
#        self.device = 'ADwin'

        self.roi = None
        self.lineROI = None
        self.EBPscatter = [None, None, None, None]
        self.EBPcenters = np.zeros((4, 2))
        self.advanced = False
        self.EBPshown = True
        self.fitting = False
        self.image = np.zeros((128, 128))
        
        self.initialDir = r'C:\Data'
        
        # set up GUI

        self.setup_gui()
                
        # connections between changes in parameters and emit_param function
        
        self.NofPixelsEdit.textChanged.connect(self.emit_param)
        self.scanRangeEdit.textChanged.connect(self.emit_param)
        self.pxTimeEdit.textChanged.connect(self.emit_param)
        self.initialPosEdit.textChanged.connect(self.emit_param)
        self.auxAccEdit.textChanged.connect(self.emit_param)
        self.waitingTimeEdit.textChanged.connect(self.emit_param)
        self.detectorType.activated.connect(self.emit_param)
        self.scanMode.activated.connect(self.emit_param)
        self.filenameEdit.textChanged.connect(self.emit_param)
        self.xStepEdit.textChanged.connect(self.emit_param)
        self.yStepEdit.textChanged.connect(self.emit_param)
        self.zStepEdit.textChanged.connect(self.emit_param)
        self.moveToEdit.textChanged.connect(self.emit_param)
        self.powerEdit.textChanged.connect(self.emit_param)
        
    def emit_param(self):
        
        params = dict()
        
        params['detectorType'] = self.detectorType.currentText()
        params['scanType'] = self.scanMode.currentText()
        params['scanRange'] = float(self.scanRangeEdit.text())
        params['NofPixels'] = int(self.NofPixelsEdit.text())
        params['pxTime'] = float(self.pxTimeEdit.text())
        params['initialPos'] = np.array(self.initialPosEdit.text().split(' '),
                                        dtype=np.float64)
        params['a_aux_coeff'] = np.array(self.auxAccEdit.text().split(' '),
                                              dtype=np.float32)/100
        
        params['waitingTime'] = int(self.waitingTimeEdit.text())  # in µs
        params['fileName'] = os.path.join(self.folderEdit.text(),
                                          self.filenameEdit.text())
        params['moveToPos'] = np.array(self.moveToEdit.text().split(' '),
                                       dtype=np.float16)
        
        params['xStep'] = float(self.xStepEdit.text())
        params['yStep'] = float(self.yStepEdit.text())
        params['zStep'] = float(self.zStepEdit.text())
        params['power'] = float(self.powerEdit.text())
        
        self.paramSignal.emit(params)
        
    @pyqtSlot(dict)
    def get_backend_param(self, params):
        
        frameTime = params['frameTime']
        pxSize = params['pxSize']
        maxCounts = params['maxCounts']
#        initialPos = np.round(params['initialPos'], 2)
        
#        print(datetime.now(), '[scan-frontend] got initialPos', initialPos)
        
        self.frameTimeValue.setText('{}'.format(np.around(frameTime, 2)))
        self.pxSizeValue.setText('{}'.format(np.around(1000 * pxSize, 3))) # in nm
        self.maxCountsValue.setText('{}'.format(maxCounts)) 
#        self.initialPosEdit.setText('{} {} {}'.format(*initialPos))
        
        self.pxSize = pxSize
     
    @pyqtSlot(np.ndarray)
    def get_image(self, image):
        
#        self.img.setImage(image, autoLevels=False)
        self.image = image.T[:,::-1]
        self.img.setImage(self.image, autoLevels=False)
        
    @pyqtSlot(np.ndarray)
    def get_real_position(self, val): 
      
      val = np.around(val, 3)
      self.moveToEdit.setText('{} {} {}'.format(*val))
        
    def main_roi(self):
        
        # TO DO: move this function to backend and implement "typedfeat" variables
        
        self.scanRangeEdit.setText('8')
        self.initialPosEdit.setText('{} {} {}'.format(*[3, 3, 10]))
    
    def toggle_advanced(self):
        
        if self.advanced:
            
            self.auxAccelerationLabel.show()
            self.auxAccEdit.show()
            self.waitingTimeLabel.show()
            self.waitingTimeEdit.show() 
            self.preview_scanButton.show()
            
            self.advanced = False
            
        else:
            
            self.auxAccelerationLabel.hide()
            self.auxAccEdit.hide()
            self.waitingTimeLabel.hide()
            self.waitingTimeEdit.hide() 
            self.preview_scanButton.hide()
            
            self.advanced = True

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
    
    def preview_scan(self):

        plt.figure('Preview scan plot x vs t')
        plt.plot(self.data_t_adwin[0:-1], self.data_x_adwin, 'go')
        plt.xlabel('t (ADwin time)')
        plt.ylabel('V (DAC units)')

        if np.max(self.data_x_adwin) > 2**16:

            plt.plot(self.data_t_adwin[0:-1],
                     2**16 * np.ones(np.size(self.data_t_adwin[0:-1])), 'r-')

    def toggle_liveview(self):

        if self.liveviewButton.isChecked():
            self.liveviewSignal.emit(True, 'liveview')
            
            if self.roi is not None:

                self.vb.removeItem(self.roi)
                self.roi.hide()
    
                self.ROIButton.setChecked(False)
            
            if self.lineROI is not None:

                self.vb.removeItem(self.lineROI)
                self.lplotWidget.hide()
                self.lineProfButton.setChecked(False)
                self.lineROI = None

            else:
    
                pass

        else:
            self.liveviewSignal.emit(False, 'liveview')
            self.emit_param()
            
    def toggle_frame_acq(self):

        if self.acquireFrameButton.isChecked():
            self.frameacqSignal.emit(True)
            
            if self.roi is not None:

                self.vb.removeItem(self.roi)
                self.roi.hide()
    
                self.ROIButton.setChecked(False)
                self.liveviewButton.setChecked(False)
            
            if self.lineROI is not None:

                self.vb.removeItem(self.lineROI)
                self.lplotWidget.hide()
                self.lineProfButton.setChecked(False)
                self.lineROI = None

            else:
    
                pass

        else:
            self.frameacqSignal.emit(False)   

    def line_profile(self):
        
        if self.lineROI is None:
            
            self.lineROI = pg.LineSegmentROI([[10, 64], [120,64]], pen='r')
            self.vb.addItem(self.lineROI)
            
            self.lplotWidget.show()
            
        else:

            self.vb.removeItem(self.lineROI)
            
            self.lineROI = pg.LineSegmentROI([[10, 64], [120,64]], pen='r')
            self.vb.addItem(self.lineROI)
            
        self.lineROI.sigRegionChanged.connect(self.update_line_profile)
        
    def update_line_profile(self):
        
        data = self.lineROI.getArrayRegion(self.image, self.img)
        self.lplotWidget.linePlot.clear()
        x = self.pxSize * np.arange(np.size(data))*1000
        self.lplotWidget.linePlot.plot(x, data)
        
    def toggle_ROI(self):
        
        ROIpen = pg.mkPen(color='y')
        npixels = int(self.NofPixelsEdit.text())

        if self.roi is None:

            ROIpos = (0.5 * npixels - 64, 0.5 * npixels - 64)
            self.roi = viewbox_tools.ROI(npixels/2, self.vb, ROIpos,
                                         handlePos=(1, 0),
                                         handleCenter=(0, 1),
                                         scaleSnap=True,
                                         translateSnap=True,
                                         pen=ROIpen)

        else:

            self.vb.removeItem(self.roi)
            self.roi.hide()

            ROIpos = (0.5 * npixels - 64, 0.5 * npixels - 64)
            self.roi = viewbox_tools.ROI(npixels/2, self.vb, ROIpos,
                                         handlePos=(1, 0),
                                         handleCenter=(0, 1),
                                         scaleSnap=True,
                                         translateSnap=True,
                                         pen=ROIpen)
            
        if self.EBProiButton.isChecked:
            self.EBProiButton.setChecked(False)
            
    def select_ROI(self):

        self.liveviewSignal.emit(False, 'liveview')
        
        ROIsize = np.array(self.roi.size())
        ROIpos = np.array(self.roi.pos())
        
        npixels = int(self.NofPixelsEdit.text())
        pxSize = self.pxSize
        initialPos = np.array(self.initialPosEdit.text().split(' '),
                              dtype=np.float64)

        newPos_px = tools.ROIscanRelativePOS(ROIpos, npixels, ROIsize[1])

        newPos_µm = newPos_px * pxSize + initialPos[0:2]

        newPos_µm = np.around(newPos_µm, 2)

        self.initialPosEdit.setText('{} {} {}'.format(newPos_µm[0],
                                                      newPos_µm[1],
                                                      initialPos[2]))

        newRange_px = ROIsize[1]
        newRange_µm = pxSize * newRange_px
        newRange_µm = np.around(newRange_µm, 2)
        self.scanRangeEdit.setText('{}'.format(newRange_µm))
        
        self.emit_param()
        
    def emit_fit_ROI(self):
      
      if self.roi is not None:
        
            xmin, ymin = self.roi.pos()
            xmax, ymax = self.roi.pos() + self.roi.size()
            
            ymin, ymax = [int(self.NofPixelsEdit.text()) - ymax, 
                          int(self.NofPixelsEdit.text()) - ymin]
            
            coordinates = np.array([xmin, xmax, ymin, ymax])  
          
          
            self.fitPSFSignal.emit(coordinates)
            
      else:
            
            print('[scan] no ROI for the fit was selected')
        
    def set_EBP(self):
        
        pxSize = self.pxSize
        ROIsize = np.array(self.roi.size())
        
        for i in range(4):
        
            if self.EBPscatter[i] is not None:
                
                self.vb.removeItem(self.EBPscatter[i])
        
#        array = self.roi.getArrayRegion(self.scworker.image, self.img)
        ROIsize = np.array(self.roi.size())
        ROIpos_µm = np.array(self.roi.pos()) * pxSize
            
        xmin = ROIpos_µm[0]
        xmax = ROIpos_µm[0] + ROIsize[0] * pxSize
        
        ymin = ROIpos_µm[1]
        ymax = ROIpos_µm[1] + ROIsize[1] * pxSize
        
        x0 = (xmax+xmin)/2
        y0 = (ymax+ymin)/2
        
        if self.EBPtype.currentText() == 'triangle':
        
            L = int(self.LEdit.text())/1000 # in µm
            θ = π * np.array([1/6, 5/6, 3/2])
            ebp = (L/2)*np.array([[0, 0], [np.cos(θ[0]), np.sin(θ[0])], 
                                 [np.cos(θ[1]), np.sin(θ[1])], 
                                 [np.cos(θ[2]), np.sin(θ[2])]])
            
            self.EBP = (ebp + np.array([x0, y0]))/pxSize
                                       
        print('[scan] EBP px', self.EBP)
            
        for i in range(4):
        
            if i == 0:
                mybrush = pg.mkBrush(255, 255, 0, 255)
                
            if i == 1:
                mybrush = pg.mkBrush(255, 127, 80, 255)
                
            if i == 2:
                mybrush = pg.mkBrush(135, 206, 235, 255)
                
            if i == 3:
                mybrush = pg.mkBrush(0, 0 ,0 , 255)
                
            self.EBPscatter[i] = pg.ScatterPlotItem([self.EBP[i][0]], 
                                                    [self.EBP[i][1]], 
                                                    size=10,
                                                    pen=pg.mkPen(None), 
                                                    brush=mybrush)            

            self.vb.addItem(self.EBPscatter[i])
        
        self.set_EBPButton.setChecked(False)
        
    def toggle_EBP(self):
        
        if self.EBPshown:
        
            for i in range(len(self.EBPscatter)):
                
                if self.EBPscatter[i] is not None:
                    self.vb.removeItem(self.EBPscatter[i])
                else:
                    pass
            
            self.EBPshown = False
            
        else:
            
            for i in range(len(self.EBPscatter)):
                
                if self.EBPscatter[i] is not None:
                    self.vb.addItem(self.EBPscatter[i])
                else:
                    pass
            
            self.EBPshown = True
    
        self.showEBPButton.setChecked(False)
        
    def setup_gui(self):
                
        # image widget set-up and layout

        imageWidget = pg.GraphicsLayoutWidget()
        self.vb = imageWidget.addViewBox(row=0, col=0)
        self.lplotWidget = linePlotWidget()
        
        imageWidget.setFixedHeight(500)
        imageWidget.setFixedWidth(500)
        
        # Viewbox and image item where the liveview will be displayed

        self.vb.setMouseMode(pg.ViewBox.RectMode)
        self.img = pg.ImageItem()
        self.img.translate(-0.5, -0.5)
        self.vb.addItem(self.img)
        self.vb.setAspectLocked(True)
        imageWidget.setAspectLocked(True)

        # set up histogram for the liveview image

        self.hist = pg.HistogramLUTItem(image=self.img)
        lut = viewbox_tools.generatePgColormap(cmaps.parula)
        self.hist.gradient.setColorMap(lut)
        self.hist.vb.setLimits(yMin=0, yMax=10000)

        for tick in self.hist.gradient.ticks:
            tick.hide()
        imageWidget.addItem(self.hist, row=0, col=1)
        
        # widget with scanning parameters

        self.paramWidget = QtGui.QFrame()
        self.paramWidget.setFrameStyle(QtGui.QFrame.Panel |
                                       QtGui.QFrame.Raised)
        
        scanParamTitle = QtGui.QLabel('<h2><strong>Scan settings</strong></h2>')
        scanParamTitle.setTextFormat(QtCore.Qt.RichText)

        # LiveView Button

        self.liveviewButton = QtGui.QPushButton('confocal liveview')
        self.liveviewButton.setFont(QtGui.QFont('Helvetica', weight=QtGui.QFont.Bold))
        self.liveviewButton.setCheckable(True)
        self.liveviewButton.setStyleSheet("font-size: 12px; background-color:rgb(180, 180, 180)")
        self.liveviewButton.clicked.connect(self.toggle_liveview)
        
        # ROI buttons

        self.ROIButton = QtGui.QPushButton('ROI')
        self.ROIButton.setCheckable(True)
        self.ROIButton.clicked.connect(self.toggle_ROI)

        self.select_ROIButton = QtGui.QPushButton('select ROI')
        self.select_ROIButton.clicked.connect(self.select_ROI)
      
        # Shutter button
        
        self.shutterButton = QtGui.QPushButton('Shutter open/close')
        self.shutterButton.setCheckable(True)
        
        # Flipper button
        
        self.flipperButton = QtGui.QPushButton('Flipper 100x up/down')
        self.flipperButton.setCheckable(True)
        
        # Save current frame button

        self.currentFrameButton = QtGui.QPushButton('Save current frame')

        # preview scan button

        self.preview_scanButton = QtGui.QPushButton('Scan preview')
        self.preview_scanButton.setCheckable(True)
        self.preview_scanButton.clicked.connect(self.preview_scan)
        
        # move to center button
        
        self.moveToROIcenterButton = QtGui.QPushButton('Move to ROI center') 
#        self.moveToROIcenterButton.clicked.connect(self.select_ROI)
        
        # dougnhut fit
        
        self.psfFitButton = QtGui.QPushButton('PSF fit and move')
        self.psfFitButton.clicked.connect(self.emit_fit_ROI)
        
        # main ROI button
        
        self.mainROIButton = QtGui.QPushButton('Go to main ROI') 
        self.mainROIButton.clicked.connect(self.main_roi)

        # line profile button
        
        self.lineProfButton = QtGui.QPushButton('Line profile')
        self.lineProfButton.setCheckable(True)
        self.lineProfButton.clicked.connect(self.line_profile)
        
        # edited scan button
        
        self.FBavScanButton = QtGui.QPushButton('F and B average scan')
        self.FBavScanButton.setCheckable(True)

        # Scanning parameters

        self.initialPosLabel = QtGui.QLabel('Initial Pos'
                                            ' [x0, y0, z0] (µm)')
        self.initialPosEdit = QtGui.QLineEdit('3 3 10')
        self.scanRangeLabel = QtGui.QLabel('Scan range (µm)')
        self.scanRangeEdit = QtGui.QLineEdit('8')
        self.pxTimeLabel = QtGui.QLabel('Pixel time (µs)')
        self.pxTimeEdit = QtGui.QLineEdit('500')
        self.NofPixelsLabel = QtGui.QLabel('Number of pixels')
        self.NofPixelsEdit = QtGui.QLineEdit('80')
        
        self.pxSizeLabel = QtGui.QLabel('Pixel size (nm)')
        self.pxSizeValue = QtGui.QLineEdit('')
        self.pxSizeValue.setReadOnly(True)
        self.frameTimeLabel = QtGui.QLabel('Frame time (s)')
        self.frameTimeValue = QtGui.QLineEdit('')
        self.frameTimeValue.setReadOnly(True)
        self.maxCountsLabel = QtGui.QLabel('Max counts per pixel')
        self.maxCountsValue = QtGui.QLineEdit('')
        self.frameTimeValue.setReadOnly(True)
        
        self.powerLabel = QtGui.QLabel('Power at BFP (µW)')
        self.powerEdit = QtGui.QLineEdit('0')
        
        self.advancedButton = QtGui.QPushButton('Advanced options')
        self.advancedButton.setCheckable(True)
        self.advancedButton.clicked.connect(self.toggle_advanced)
        
        self.auxAccelerationLabel = QtGui.QLabel('Aux acc'
                                                 ' (% of a_max)')
        self.auxAccEdit = QtGui.QLineEdit('1 1 1 1')
        self.waitingTimeLabel = QtGui.QLabel('Scan waiting time (µs)')
        self.waitingTimeEdit = QtGui.QLineEdit('0')
        
        self.toggle_advanced()

        # file/folder widget
        
        self.fileWidget = QtGui.QFrame()
        self.fileWidget.setFrameStyle(QtGui.QFrame.Panel |
                                      QtGui.QFrame.Raised)
        
        self.fileWidget.setFixedHeight(120)
        self.fileWidget.setFixedWidth(230)

        # folder and buttons
        
        today = str(date.today()).replace('-', '')
        root = r'C:\\Data\\'
        folder = root + today
        
        try:  
            os.mkdir(folder)
        except OSError:  
            print(datetime.now(), '[scan] Directory {} already exists'.format(folder))
        else:  
            print(datetime.now(), '[scan] Successfully created the directory {}'.format(folder))
        
        self.filenameLabel = QtGui.QLabel('File name')
        self.filenameEdit = QtGui.QLineEdit('filename')
        self.folderLabel = QtGui.QLabel('Folder')
        self.folderEdit = QtGui.QLineEdit(folder)
        self.browseFolderButton = QtGui.QPushButton('Browse')
        self.browseFolderButton.setCheckable(True)
        self.browseFolderButton.clicked.connect(self.load_folder)

        # scan selection

        self.scanModeLabel = QtGui.QLabel('Scan type')

        self.scanMode = QtGui.QComboBox()
        self.scanModes = ['xy', 'xz', 'yz']
        self.scanMode.addItems(self.scanModes)
        
        self.detectorType = QtGui.QComboBox()
        self.dettypes = ['APD','photodiode']
        self.detectorType.addItems(self.dettypes)
        
        # widget with EBP parameters and buttons
        
        self.EBPWidget = QtGui.QFrame()
        self.EBPWidget.setFrameStyle(QtGui.QFrame.Panel |
                                       QtGui.QFrame.Raised)
        
        EBPparamTitle = QtGui.QLabel('<h2><strong>Excitation Beam Pattern</strong></h2>')
        EBPparamTitle.setTextFormat(QtCore.Qt.RichText)
        
        self.EBProiButton = QtGui.QPushButton('EBP ROI')
        self.EBProiButton.setCheckable(True)
        self.EBProiButton.clicked.connect(self.toggle_ROI)
        
        self.showEBPButton = QtGui.QPushButton('show/hide EBP')
        self.showEBPButton.setCheckable(True)
        self.showEBPButton.clicked.connect(self.toggle_EBP)

        self.set_EBPButton = QtGui.QPushButton('set EBP')
        self.set_EBPButton.clicked.connect(self.set_EBP)
        
        # EBP selection

        self.EBPtypeLabel = QtGui.QLabel('EBP type')

        self.EBPtype = QtGui.QComboBox()
        self.EBPoptions = ['triangle', 'square']
        self.EBPtype.addItems(self.EBPoptions)
        
        self.Llabel = QtGui.QLabel('L (nm)')
        self.LEdit = QtGui.QLineEdit('100')
        
        # piezo navigation widget
        
        self.positioner = QtGui.QFrame()
        self.positioner.setFrameStyle(QtGui.QFrame.Panel |
                                      QtGui.QFrame.Raised)
        
        self.xUpButton = QtGui.QPushButton("(+x) ►")  # →
        self.xDownButton = QtGui.QPushButton("◄ (-x)")  # ←

        self.yUpButton = QtGui.QPushButton("(+y) ▲")  # ↑
        self.yDownButton = QtGui.QPushButton("(-y) ▼")  # ↓
        
        self.zUpButton = QtGui.QPushButton("(+z) ▲")  # ↑
        self.zDownButton = QtGui.QPushButton("(-z) ▼")  # ↓
        
        self.xStepLabel = QtGui.QLabel('x step (µm)')
        self.xStepEdit = QtGui.QLineEdit('0.050')
        
        self.yStepLabel = QtGui.QLabel('y step (µm)')
        self.yStepEdit = QtGui.QLineEdit('0.050')
        
        self.zStepLabel = QtGui.QLabel('z step (µm)')
        self.zStepEdit = QtGui.QLineEdit('0.050')
        
        # move to button

        self.moveToButton = QtGui.QPushButton('Move to')
        self.moveToLabel = QtGui.QLabel('Move to [x0, y0, z0] (µm)')
        self.moveToEdit = QtGui.QLineEdit('0 0 10')
        
        # scan GUI layout

        grid = QtGui.QGridLayout()
        self.setLayout(grid)

        dockArea = DockArea() 
        grid.addWidget(dockArea, 0, 0)
        
        EBPDock = Dock('EBP')
        EBPDock.setOrientation(o="vertical", force=True)
        EBPDock.updateStyle()
        EBPDock.addWidget(self.EBPWidget)
        dockArea.addDock(EBPDock)
        
        positionerDock = Dock('Positioner')
        positionerDock.setOrientation(o="vertical", force=True)
        positionerDock.updateStyle()
        positionerDock.addWidget(self.positioner)
        dockArea.addDock(positionerDock, 'above', EBPDock)
        
        paramDock = Dock('Scan parameters')
        paramDock.setOrientation(o="vertical", force=True)
        paramDock.updateStyle()
        paramDock.addWidget(self.paramWidget)
        paramDock.addWidget(self.fileWidget)
        dockArea.addDock(paramDock, 'above', positionerDock)
        
        imageDock = Dock('Confocal view')
        imageDock.addWidget(imageWidget)
        dockArea.addDock(imageDock, 'right', paramDock)
        
        # parameters widget layout

        subgrid = QtGui.QGridLayout()
        self.paramWidget.setLayout(subgrid)
        
        subgrid.addWidget(scanParamTitle, 0, 0, 1, 3)
        
        subgrid.addWidget(self.scanModeLabel, 2, 2)
        subgrid.addWidget(self.scanMode, 3, 2)
        subgrid.addWidget(self.detectorType, 4, 2)
        subgrid.addWidget(self.liveviewButton, 5, 2)
        
        subgrid.addWidget(self.shutterButton, 7, 2)
        subgrid.addWidget(self.flipperButton, 8, 2)
        subgrid.addWidget(self.currentFrameButton, 9, 2)
        subgrid.addWidget(self.ROIButton, 10, 2)
        subgrid.addWidget(self.select_ROIButton, 11, 2)

        subgrid.addWidget(self.moveToROIcenterButton, 13, 2)
        subgrid.addWidget(self.mainROIButton, 14, 2)
        subgrid.addWidget(self.lineProfButton, 15, 2)
        subgrid.addWidget(self.FBavScanButton, 16, 2)
        subgrid.addWidget(self.psfFitButton, 17, 2)
        
        subgrid.addWidget(self.initialPosLabel, 2, 0, 1, 2)
        subgrid.addWidget(self.initialPosEdit, 3, 0, 1, 2)
        subgrid.addWidget(self.scanRangeLabel, 4, 0)
        subgrid.addWidget(self.scanRangeEdit, 4, 1)
        subgrid.addWidget(self.pxTimeLabel, 5, 0)
        subgrid.addWidget(self.pxTimeEdit, 5, 1)
        subgrid.addWidget(self.NofPixelsLabel, 6, 0)
        subgrid.addWidget(self.NofPixelsEdit, 6, 1)
        
        subgrid.addWidget(self.pxSizeLabel, 7, 0)
        subgrid.addWidget(self.pxSizeValue, 7, 1)
        subgrid.addWidget(self.frameTimeLabel, 8, 0)
        subgrid.addWidget(self.frameTimeValue, 8, 1)
        subgrid.addWidget(self.maxCountsLabel, 9, 0)
        subgrid.addWidget(self.maxCountsValue, 9, 1)
        subgrid.addWidget(self.powerLabel, 10, 0)
        subgrid.addWidget(self.powerEdit, 10, 1)
        
        subgrid.addWidget(self.advancedButton, 11, 0)
        
        subgrid.addWidget(self.auxAccelerationLabel, 12, 0)
        subgrid.addWidget(self.auxAccEdit, 13, 0)
        subgrid.addWidget(self.waitingTimeLabel, 14, 0)
        subgrid.addWidget(self.waitingTimeEdit, 15, 0)
        subgrid.addWidget(self.preview_scanButton, 16, 0)
        
        self.paramWidget.setFixedHeight(370)
        self.paramWidget.setFixedWidth(300)
        
#        subgrid.setColumnMinimumWidth(1, 130)
#        subgrid.setColumnMinimumWidth(1, 50)
        
        # file/folder widget layout
        
        file_subgrid = QtGui.QGridLayout()
        self.fileWidget.setLayout(file_subgrid)
        
        file_subgrid.addWidget(self.filenameLabel, 0, 0, 1, 2)
        file_subgrid.addWidget(self.filenameEdit, 1, 0, 1, 2)
        file_subgrid.addWidget(self.folderLabel, 2, 0, 1, 2)
        file_subgrid.addWidget(self.folderEdit, 3, 0, 1, 2)
        file_subgrid.addWidget(self.browseFolderButton, 4, 0)
        
        # EBP widget layout
        
        subgridEBP = QtGui.QGridLayout()
        self.EBPWidget.setLayout(subgridEBP)
        
        subgridEBP.addWidget(EBPparamTitle, 0, 0, 2, 4)
        
        subgridEBP.addWidget(self.EBProiButton, 2, 0, 1, 1)
        subgridEBP.addWidget(self.set_EBPButton, 3, 0, 1, 1)
        subgridEBP.addWidget(self.showEBPButton, 4, 0, 2, 1)
        subgridEBP.addWidget(self.EBPtypeLabel, 2, 1)
        subgridEBP.addWidget(self.EBPtype, 3, 1)
        subgridEBP.addWidget(self.Llabel, 4, 1)
        subgridEBP.addWidget(self.LEdit, 5, 1)
        
        self.EBPWidget.setFixedHeight(150)
        self.EBPWidget.setFixedWidth(250)
        
        # Piezo navigation widget layout

        layout = QtGui.QGridLayout()
        self.positioner.setLayout(layout)
        
        positionerTitle = QtGui.QLabel('<h2><strong>Position</strong></h2>')
        positionerTitle.setTextFormat(QtCore.Qt.RichText)
        
        layout.addWidget(positionerTitle, 0, 0, 2, 3)
        layout.addWidget(self.xUpButton, 2, 4, 2, 1)
        layout.addWidget(self.xDownButton, 2, 2, 2, 1)
        
        layout.addWidget(self.xStepLabel, 0, 6)        
        layout.addWidget(self.xStepEdit, 1, 6)
        
        layout.addWidget(self.yUpButton, 1, 3, 2, 1)
        layout.addWidget(self.yDownButton, 3, 3, 2, 1)
        
        layout.addWidget(self.yStepLabel, 2, 6)        
        layout.addWidget(self.yStepEdit, 3, 6)

        layout.addWidget(self.zUpButton, 1, 5, 2, 1)
        layout.addWidget(self.zDownButton, 3, 5, 2, 1)
        
        layout.addWidget(self.zStepLabel, 4, 6)        
        layout.addWidget(self.zStepEdit, 5, 6)
        
        layout.addWidget(self.moveToLabel, 6, 1, 1, 3)
        layout.addWidget(self.moveToEdit, 7, 1, 1, 2)
        layout.addWidget(self.moveToButton, 8, 1, 1, 2)
        
        self.positioner.setFixedHeight(250)
        self.positioner.setFixedWidth(400)
        
    # make connections between GUI and worker functions
            
    def make_connection(self, backend):
        
        backend.paramSignal.connect(self.get_backend_param)
        backend.imageSignal.connect(self.get_image)
        backend.realPositionSignal.connect(self.get_real_position)
        
    def closeEvent(self, *args, **kwargs):

        # Emit close signal

        self.closeSignal.emit()
        
        
      
class Backend(QtCore.QObject):
    
    paramSignal = pyqtSignal(dict)
    imageSignal = pyqtSignal(np.ndarray)
    frameIsDone = pyqtSignal(bool, np.ndarray) 
    ROIcenterSignal = pyqtSignal(np.ndarray)
    realPositionSignal = pyqtSignal(np.ndarray)
    auxFitSignal = pyqtSignal()
    
    """
    Signals
    
    - paramSignal:
         To: [frontend]
         Description: 
             
    - imageSignal:
         To: [frontend]
         Description: 
        
    - frameIsDone:
         To: [psf]
         Description: 
        
    - ROIcenterSignal:
         To: [minflux]
         Description:
        
    """
    
    def __init__(self, adwin, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.adw = adwin
        self.saveScanData = False
        self.feedback_active = False

        # full_scan: True --> full scan including aux parts
        # full_scan: False --> forward part of the scan
        
        self.full_scan = False
        self.FBaverage_scan = False
        
        # 5MHz is max count rate of the P. Elmer APD
        
        self.APDmaxCounts = 5*10**6   

        # Create a timer for the update of the liveview

        self.viewtimer = QtCore.QTimer()

        # Counter for the saved images

        self.imageNumber = 0

        # initialize flag for the linescan function

        self.flag = 0
        
        # initialize fpar_50, fpar_51, fpar_52 ADwin position parameters
        
        pos_zero = tools.convert(0, 'XtoU')
        
        self.adw.Set_FPar(70, pos_zero)
        self.adw.Set_FPar(71, pos_zero)
        self.adw.Set_FPar(72, pos_zero)
        
        # move to z = 10 µm

        self.moveTo(3, 3, 10)

        # initial directory

        self.initialDir = r'C:\Data'
        
        # initialize image
        
        self.image = None
        
    @pyqtSlot(dict)
    def get_frontend_param(self, params):
        
        # updates parameters according to what is input in the GUI
        
        self.detector = params['detectorType']
        self.scantype = params['scanType']
        self.scanRange = params['scanRange']
        self.NofPixels = int(params['NofPixels'])
        self.pxTime = params['pxTime']
        self.initialPos = params['initialPos']
        self.powerBFP = params['power']
        
        self.waitingTime = params['waitingTime']
        self.a_aux_coeff = params['a_aux_coeff']
        
        self.filename = params['fileName']
        
        self.moveToPos = params['moveToPos']
        
        self.xStep = params['xStep']
        self.yStep = params['yStep']
        self.zStep = params['zStep']
        
#        self.selectedCoord = params['ROIcoordinates']
#        
#        print('[scan] selected ROI coordinates are:', self.selectedCoord)
                
        self.calculate_derived_param()
    
    @pyqtSlot(np.ndarray)
    def get_ROI_coords_and_fit(self, array):
      
        self.selectedCoord = array
        print('[scan] selected fit ROI coordinates are:', self.selectedCoord)
        
        self.auxFitSignal.emit()
      
    def calculate_derived_param(self):
        
        self.image_to_save = self.image

        self.pxSize = self.scanRange/self.NofPixels   # in µm
        self.frameTime = self.NofPixels**2 * self.pxTime / 10**6
        self.maxCounts = int(self.APDmaxCounts/(1/(self.pxTime*10**-6)))
        self.linetime = (1/1000)*self.pxTime*self.NofPixels  # in ms
        
        #  aux scan parameters

        self.a_max = 4 * 10**-6  # in µm/µs^2

        if np.all(self.a_aux_coeff) <= 1:
            self.a_aux = self.a_aux_coeff * self.a_max
        else:
            self.a_aux[self.a_aux > 1] = self.a_max

        self.NofAuxPixels = 100

        self.waiting_pixels = int(self.waitingTime/self.pxTime)
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
                                         self.waitingTime)

#        self.viewtimer_time = (1/1000) * self.data_t[-1]    # in ms
        
        self.viewtimer_time = 0  # timer will timeout as soon after it has executed all functions

        # Create blank image
        # full_scan = True --> size of the full scan including aux parts 
        # full_scan = False --> size of the forward part of the scan

        if self.full_scan is True:

            size = (self.tot_pixels, self.tot_pixels)
            
        else:
            
            size = (self.NofPixels, self.NofPixels)

#        self.blankImage = np.zeros(size)
        
        self.image = np.zeros(size)
        self.imageF = np.zeros(size)
        self.imageB = np.zeros(size)
        
        self.i = 0

        # load the new parameters into the ADwin system
        
        self.update_device_param()

        # emit calculated parameters

        self.emit_param()
        
    def emit_param(self):
        
        params = dict()
        
        params['frameTime'] = self.frameTime
        params['pxSize'] = self.pxSize
        params['maxCounts'] = self.maxCounts
#        params['initialPos'] = np.float64(self.initialPos)
        
        self.paramSignal.emit(params)
        
    def update_device_param(self):
        
        if self.detector == 'APD':
            self.adw.Set_Par(3, 0)  # Digital input (APD)

        if self.detector == 'photodiode':
            self.adw.Set_Par(3, 1)  # Analog input (photodiode)

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

    def set_moveTo_param(self, x_f, y_f, z_f, n_pixels_x=128, n_pixels_y=128,
                         n_pixels_z=128, pixeltime=2000):

        x_f = tools.convert(x_f, 'XtoU')
        y_f = tools.convert(y_f, 'XtoU')
        z_f = tools.convert(z_f, 'XtoU')

#        print(x_f, y_f, z_f)

        self.adw.Set_Par(21, n_pixels_x)
        self.adw.Set_Par(22, n_pixels_y)
        self.adw.Set_Par(23, n_pixels_z)

        self.adw.Set_FPar(23, x_f)
        self.adw.Set_FPar(24, y_f)
        self.adw.Set_FPar(25, z_f)

        self.adw.Set_FPar(26, tools.timeToADwin(pixeltime))

    def moveTo(self, x_f, y_f, z_f):

        self.set_moveTo_param(x_f, y_f, z_f)
        self.adw.Start_Process(2)

    def moveTo_action(self):

        self.moveTo(*self.moveToPos)
        
    def moveTo_roi_center(self):
        
#        self.ROIcenter = self.initialPos + np.array([self.scanRange/2, self.scanRange/2, 0])
        
        xi, xf, yi, yf = self.selectedCoord
        self.ROIcenter = self.initialPos + np.array([(xf+xi)/2, (yf+yi)/2, 0]) * self.pxSize
        
#        print('[scan] self.initialPos[0:2]', self.initialPos[0:2])
        print('[scan] moved to center of ROI:', self.ROIcenter, 'µm')
        
        self.moveTo(*self.ROIcenter)
#        self.ROIcenterSignal.emit(self.ROIcenter)
        
#        # keep track of the new position to where you've moved
#        
#        self.initialPos = self.ROIcenter
#        self.emit_param()
        
    def psf_fit_FandB(self):
        
        target_F = self.psf_fit(self.imageF_copy, d='F')
        target_B = self.psf_fit(self.imageB_copy, d='B')
        
        print('[scan] target_F', target_F)
        print('[scan] target_B', target_B)
        
        target_position = 0.5*target_F + 0.5*target_B
        
        print('[scan] target_position', target_position)
        
        self.moveTo(*target_position)
#        self.ROIcenterSignal.emit(target_position)
        
        self.realPositionSignal.emit(target_position)
        
        time.sleep(.2)
        
        self.trace_measurement()
        
    def psf_fit(self, image, d, function='gaussian'):
        
        # set main reference frame (relative to the confocal image)
        
        px_size_nm = self.pxSize * 1000 # in nm
        
        xmin, xmax, ymin, ymax = np.array(self.selectedCoord, dtype=np.int)
        
        if d == 'B':
            
            xmin = xmin - 8
            xmax = xmax - 8
        
        elif d == 'F':
            
            pass
        
        else:
            
            print('[scan] Invalid direction of scan selected')
        
        # select the data of the image corresponding to the ROI

#        array = self.image_copy[xmin:xmax, ymin:ymax]
        array = image[ymin:ymax, xmin:xmax]
        
        print('shape of array', array.shape)
        
        if array.shape[0] > array.shape[1]:
            
            xmax  = xmax + 1
            array = image[ymin:ymax, xmin:xmax]
            
        elif array.shape[1] > array.shape[0]:
            
            ymax = ymax + 1
            array = image[ymin:ymax, xmin:xmax]
            
        else:
            
            pass
        
        shape = array.shape
        
        print('[scan] shape of array', array.shape)
            
        # create x and y arrays and meshgrid
        
        xmin_nm, xmax_nm, ymin_nm, ymax_nm = np.array([xmin, xmax, ymin, ymax]) * px_size_nm
        
        print('[scan] xmin_nm, xmax_nm, ymin_nm, ymax_nm', xmin_nm, xmax_nm, ymin_nm, ymax_nm)
        
        extent = [xmin_nm + self.initialPos[0] * 1000, self.initialPos[0] * 1000 + xmax_nm,
                  self.initialPos[1] * 1000 + ymax_nm, self.initialPos[1] * 1000 + ymin_nm]
        
        plt.figure()
        plt.imshow(array, cmap=cmaps.parula, interpolation='None', extent=extent)
        plt.xlabel('x (nm)')
        plt.ylabel('y (nm)')
             
#        x_nm = np.arange(xmin_nm + px_size_nm/2, xmax_nm + px_size_nm/2, px_size_nm) # TO DO: check +/- px_size_nm/2
#        y_nm = np.arange(ymin_nm + px_size_nm/2, ymax_nm + px_size_nm/2, px_size_nm)
        size = array.shape[0]
        x_nm = np.linspace(xmin_nm + px_size_nm/2, xmax_nm + px_size_nm/2, size)
        y_nm = np.linspace(ymin_nm + px_size_nm/2, ymax_nm + px_size_nm/2, size)
        
        print('[scan] x_nm', x_nm)
        print('[scan] x_nm shape', x_nm.shape)
        
        print('[scan] y_nm', y_nm)
        print('[scan] y_nm shape', y_nm.shape)
        
        (Mx_nm, My_nm) = np.meshgrid(x_nm, y_nm)
        
        print('[scan] shape grid', Mx_nm.shape)
        
        # make initial guess for parameters
        
        offset = np.min(array)
        d = 300 # nm
        x0 = (xmin_nm + xmax_nm)/2
        y0 = (ymin_nm + ymax_nm)/2
        A = np.max(array)*d**2 # check this estimation ????
        
        if function == 'doughnut':
        
          initial_guess = [A, x0, y0, d, offset]
      
          popt, pcov = opt.curve_fit(PSF.doughnut2D, (Mx_nm, My_nm), array.ravel(), p0=initial_guess)
          
          # retrieve results
  
          print('[scan] doughnut fit parameters', popt)
          
          dougnutFit = PSF.doughnut2D((Mx_nm, My_nm), *popt).reshape(shape)
          
          plt.figure()
          plt.imshow(dougnutFit, cmap=cmaps.parula, interpolation='None', extent=extent)
          plt.xlabel('x (nm)')
          plt.ylabel('y (nm)')
          
          x0_fit = popt[1]
          y0_fit = popt[2]
          
          doughnut_center = np.array([x0_fit, y0_fit, 0], dtype=np.float64)/1000 # in µm
          target = self.initialPos + doughnut_center 
          
          print('[scan] target', target)
          
        if function == 'gaussian':
          
          σ_x = 130
          σ_y = 130
          initial_guess = [A, x0, y0, σ_x, σ_y, offset]
          
          popt, pcov = opt.curve_fit(PSF.gaussian2D, (Mx_nm, My_nm), array.ravel(), p0=initial_guess)
        
          print('[scan] gaussian fit parameters', popt)
          
          gaussianFit = PSF.gaussian2D((Mx_nm, My_nm), *popt).reshape(shape)
          
          plt.figure()
          plt.imshow(gaussianFit, cmap=cmaps.parula, interpolation='None', extent=extent)
          plt.xlabel('x (nm)')
          plt.ylabel('y (nm)')
          
          x0_fit = popt[1]
          y0_fit = popt[2]
          
          gaussian_center = np.array([x0_fit, y0_fit, 0], dtype=np.float64)/1000 # in µm
          target = self.initialPos + gaussian_center 
        
        return target

    @pyqtSlot()
    def get_moveTo_initial_signal(self):
        
        self.moveTo(*self.initialPos)
    
    def relative_move(self, axis, direction):
        
        if axis == 'x' and direction == 'up':
            
            newPos_µm = self.initialPos[0] - self.xStep
            newPos_µm = round(newPos_µm, 3)
            self.initialPos = np.array([newPos_µm, self.initialPos[1],
                                        self.initialPos[2]])
            
        if axis == 'x' and direction == 'down':
            
            newPos_µm = self.initialPos[0] + self.xStep
            newPos_µm = np.around(newPos_µm, 3)
            self.initialPos = np.array([newPos_µm, self.initialPos[1],
                                        self.initialPos[2]])
            
        if axis == 'y' and direction == 'up':
            
            newPos_µm = self.initialPos[1] + self.yStep
            newPos_µm = np.around(newPos_µm, 3)       
            self.initialPos = np.array([self.initialPos[0], newPos_µm,
                                        self.initialPos[2]])
            
        if axis == 'y' and direction == 'down':
            
            newPos_µm = self.initialPos[1] - self.yStep
            newPos_µm = np.around(newPos_µm, 3)
            self.initialPos = np.array([self.initialPos[0], newPos_µm,
                                        self.initialPos[2]])
            
        if axis == 'z' and direction == 'up':
            
            newPos_µm = self.initialPos[2] + self.zStep
            newPos_µm = np.around(newPos_µm, 3)
            self.initialPos = np.array([self.initialPos[0], self.initialPos[1], 
                                        newPos_µm])
        
        if axis == 'z' and direction == 'down':
            
            newPos_µm = self.initialPos[2] - self.zStep
            newPos_µm = np.around(newPos_µm, 3)
            self.initialPos = np.array([self.initialPos[0], self.initialPos[1], 
                                        newPos_µm])
    
        self.update_device_param()
        self.emit_param()    
            
    def save_current_frame(self):
      
        self.save_FB = True
        
        # experiment parameters
        
        name = tools.getUniqueName(self.filename)
        now = time.strftime("%c")
        tools.saveConfig(self, now, name)

        # save image
        
        data = self.image_to_save
        result = Image.fromarray(data.astype('uint16'))
        result.save(r'{}.tiff'.format(name))
        
        if self.save_FB is True:
          
          print('[scan] Saved current frame F and B', name)
          
          # save image F
          
          data = self.imageF_copy
          result = Image.fromarray(data.astype('uint16'))
          result.save(r'{} F.tiff'.format(name))
          
          # save image B
          
          data = self.imageB_copy
          result = Image.fromarray(data.astype('uint16'))
          result.save(r'{} B.tiff'.format(name))
        
        print('[scan] Saved current frame', name)

#        self.gui.currentFrameButton.setChecked(False)
        
    @pyqtSlot(bool, str, np.ndarray)
    def get_scan_signal(self, lvbool, mode, initialPos):
    
        """
        Connection: [psf] scanSignal
        Description: get drift-corrected initial position, calculates the
        derived parameters (and updates ADwin data)
        """
        
        self.initialPos = initialPos
        self.calculate_derived_param()
        
        self.liveview(lvbool, mode)
        
    def line_acquisition(self):
        
        self.adw.Start_Process(1)
        
        line_time = (1/1000) * self.data_t[-1]  # target linetime in ms
        wait_time = line_time * 1.05 # TO DO: optimize this, it should work with 1.00, or maybe even less?
                                     # it should even work without the time.sleep()
        
        time.sleep(wait_time/1000) # in s

        line_data = self.adw.GetData_Long(1, 0, self.tot_pixels)
        
        line_data[0] = 0  # TO DO: fix the high count error on first element

        return line_data
   
    @pyqtSlot(bool, str)
    def liveview(self, lvbool, mode):
        
        if lvbool:
            
            self.acquisitionMode = mode # modes: 'liveview', 'frame'
            self.liveview_start()
            
        else:
            
            self.liveview_stop()

    def liveview_start(self):
        
        self.plot_scan()

        if self.scantype == 'xy':

            self.moveTo(self.x_i, self.y_i, self.z_i)

        if self.scantype == 'xz':

            self.moveTo(self.x_i, self.y_i + self.scanRange/2,
                        self.z_i - self.scanRange/2)

        if self.scantype == 'yz':

            self.moveTo(self.x_i + self.scanRange/2, self.y_i,
                        self.z_i - self.scanRange/2)

        self.viewtimer.start(self.viewtimer_time)

    def liveview_stop(self):

        self.viewtimer.stop()

    def update_view(self):
      
        if self.i < self.NofPixels:

            if self.scantype == 'xy':
    
                dy = tools.convert(self.dy, 'ΔXtoU')
                self.y_offset = int(self.y_offset + dy)
                self.adw.Set_FPar(2, self.y_offset)
    
            if self.scantype == 'xz' or self.scantype == 'yz':
    
                dz = tools.convert(self.dz, 'ΔXtoU')
                self.z_offset = int(self.z_offset + dz)
                self.adw.Set_FPar(2, self.z_offset)
    
            self.lineData = self.line_acquisition()
    
            if self.full_scan is True:
    
                self.image[self.i, :] = self.lineData
                
            elif self.FBaverage_scan is True:
                
                # display average of forward and backward image
                
#                c0 = self.NofAuxPixels
#                c1 = self.NofPixels
                
                c0 = self.NofAuxPixels+1  # this +1 is an empirical temporary fix
                c1 = self.NofPixels
                
                lineData_F = self.lineData[c0:c0+c1]
                lineData_B = self.lineData[3*c0+c1:3*c0+2*c1]
                
                if self.i % 2 == 0:
                
                    self.image[self.i, :] = lineData_F
                
                if self.i % 2 != 0:
                    
                    self.image[self.i, :] = lineData_B[::-1]
    
            else: 
    
                # displays only forward image
                
#                c0 = self.NofAuxPixels
#                c1 = self.NofPixels
                
                c0 = self.NofAuxPixels+1  # this +1 is an empirical temporary fix
                c1 = self.NofPixels
                
                print('[scan] c0', c0)
                print('[scan] c1', c1)
    
                lineData_F = self.lineData[c0:c0+c1]
                lineData_B = self.lineData[3*c0+c1:3*c0+2*c1]
                
                self.imageF[self.i, :] = lineData_F
                self.imageB[self.i, :] = lineData_B[::-1]
                
                self.image[self.i, :] = lineData_F
    
            # display image after every scanned line
                
            self.image_to_save = self.image
            self.imageF_copy = self.imageF     # TO DO: clean up with fit signal to avoid the copy image
            self.imageB_copy = self.imageB
    #        self.image_copy = self.image
            
            self.imageSignal.emit(self.image)
    #        print(datetime.now(), '[scan] Image emitted to frontend')
    
    #        if self.i < self.NofPixels-1:
    #
    #            self.i = self.i + 1
    
            self.i = self.i + 1

        else:

            print(datetime.now(), '[scan] Frame ended')

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
                
            if self.acquisitionMode == 'frame':
                
                self.liveview_stop()
                self.frameIsDone.emit(True, self.image)
                
            self.update_device_param()  
            
    @pyqtSlot(bool)
    def toggle_shutter(self, val):
        
        if val is True:
            
            self.shutter_state = True
            
            self.adw.Set_Par(55, 0)
            self.adw.Set_Par(50, 1)
            self.adw.Start_Process(5)
            
            print('[scan] Shutter opened')
            
        if val is False:
            
            self.shutte_state = False
            
            self.adw.Set_Par(55, 0)
            self.adw.Set_Par(50, 0)
            self.adw.Start_Process(5)

            print('[scan] Shutter closed')
            
    @pyqtSlot(bool)
    def toggle_flipper(self, val):
        
        if val is True:
            
            self.flipper_state = True
            
            self.adw.Set_Par(55, 1)
            self.adw.Set_Par(51, 1)
            self.adw.Start_Process(5)
            
            print('[scan] Flipper up')
            
        if val is False:
            
            self.flipper_state = False
            
            self.adw.Set_Par(55, 1)
            self.adw.Set_Par(51, 0)
            self.adw.Start_Process(5)

            print('[scan] Flipper down')
    
    @pyqtSlot(bool)        
    def toggle_FBav_scan(self, val):
        
        if val is True:
            
            self.FBaverage_scan = True
        
        if val is False:
            
            self.FBaverage_scan = False
        
    def emit_ROI_center(self):
        
        self.ROIcenterSignal.emit(self.ROIcenter)
        
        print('[scan] ROI center emitted')
        
    def trace_measurement(self):
      
        n = 100
        pxtime = 1000
        trace_data = self.trace_acquisition(Npoints=n, pixeltime=pxtime)
        
        time = np.arange(n)
        
        plt.style.use('dark_background')
        plt.figure()
        plt.plot(time, trace_data)
        plt.xlabel('Time (ms)')
        plt.ylabel('Count rate (kHz)')
        
    def trace_acquisition(self, Npoints, pixeltime):
      
        """ 
        Method to acquire a trace of photon counts at the current position.
        
        Npoints = number of points to be acquired (max = 1024)
        pixeltime = time per point (in μs)
        
        
        """
        
        # pixeltime in μs
      
        self.adw.Set_FPar(65, tools.timeToADwin(pixeltime))
        self.adw.Set_Par(60, Npoints+1)

        self.adw.Start_Process(6)
        
        trace_time = Npoints * (pixeltime/1000)  # target linetime in ms
        wait_time = trace_time * 1.05 # TO DO: optimize this, it should work with 1.00, or maybe even less?
                                     # it should even work without the time.sleep()
        
        time.sleep(wait_time/1000) # in s

        trace_data = self.adw.GetData_Long(6, 0, Npoints+1)
        
        trace_data = trace_data[1:]# TO DO: fix the high count error on first element

        return trace_data
        
    def plot_scan(self):
        
        # save scan plot (x vs t)
        plt.figure()
        plt.title('Scan plot x vs t')
        plt.plot(self.data_t_adwin[0:-1], self.data_x_adwin, 'go')
        plt.xlabel('t (ADwin time)')
        plt.ylabel('V (DAC units)')
        
#        c0 = self.NofAuxPixels
        c0 = self.NofAuxPixels+1
        c1 = self.NofPixels
        
        plt.plot(self.data_t_adwin[c0], self.data_x_adwin[c0], 'r*')
        plt.plot(self.data_t_adwin[c0+c1-1], self.data_x_adwin[c0+c1-1], 'r*')
                
        plt.plot(self.data_t_adwin[3*c0+c1], self.data_x_adwin[3*c0+c1], 'r*')
        plt.plot(self.data_t_adwin[3*c0+2*c1-1], self.data_x_adwin[3*c0+2*c1-1], 'r*')

        fname = tools.getUniqueName(self.filename)
        fname = fname + '_scan_plot'
        plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None)

                      
    def make_connection(self, frontend):
        
        frontend.liveviewSignal.connect(self.liveview)
        frontend.moveToROIcenterButton.clicked.connect(self.moveTo_roi_center)
        frontend.currentFrameButton.clicked.connect(self.save_current_frame)
        frontend.moveToButton.clicked.connect(self.moveTo_action)
        frontend.paramSignal.connect(self.get_frontend_param)
        frontend.closeSignal.connect(self.stop)
        
        frontend.fitPSFSignal.connect(self.get_ROI_coords_and_fit)
        self.auxFitSignal.connect(self.psf_fit_FandB)
        
        frontend.shutterButton.clicked.connect(lambda: self.toggle_shutter(frontend.shutterButton.isChecked()))
        frontend.flipperButton.clicked.connect(lambda: self.toggle_flipper(frontend.flipperButton.isChecked()))
        frontend.FBavScanButton.clicked.connect(lambda: self.toggle_FBav_scan(frontend.FBavScanButton.isChecked()))
        
        frontend.xUpButton.pressed.connect(lambda: self.relative_move('x', 'up'))
        frontend.xDownButton.pressed.connect(lambda: self.relative_move('x', 'down'))
        frontend.yUpButton.pressed.connect(lambda: self.relative_move('y', 'up'))
        frontend.yDownButton.pressed.connect(lambda: self.relative_move('y', 'down'))        
        frontend.zUpButton.pressed.connect(lambda: self.relative_move('z', 'up'))
        frontend.zDownButton.pressed.connect(lambda: self.relative_move('z', 'down'))
          
    def stop(self):

        self.toggle_shutter(False)
        self.liveview_stop()
        
        # Go back to 0 position

        x_0 = 0
        y_0 = 0
        z_0 = 0

        self.moveTo(x_0, y_0, z_0)
        

if __name__ == '__main__':

    app = QtGui.QApplication([])
    app.setStyle(QtGui.QStyleFactory.create('fusion'))
#    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    
    DEVICENUMBER = 0x1
    adw = ADwin.ADwin(DEVICENUMBER, 1)
    setupDevice(adw)
    
    worker = Backend(adw)    
    gui = Frontend()
    
    worker.make_connection(gui)
    gui.make_connection(worker)
    
    gui.emit_param()
    worker.emit_param()
    
#
    scanThread = QtCore.QThread()
    worker.moveToThread(scanThread)
    worker.viewtimer.moveToThread(scanThread)
    worker.viewtimer.timeout.connect(worker.update_view)
    
    scanThread.start()

    
    gui.setWindowTitle('scan')
    gui.show()

    app.exec_()
