# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 11:59:13 2019

@author: Luciano A. Masullo
"""

import numpy as np
import time
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import ctypes as ct

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import Dock, DockArea
import pyqtgraph.ptime as ptime
import tools.viewbox_tools as viewbox_tools
import tools.colormaps as cmaps
import tools.PSF as PSF
from scipy import optimize as opt

import lantz.drivers.legacy.andor.ccd as ccd

class xyWidget(QtGui.QFrame):
    
    def __init__(self, camera, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        # TO DO: finish gaussian fit calculation and plot
        # TO DO: fix multithreading
        
        self.andor = camera
        cam = 0
        self.andor.current_camera = self.andor.camera_handle(cam)
        self.andor.lib.Initialize()
        print('idn:', self.andor.idn)
        
        self.lvworker = LiveViewWorker(self, self.andor)
        
        self.setUpGUI()
        
        # initial ROI parameters        
        
        self.NofPixels = 200
        self.roi = None
        self.ROInumber = 0
        self.roilist = []

    def createROI(self):
        
        ROIpen = pg.mkPen(color='r')

        ROIpos = (0.5 * self.NofPixels - 64, 0.5 * self.NofPixels - 64)
        self.roi = viewbox_tools.ROI2(self.NofPixels/2, self.vb, ROIpos,
                                     handlePos=(1, 0),
                                     handleCenter=(0, 1),
                                     scaleSnap=True,
                                     translateSnap=True,
                                     pen=ROIpen, number=self.ROInumber)
        
        self.ROInumber += 1
        
        self.roilist.append(self.roi)
        
        self.ROIButton.setChecked(False)

#        else:
#
#            self.vb.removeItem(self.roi)
#            self.roi.hide()

    def deleteROIs(self):
        
        for i in range(len(self.roilist)):
            
            self.vb.removeItem(self.roilist[i])
            self.roilist[i].hide()
            
        self.roilist = []
        self.deleteROIsButton.setChecked(False)
        self.ROInumber = 0
            
        
    def setUpGUI(self):
        
        # GUI layout
        
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        
        # parameters widget
        
        self.paramWidget = QtGui.QFrame()
        self.paramWidget.setFrameStyle(QtGui.QFrame.Panel |
                                       QtGui.QFrame.Raised)
        
        self.paramWidget.setFixedHeight(320)
        self.paramWidget.setFixedWidth(200)
        
        grid.addWidget(self.paramWidget, 0, 1)
        
        # image widget layout
        
        imageWidget = pg.GraphicsLayoutWidget()
        self.vb = imageWidget.addViewBox(row=0, col=0)
        self.vb.setMouseMode(pg.ViewBox.RectMode)
        self.img = pg.ImageItem()
        self.img.translate(-0.5, -0.5)
        self.vb.addItem(self.img)
        self.vb.setAspectLocked(True)
        imageWidget.setAspectLocked(True)
        grid.addWidget(imageWidget, 0, 0)
        
        # set up histogram for the liveview image

        self.hist = pg.HistogramLUTItem(image=self.img)
        lut = viewbox_tools.generatePgColormap(cmaps.parula)
        self.hist.gradient.setColorMap(lut)
#        self.hist.vb.setLimits(yMin=800, yMax=3000)

        ## TO DO: fix histogram range


        for tick in self.hist.gradient.ticks:
            tick.hide()
        imageWidget.addItem(self.hist, row=0, col=1)
        
        # xy drift graph (graph without a fixed range)
        
        self.xyGraph = pg.GraphicsWindow()
        self.xyGraph.setAntialiasing(True)
        
        self.xyGraph.statistics = pg.LabelItem(justify='right')
        self.xyGraph.addItem(self.xyGraph.statistics)
        self.xyGraph.statistics.setText('---')
        
        self.xyGraph.xPlot = self.xyGraph.addPlot(row=0, col=0)
        self.xyGraph.xPlot.setLabels(bottom=('Time', 's'),
                            left=('X position', 'nm'))
        self.xyGraph.xPlot.showGrid(x=True, y=True)
        self.xCurve = self.xyGraph.xPlot.plot(pen='b')
        
        self.xyGraph.yPlot = self.xyGraph.addPlot(row=1, col=0)
        self.xyGraph.yPlot.setLabels(bottom=('Time', 's'),
                            left=('Y position', 'nm'))
        self.xyGraph.yPlot.showGrid(x=True, y=True)
        self.yCurve = self.xyGraph.yPlot.plot(pen='r')
        
        # LiveView Button

        self.liveviewButton = QtGui.QPushButton('camera LIVEVIEW')
        self.liveviewButton.setCheckable(True)
        self.liveviewButton.clicked.connect(self.lvworker.liveview)
        
        # create ROI button
    
        self.ROIButton = QtGui.QPushButton('add ROI')
        self.ROIButton.setCheckable(True)
        self.ROIButton.clicked.connect(self.createROI)
        
        # delete ROI button
        
        self.deleteROIsButton = QtGui.QPushButton('delete ROIs')
        self.deleteROIsButton.setCheckable(True)
        self.deleteROIsButton.clicked.connect(self.deleteROIs)
        
        # position tracking checkbox
        
        self.trackingBeadsBox = QtGui.QCheckBox('track beads positions')
        
        # buttons and param layout
        
        subgrid = QtGui.QGridLayout()
        self.paramWidget.setLayout(subgrid)

        subgrid.addWidget(self.liveviewButton, 0, 0)
        subgrid.addWidget(self.ROIButton, 1, 0)
        subgrid.addWidget(self.deleteROIsButton, 2, 0)
        subgrid.addWidget(self.trackingBeadsBox, 3, 0)
        
        grid.addWidget(self.xyGraph, 1, 0)
        
        
    def closeEvent(self, *args, **kwargs):
        
        self.andor.shutter(0, 2, 0, 0, 0)
        self.andor.abort_acquisition()
        self.andor.finalize()

        super().closeEvent(*args, **kwargs)
        
class LiveViewWorker(QtCore.QObject):
    
    def __init__(self, gui, andor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.andor = andor
        self.gui = gui
        self.setUpCamera()
        
        self.viewtimer = QtCore.QTimer()
        self.viewtimer.timeout.connect(self.updateView)
        
        self.n = 0  # number of frames that it are averaged, 0 means no average
        self.i = 0  # update counter
        self.npoints = 400
        
        self.reset()
        
    def setUpCamera(self):
        
        self.pxSize = 80  # in nm
        self.shape = (512, 512)
        self.expTime = 0.300   # in sec
        
        self.andor.set_exposure_time(self.expTime)
        self.andor.set_image(shape=self.shape)
        
        print('FOV size = {}'.format(self.shape))

        # Temperature

        self.andor.cooler_on = True
        self.andor.temperature_setpoint = -20   # in °C
        
        # Frame transfer mode
        
        self.andor.frame_transfer_mode = True
        print('Frame transfer mode =', self.andor.frame_transfer_mode)

        # Horizontal readout speed

        ad = 1   # 16-bit DAC
        typ = 0   # EM mode
        index = 0   # 1 MHz
        self.andor.lib.SetHSSpeed(ct.c_int(ad), ct.c_int(typ), ct.c_int(index))
        
        hrate = self.andor.true_horiz_shift_speed(index=0, typ=0, ad=1)
        print('Horizontal readout rate = {} MHz'.format(hrate.magnitude))
        
        # pre-amp GAIN

        self.andor.preamp = 2  # index 2 for preamp gain = 4.7 
        
        gain = self.andor.true_preamp(2)
        print('PreAmp gain = {}'.format(np.round(gain, 1)))

        # EM GAIN
        
        self.andor.EM_gain_mode = 'DAC255'
        self.andor.EM_gain = 1  # EM gain set to 100

        print('EM gain = {}'.format(self.andor.EM_gain))
    
        # Vertical shift speed
        
        self.andor.vert_shift_speed = 4
        
        vspeed = self.andor.true_vert_shift_speed(4)
        print('Vertical shift speed = {} µs'.format(np.round(vspeed.magnitude,
                                                             1)))
    
    def liveview(self):

        if self.gui.liveviewButton.isChecked():
            self.liveviewStart()

        else:
            self.liveviewStop()

        
    def liveviewStart(self):
        
        self.initial = True
        
        print('Temperature = {} °C'.format(self.andor.temperature))
        print(self.andor.temperature_status)

#        # Initial image
        
        self.andor.acquisition_mode = 'Run till abort'
        print('Acquisition mode:', self.andor.acquisition_mode)
        self.andor.shutter(0, 1, 0, 0, 0)
        self.andor.start_acquisition()
        
        time.sleep(self.expTime * 2)
        self.image = self.andor.most_recent_image16(self.shape)
#
#        self.gui.img.setImage(np.transpose(self.image), autoLevels=False)
        self.gui.img.setImage(self.image, autoLevels=False)
#
##        self.vb.scene().sigMouseMoved.connect(self.mouseMoved)
#
        self.viewtimer.start(50)
    
    def liveviewStop(self):
        
        self.viewtimer.stop()
        self.andor.abort_acquisition()
        self.andor.shutter(0, 2, 0, 0, 0)
        
        self.gui.liveviewButton.setChecked(False)
            
    def updateView(self):
        """ Image update while in Liveview mode
        """
        
        self.image = self.andor.most_recent_image16(self.shape)

#        self.gui.img.setImage(np.transpose(self.image), autoLevels=False)
        self.gui.img.setImage(self.image, autoLevels=False)

        if self.gui.trackingBeadsBox.isChecked():
            
            for i in range(len(self.gui.roilist)):
                
                self.trackBead(i, self.initial)
                self.update()
                
            
    def trackBead(self, i, initial=False):
        
        # gaussian fit for every selected particle
    
        array = self.gui.roilist[i].getArrayRegion(self.image, self.gui.img)
        
        ROIpos_nm = np.array(self.gui.roilist[i].pos()) * self.pxSize
        
        ymin = ROIpos_nm[1]
        ymax = ROIpos_nm[1] + np.shape(array)[1] * self.pxSize
        
        xmin = ROIpos_nm[0]
        xmax = ROIpos_nm[0] + np.shape(array)[0] * self.pxSize
        
        print('xmin', xmin)
        print('ymin', ymin)
        
        x = np.arange(xmin, xmax, self.pxSize)
        y = np.arange(ymin, ymax, self.pxSize)
        
        (Mx, My) = np.meshgrid(x, y)
        
        bkg = np.min(array)
        A = np.max(array) - bkg
        x0 = (xmax+xmin)/2
        y0 = (ymax+ymin)/2

        σ = 130   # in nm
        
        initial_guess_G = [A, x0, y0, σ, σ, bkg]
#
#                t0 = time.time()
#                
        poptG, pcovG = opt.curve_fit(PSF.gaussian2D, (Mx, My), array.ravel(), 
                                     p0=initial_guess_G)
        
        if initial is True:
            
            self.x0 = poptG[1] - xmin
            self.y0 = poptG[2] - ymin
            
            self.initial = False
            print('initial')
            
            dataG = PSF.gaussian2D((Mx, My), *poptG)
            dataG_2d = dataG.reshape(int(np.shape(array)[0]), int(np.shape(array)[0]))
            
#            plt.figure()
#            plt.imshow(array, interpolation=None, cmap=cmaps.parula)
#            
#            plt.figure()
#            plt.imshow(dataG_2d, interpolation=None, cmap=cmaps.parula)
            

        
        self.x = poptG[1] - xmin - self.x0
        self.y = poptG[2] - ymin - self.y0
        
        print(self.x, self.y)
        
        
    def update(self):
        """ Update the data displayed in the graphs
        """
        
        # i counter goes from 0 to n, at n it actually does the update
        
        if self.i == self.n:
            
            self.xPosition = self.x
            self.yPosition = self.y
    
            if self.ptr < self.npoints:
                self.xData[self.ptr] = self.xPosition
                self.yData[self.ptr] = self.yPosition
                self.time[self.ptr] = ptime.time() - self.startTime
                
                self.gui.xCurve.setData(self.time[1:self.ptr + 1],
                                        self.xData[1:self.ptr + 1])
                self.gui.yCurve.setData(self.time[1:self.ptr + 1],
                                        self.yData[1:self.ptr + 1])
    
            else:
                self.xData[:-1] = self.xData[1:]
                self.xData[-1] = self.xPosition
                self.yData[:-1] = self.yData[1:]
                self.yData[-1] = self.yPosition
                self.time[:-1] = self.time[1:]
                self.time[-1] = ptime.time() - self.startTime
                
                self.gui.xCurve.setData(self.time, self.xData)
                self.gui.yCurve.setData(self.time, self.yData)
    
            self.ptr += 1
            
        else:
            
            self.i += 1
            
            
    def reset(self):
        
        self.xData = np.zeros(self.npoints)
        self.yData = np.zeros(self.npoints)
        self.time = np.zeros(self.npoints)
        self.ptr = 0
        self.startTime = ptime.time()
            
if __name__ == '__main__':

    app = QtGui.QApplication([])
    andor = ccd.CCD()

    win = xyWidget(andor)
    win.setWindowTitle('xy drift correction')
    win.show()
    app.exec_()
        