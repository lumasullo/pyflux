# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 13:41:48 2018

@authors: Luciano Masullo
"""

import numpy as np
import time
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from scipy import optimize as opt

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import Dock, DockArea
import pyqtgraph.ptime as ptime

# install from https://instrumental-lib.readthedocs.io/en/stable/install.html
from instrumental.drivers.cameras import uc480
import tools.viewbox_tools as viewbox_tools
import tools.tools as tools
import tools.colormaps as cmaps
import tools.pi as pi
import scan
import drivers.ADwin as ADwin

def actuatorParameters(adwin, z_f, n_pixels_z=50, pixeltime=1000):

    z_f = tools.convert(z_f, 'XtoU')

    adwin.Set_Par(23, n_pixels_z)
    
    adwin.Set_FPar(25, z_f)

    adwin.Set_FPar(26, tools.timeToADwin(pixeltime))

def zMoveTo(adwin, z_f):

    actuatorParameters(adwin, z_f)
    adwin.Start_Process(3)


class focusWidget(QtGui.QFrame):

    def __init__(self, camera, actuator=None, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        self.setMinimumSize(2, 200)

        self.cam = camera
        self.roi = None
        self.cropped = False

        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)

        self.fworker = focusWorker(self, camera, actuator)
        
        self.setUpGUI()

    def ROImethod(self):
        
        if self.cropped is True:  # code to go back to the (1280 x 1024) ROI
            
            x0 = 0
            y0 = 0
            x1 = 1280 
            y1 = 1024 
            self.fworker.camera._set_AOI(x0, y0, x1, y1)
            self.cropped = False
        
        ROIpen = pg.mkPen(color='y')

        if self.roi is None:

            ROIpos = (0, 0)
            self.roi = viewbox_tools.ROI(300, self.vb, ROIpos,
                                         handlePos=(1, 0),
                                         handleCenter=(0, 1),
                                         scaleSnap=True,
                                         translateSnap=True,
                                         pen=ROIpen)

        else:

            self.vb.removeItem(self.roi)
            self.roi.hide()

            ROIpos = (0, 0)
            self.roi = viewbox_tools.ROI(300, self.vb, ROIpos,
                                         handlePos=(1, 0),
                                         handleCenter=(0, 1),
                                         scaleSnap=True,
                                         translateSnap=True,
                                         pen=ROIpen)
            
    def selectROI(self):
        
        self.cropped = True
        self.getStats = True
        
        array = self.roi.getArrayRegion(self.fworker.image, self.img)
        ROIpos = np.array(self.roi.pos())
        
        print('max image size', self.cam._get_max_img_size())
        print('initial ROI size', self.cam._get_AOI())
        
        print('ROIpos', ROIpos)
        print('array', np.shape(array))
        
        y0 = int(ROIpos[0])
        x0 = int(ROIpos[1])
        y1 = int(ROIpos[0] + np.shape(array)[0])
        x1 = int(ROIpos[1] + np.shape(array)[1])

        
        print('setpoint ROI', [x0, y0, x1, y1])

        self.fworker.camera._set_AOI(x0, y0, x1, y1)
        print('focus lock ROI changed to', self.fworker.camera._get_AOI())
        
        self.vb.removeItem(self.roi)
        self.roi.hide()
        self.roi = None
        
    def setUpGUI(self):
        
         # Focus lock widget
        
        self.kpEdit = QtGui.QLineEdit('-0.3')
        self.kpEdit.setFixedWidth(60)

        self.kpLabel = QtGui.QLabel('kp')
        self.kiEdit = QtGui.QLineEdit('0.00001')
        self.kiEdit.setFixedWidth(60)
        self.kiLabel = QtGui.QLabel('ki')
        self.lockButton = QtGui.QPushButton('Lock focus')
        self.lockButton.setCheckable(True)
        self.lockButton.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                      QtGui.QSizePolicy.Expanding)
        moveLabel = QtGui.QLabel('Move [nm]')
        moveLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.moveEdit = QtGui.QLineEdit('0')
        self.moveEdit.setFixedWidth(60)
        self.ROIbutton = QtGui.QPushButton('ROI')
        self.selectROIbutton = QtGui.QPushButton('select ROI')
        self.calibrationButton = QtGui.QPushButton('Calibrate')
        
        self.kiEdit.textChanged.connect(self.fworker.unlockFocus)
        self.kpEdit.textChanged.connect(self.fworker.unlockFocus)
        self.lockButton.clicked.connect(self.fworker.toggleFocus)
        self.ROIbutton.clicked.connect(self.ROImethod)
        self.selectROIbutton.clicked.connect(self.selectROI)
        self.calibrationButton.clicked.connect(self.fworker.calibrate)

        self.focusPropertiesDisplay = QtGui.QLabel(' st_dev = 0  max_dev = 0')

        # focus camera display
        
        self.camDisplay = pg.GraphicsLayoutWidget()
        self.vb = self.camDisplay.addViewBox(row=0, col=0)

        self.vb.setMouseMode(pg.ViewBox.RectMode)
        self.img = pg.ImageItem()
        self.img.translate(-0.5, -0.5)
        self.vb.addItem(self.img)
        self.vb.setAspectLocked(True)

        self.hist = pg.HistogramLUTItem(image=self.img)   # set up histogram for the liveview image
        lut = viewbox_tools.generatePgColormap(cmaps.inferno)
        self.hist.gradient.setColorMap(lut)
        self.hist.vb.setLimits(yMin=0, yMax=10000)

        for tick in self.hist.gradient.ticks:
            tick.hide()
            
        self.camDisplay.addItem(self.hist, row=0, col=1)
        
        # focus lock graph
        
        self.focusGraph = pg.GraphicsWindow()
        self.focusGraph.setAntialiasing(True)
        
        self.focusGraph.statistics = pg.LabelItem(justify='right')
        self.focusGraph.addItem(self.focusGraph.statistics, row=0, col=0)
        self.focusGraph.statistics.setText('---')
        
        self.focusGraph.zPlot = self.focusGraph.addPlot(row=0, col=0)
        self.focusGraph.zPlot.setLabels(bottom=('Time', 's'),
                                        left=('CM x position', 'px'))
        self.focusGraph.zPlot.showGrid(x=True, y=True)
        self.focusCurve = self.focusGraph.zPlot.plot(pen='y')

        # GUI layout
        
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        
        # parameters widget
        
        self.paramWidget = QtGui.QFrame()
        self.paramWidget.setFrameStyle(QtGui.QFrame.Panel |
                                       QtGui.QFrame.Raised)
        
        self.paramWidget.setFixedHeight(150)
        self.paramWidget.setFixedWidth(200)
        
        subgrid = QtGui.QGridLayout()
        self.paramWidget.setLayout(subgrid)
        
        subgrid.addWidget(self.calibrationButton, 0, 0, 1, 4)
        subgrid.addWidget(self.kpLabel, 1, 2)
        subgrid.addWidget(self.kpEdit, 1, 3)
        subgrid.addWidget(self.kiLabel, 2, 2)
        subgrid.addWidget(self.kiEdit, 2, 3)
        subgrid.addWidget(self.lockButton, 3, 0, 1, 4)
        subgrid.addWidget(self.ROIbutton, 1, 0)
        subgrid.addWidget(self.selectROIbutton, 2, 0)
        

        
        grid.addWidget(self.paramWidget, 0, 0)
        grid.addWidget(self.focusGraph, 0, 1)
        grid.addWidget(self.camDisplay, 0, 2)

        
    def closeEvent(self, *args, **kwargs):
        
        self.fworker.closeEvent()
        
        super().closeEvent(*args, **kwargs)
        
        

class focusWorker(QtCore.QObject):

    def __init__(self, gui, camera, actuator, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gui = gui
        self.camera = camera
        self.actuator = actuator
        self.locked = False
        self.standAlone = False
    
        self.npoints = 400
        
        # checks image size
        
        rawimage = self.camera.latest_frame()
        image = np.sum(rawimage, axis=2)
        
        self.sensorSize = np.array(image.shape)
        self.focusSignal = 0
        
        # set focus update rate
        
        self.scansPerS = 20
        self.camera.start_live_video(framerate='20 Hz')

        self.focusTime = 1000 / self.scansPerS
        self.focusTimer = QtCore.QTimer()
        self.focusTimer.timeout.connect(self.update)
        self.focusTimer.start(self.focusTime)
        
        self.currentZ = tools.convert(self.actuator.Get_FPar(52), 'UtoX')
        self.currentX = tools.convert(self.actuator.Get_FPar(50), 'UtoX')
        self.currentY = tools.convert(self.actuator.Get_FPar(51), 'UtoX')
        
        self.reset()
        
    def setupPI(self):
        
        self.setPoint = self.focusSignal
        self.PI = pi.PI(self.setPoint, 0.001, np.float(self.gui.kpEdit.text()),
                        np.float(self.gui.kiEdit.text()))

        self.initialZ = self.currentZ
        
    def updatePI(self):
        
        # TO DO: fix this function

        self.distance = self.currentZ - self.initialZ
        
        cm = self.focusSignal
        out = self.PI.update(cm)

#        self.lockN += 1
#        self.lockMean += (cm - self.lockMean)/(self.lockN + 1)
#        self.graph.setLine.setValue(self.lockMean)

        # Safety unlocking
        
        if abs(self.distance) > 1  or abs(out) > 3 or self.currentZ < 0:  # in µm
        
            self.unlockFocus()
            
        else:
            
            self.currentZ = self.currentZ + out
#            print('moved to', self.currentZ, 'µm')
            
        zMoveTo(self.actuator, self.currentZ)
            

    def updateStats(self):
        
        # TO DO: fix this function

        signal = self.focusSignal

        if self.n == 1:
            self.mean = signal
            self.mean2 = self.mean**2
        else:
            self.mean += (signal - self.mean)/self.n
            self.mean2 += (signal**2 - self.mean2)/self.n

        # Stats
        self.std = np.sqrt(self.mean2 - self.mean**2)
        self.max_dev = np.max([self.max_dev,
                              self.focusSignal - self.setPoint])
        statData = 'std = {}    max_dev = {}'.format(np.round(self.std, 3),
                                                     np.round(self.max_dev, 3))
        self.gui.focusGraph.statistics.setText(statData)

        self.n += 1
        
    def update(self, delay=0.000):
        
        time0 = time.time()

        time.sleep(delay)
        
        raw_image = self.camera.latest_frame()
        
#        r = raw_image[:, :, 0]
#        g = raw_image[:, :, 1]
#        b = raw_image[:, :, 2]
        
        image = np.sum(raw_image, axis=2)
        
        self.image = image
        
        #  The croppingis done because otherwise the displayed image will be
        #  300 x 1024. It doesn't affect the performance of the system
        
        if self.gui.cropped is False: 
            
            self.gui.img.setImage(self.image, autoLevels=False)
        
        else:

            croppedimg = self.image[0:300, 0:300]
            self.gui.img.setImage(croppedimg, autoLevels=False)
            
        # get mass center
            
        self.massCenter = np.array(ndi.measurements.center_of_mass(image))
        self.focusSignal = self.massCenter[0]
    
        # update of the data displayed in the graph

        if self.ptr < self.npoints:
            self.data[self.ptr] = self.focusSignal
            self.time[self.ptr] = ptime.time() - self.startTime
            
            self.gui.focusCurve.setData(self.time[1:self.ptr + 1],
                                        self.data[1:self.ptr + 1])

        else:
            self.data[:-1] = self.data[1:]
            self.data[-1] = self.focusSignal
            self.time[:-1] = self.time[1:]
            self.time[-1] = ptime.time() - self.startTime

            self.gui.focusCurve.setData(self.time, self.data)

        self.ptr += 1
        
        # update PI
        
        if self.locked:
            self.updatePI()
            self.updateStats()
            
        time1 = time.time()
        
#        print('focus signal update took {}'.format(np.around(time1-time0, 6))) 
        
    def toggleFocus(self, delay=0):
        
        if self.gui.lockButton.isChecked():

            self.reset()
            self.setupPI()
            self.update(delay)
            
            self.setpointLine = self.gui.focusGraph.zPlot.addLine(y=self.setPoint, pen='r')
            
            self.locked = True

        else:
            self.unlockFocus()

    def unlockFocus(self):
        
        if self.locked is True:
            self.locked = False
            self.gui.lockButton.setChecked(False)
            self.gui.focusGraph.zPlot.removeItem(self.setpointLine)
            
    def calibrate(self):
        
        self.focusTimer.stop()
        time.sleep(0.100)
        
        nsteps = 40
        xmin = 9.5  # in µm
        xmax = 10.5   # in µm
        xrange = xmax - xmin  
        
        calibData = np.zeros(40)
        xData = np.arange(xmin, xmax, xrange/nsteps)
        
        zMoveTo(self.actuator, xmin)
        
        time.sleep(0.100)
        
        for i in range(nsteps):
            
            zMoveTo(self.actuator, xmin + (i * 1/nsteps) * xrange)
            self.update()
            calibData[i] = self.focusSignal
            
        plt.plot(xData, calibData, 'o')
            
        time.sleep(0.200)
        
        self.focusTimer.start(self.focusTime)
            
            
    def reset(self):
        
        self.data = np.zeros(self.npoints)
        self.time = np.zeros(self.npoints)
        self.ptr = 0
        self.startTime = ptime.time()

        self.max_dev = 0
        self.mean = self.focusSignal
        self.std = 0
        self.n = 1
        
    def closeEvent(self, *args, **kwargs):
        
        self.focusTimer.stop()
        self.camera.close()
        
        if self.standAlone is True:
            zMoveTo(self.actuator, 0)
            

    

if __name__ == '__main__':
    
    print('Focus lock module running in stand-alone mode')

    app = QtGui.QApplication([])
    cam = uc480.UC480_Camera()
    
    DEVICENUMBER = 0x1
    adw = ADwin.ADwin(DEVICENUMBER, 1)
    scan.setupDevice(adw)
    
    # initialize fpar_52 (z) ADwin position parameters
        
    pos_zero = tools.convert(0, 'XtoU')
        
    adw.Set_FPar(52, pos_zero)
    
    zMoveTo(adw, 10)

    win = focusWidget(cam, adw)
    win.fworker.standAlone = True

    win.setWindowTitle('Focus lock')
    win.resize(1500, 500)

    win.show()
    app.exec_()
        