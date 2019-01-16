# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 13:41:48 2014

@authors: Federico Barabas, Luciano Masullo
"""

import numpy as np
import time
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import Dock, DockArea
import pyqtgraph.ptime as ptime
from instrumental.drivers.cameras import uc480
import tools.viewbox_tools as viewbox_tools
import tools.tools as tools
import tools.colormaps as cmaps
import tools.PSF as PSF
from scipy import optimize as opt



from lantz import Q_

#import tormenta.control.instruments as instruments
import tools.pi as pi


class FocusWidget(QtGui.QFrame):

    def __init__(self, camera, actuator=None, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        self.setMinimumSize(2, 350)

        self.cam = camera
        self.roi = None

        self.setPoint = 0

        self.V = Q_(1, 'V')
        self.um = Q_(1, 'um')
        self.nm = Q_(1, 'nm')

        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)

        self.scansPerS = 20
        self.cam.start_live_video(framerate='20 Hz')
        self.naverage = 1
        self.displayRate = 5
        self.i = 0
        
        self.ProcessData = ProcessData(self.cam, self.naverage)

        self.focusTime = 1000 / self.scansPerS
        self.focusTimer = QtCore.QTimer()
        self.focusTimer.timeout.connect(self.update)
        self.focusTimer.start(self.focusTime)

        self.locked = False
        self.n = 1
        self.max_dev = 0
        
        self.setUpGUI()

        

#        self.focusCalib = FocusCalibration(self)
#        self.focusCalibThread = QtCore.QThread(self)
#        self.focusCalib.moveToThread(self.focusCalibThread)
#        self.focusCalibButton = QtGui.QPushButton('Calibrate')
#        self.focusCalibButton.clicked.connect(self.focusCalib.start)
#        self.focusCalibThread.start()

    def zMoveEdit(self):
#        self.zMove(float(self.moveEdit.text())/1000 * self.um)

        pass

    def zMove(self, step):
        if self.locked:
            self.unlockFocus()
            self.z.zMoveRelative(step)
            self.update()
            self.lockButton.setChecked(True)
            self.toggleFocus(1)

        else:
            self.z.zMoveRelative(step)

    def update(self):

        t0 = time.time()

        self.ProcessData.update()
        
        t1 = time.time()
#        print('t1', t1-t0)
#        
        self.graph.update()
        
        t2 = time.time()
#        print('t2', t2-t1)
        if self.i == self.displayRate:
        
            self.img.setImage(self.ProcessData.image, autoLevels=False)
#            print(np.shape(self.ProcessData.image))
            self.i = 0
            
        else:
            
            self.i += 1
        
        t3 = time.time()
#        print('t3', t3-t2)
        
        if self.locked:
            self.updatePI()

    def toggleFocus(self, delay=0):
        if self.lockButton.isChecked():

            self.graph.reset()
            self.ProcessData.update(delay)
            self.setPoint = self.ProcessData.focusSignal
            self.graph.line = self.graph.plot.addLine(y=self.setPoint, pen='r')
            self.PI = pi.PI(self.setPoint, 0.001, np.float(self.kpEdit.text()),
                            np.float(self.kiEdit.text()))

            self.lockN = 1
            self.lockMean = self.setPoint
            self.graph.setLine = self.graph.plot.addLine(y=self.lockMean,
                                                         pen='c')
            self.initialZ = self.z.zPosition
            self.locked = True

        else:
            self.unlockFocus()

    def unlockFocus(self):
        if self.locked:
            self.locked = False
            self.lockButton.setChecked(False)
            self.graph.plot.removeItem(self.graph.line)
            self.graph.plot.removeItem(self.graph.setLine)

    def updatePI(self):

        self.distance = self.z.zPosition - self.initialZ
        cm = self.ProcessData.focusSignal
        out = self.PI.update(cm)

        self.lockN += 1
        self.lockMean += (cm - self.lockMean)/(self.lockN + 1)
        self.graph.setLine.setValue(self.lockMean)

        # Safety unlocking
        if abs(self.distance) > 10 * self.um or abs(out) > 5:
            self.unlockFocus()
        else:
            self.z.zMoveRelative(out * self.um)

#    def exportData(self):
#
#        self.sizeofData = np.size(self.graph.savedDataSignal)
#        self.savedData = [np.ones(self.sizeofData)*self.setPoint,
#                          self.graph.savedDataSignal, self.graph.savedDataTime]
#        np.savetxt(self.mainRec.name + '_focusdata', self.savedData)
#        self.graph.savedDataSignal = []
#        self.graph.savedDataTime = []

    def analizeFocus(self):

        signal = self.ProcessData.focusSignal

        if self.n == 1:
            self.mean = signal
            self.mean2 = self.mean**2
        else:
            self.mean += (signal - self.mean)/self.n
            self.mean2 += (signal**2 - self.mean2)/self.n

        # Stats
        self.std = np.sqrt(self.mean2 - self.mean**2)
        self.max_dev = np.max([self.max_dev,
                              self.ProcessData.focusSignal - self.setPoint])
        statData = 'std = {}    max_dev = {}'.format(np.round(self.std, 3),
                                                     np.round(self.max_dev, 3))
        self.graph.statistics.setText(statData)

        self.n += 1

    def closeEvent(self, *args, **kwargs):
        self.focusTimer.stop()
        self.cam.close()
        super().closeEvent(*args, **kwargs)
        
    def ROImethod(self):
        
        ROIpen = pg.mkPen(color='y')

        if self.roi is None:

#            ROIpos = (0.5 * self.cam.width - 64, 0.5 * self.cam.height - 64)
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

#            ROIpos = (0.5 * self.cam.width - 64, 0.5 * self.cam.height - 64)
            ROIpos = (0, 0)
            self.roi = viewbox_tools.ROI(300, self.vb, ROIpos,
                                         handlePos=(1, 0),
                                         handleCenter=(0, 1),
                                         scaleSnap=True,
                                         translateSnap=True,
                                         pen=ROIpen)
            
    def selectROI(self):
        
        array = self.roi.getArrayRegion(self.ProcessData.image, self.img)
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

        self.cam._set_AOI(x0, y0, x1, y1)
        print('focus lock ROI changed to', self.cam._get_AOI())
        
        self.vb.removeItem(self.roi)
        self.roi.hide()
        
    def setUpGUI(self):
        
         # Focus lock widget
        
        self.kpEdit = QtGui.QLineEdit('4')
        self.kpEdit.setFixedWidth(60)

        self.kpLabel = QtGui.QLabel('kp')
        self.kiEdit = QtGui.QLineEdit('0.01')
        self.kiEdit.setFixedWidth(60)
        self.kiLabel = QtGui.QLabel('ki')
        self.lockButton = QtGui.QPushButton('Lock')
        self.lockButton.setCheckable(True)
        self.lockButton.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                      QtGui.QSizePolicy.Expanding)
        moveLabel = QtGui.QLabel('Move [nm]')
        moveLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.moveEdit = QtGui.QLineEdit('0')
        self.moveEdit.setFixedWidth(60)
        self.ROIbutton = QtGui.QPushButton('ROI')
        self.selectROIbutton = QtGui.QPushButton('select ROI')
        
        self.kiEdit.textChanged.connect(self.unlockFocus)
        self.kpEdit.textChanged.connect(self.unlockFocus)
        self.moveEdit.returnPressed.connect(self.zMoveEdit)
        self.lockButton.clicked.connect(self.toggleFocus)
        self.ROIbutton.clicked.connect(self.ROImethod)
        self.selectROIbutton.clicked.connect(self.selectROI)

        self.focusDataBox = QtGui.QCheckBox('Save focus data')
        self.focusPropertiesDisplay = QtGui.QLabel(' st_dev = 0  max_dev = 0')
        
        try:
            prevCal = np.around(np.loadtxt('calibration')[0]/10)
            text = '1 px --> {} nm'.format(prevCal)
            self.calibrationDisplay = QtGui.QLineEdit(text)
        except:
            self.calibrationDisplay = QtGui.QLineEdit('0 px --> 0 nm')

        self.calibrationDisplay.setReadOnly(False)

        self.camDisplay = pg.GraphicsLayoutWidget()
        self.vb = self.camDisplay.addViewBox(row=0, col=0)

        self.vb.setMouseMode(pg.ViewBox.RectMode)
        self.img = pg.ImageItem()
        self.img.translate(-0.5, -0.5)
        self.vb.addItem(self.img)
        self.vb.setAspectLocked(True)

        # set up histogram for the liveview image

        self.hist = pg.HistogramLUTItem(image=self.img)
        lut = viewbox_tools.generatePgColormap(cmaps.inferno)
        self.hist.gradient.setColorMap(lut)
        self.hist.vb.setLimits(yMin=0, yMax=10000)

        for tick in self.hist.gradient.ticks:
            tick.hide()
        self.camDisplay.addItem(self.hist, row=0, col=1)
        
        self.graph = FocusLockGraph(self, self.naverage)

        dockArea = DockArea()
        graphDock = Dock("Focus laser graph", size=(400, 200))
        graphDock.addWidget(self.graph)
        dockArea.addDock(graphDock)
        camDock = Dock("Focus laser view", size=(200, 200))
        camDock.addWidget(self.camDisplay)
        dockArea.addDock(camDock, 'right', graphDock)

        # GUI layout
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(dockArea, 0, 0, 1, 7)
#        grid.addWidget(self.focusCalibButton, 1, 0)
        grid.addWidget(self.calibrationDisplay, 2, 0)
        grid.addWidget(self.focusDataBox, 1, 1)
        grid.addWidget(moveLabel, 1, 2)
        grid.addWidget(self.moveEdit, 2, 2)
        grid.addWidget(self.kpLabel, 1, 3)
        grid.addWidget(self.kpEdit, 1, 4)
        grid.addWidget(self.kiLabel, 2, 3)
        grid.addWidget(self.kiEdit, 2, 4)
        grid.addWidget(self.lockButton, 1, 6, 2, 1)
        grid.addWidget(self.ROIbutton, 1, 5, 1, 1)
        grid.addWidget(self.selectROIbutton, 2, 5, 1, 1)

        grid.setColumnMinimumWidth(5, 170)

        
class ProcessData(QtCore.QObject):

    def __init__(self, camera, naverage, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # for self.i = 1 and self.n = 1, no average is computed
        
        self.n = naverage
        self.datamass = np.zeros(self.n)

        self.i = 1
        
        self.method = 'mass center'
        self.camera = camera
        
        # checks image size
        
        rawimage = self.camera.latest_frame()
        image = np.sum(rawimage, axis=2)
        
        self.sensorSize = np.array(image.shape)
        self.focusSignal = 0

    def update(self, delay=0.000):

        time.sleep(delay)
        
        raw_image = self.camera.latest_frame()
        
        r = raw_image[:, :, 0]
        g = raw_image[:, :, 1]
        b = raw_image[:, :, 2]

        image = np.sum(raw_image, axis=2)
        
        self.massCenterAux = np.array(ndi.measurements.center_of_mass(image))
        
        self.datamass[self.i-1] = self.massCenterAux[0]
        
        if self.i == self.n:
            
            self.focusSignal = np.mean(self.datamass)
            
            self.i = 0
            self.datamass = np.zeros(self.n)
            
        else:
            self.i += 1
        
        self.image = image
        
#        print('max R', np.max(r))
#        print('min R', np.min(r))


class FocusLockGraph(pg.GraphicsWindow):

    def __init__(self, focusWidget, naverage, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.n = naverage  # number of frames that it are averaged
        self.i = 0  # update counter
        
        self.focusWidget = focusWidget
        self.analize = self.focusWidget.analizeFocus
        self.focusDataBox = self.focusWidget.focusDataBox
        self.savedDataSignal = []
        self.savedDataTime = []
        self.savedDataPosition = []

        self.setWindowTitle('Focus')
        self.setAntialiasing(True)

        self.npoints = 400

        # Graph without a fixed range
        self.statistics = pg.LabelItem(justify='right')
        self.addItem(self.statistics)
        self.statistics.setText('---')
        self.plot = self.addPlot(row=1, col=0)
        self.plot.setLabels(bottom=('Time', 's'),
                            left=('Laser position', 'px'))
        self.plot.showGrid(x=True, y=True)
        self.focusCurve = self.plot.plot(pen='y')

        self.reset()

    def reset(self):
        self.data = np.zeros(self.npoints)
        self.time = np.zeros(self.npoints)
        self.ptr = 0
        self.startTime = ptime.time()

        self.focusWidget.n = 1
        self.focusWidget.max_dev = 0
        self.focusWidget.mean = self.focusWidget.ProcessData.focusSignal
        self.focusWidget.std = 0

    def update(self):
        """ Update the data displayed in the graphs
        """
        
        # i counter goes from 0 to n, at n it actually does the update
        
        if self.i == self.n:
            
            self.focusSignal = self.focusWidget.ProcessData.focusSignal
    
            if self.ptr < self.npoints:
                self.data[self.ptr] = self.focusSignal
                self.time[self.ptr] = ptime.time() - self.startTime
                self.focusCurve.setData(self.time[1:self.ptr + 1],
                                        self.data[1:self.ptr + 1])
    
            else:
                self.data[:-1] = self.data[1:]
                self.data[-1] = self.focusSignal
                self.time[:-1] = self.time[1:]
                self.time[-1] = ptime.time() - self.startTime
    
                self.focusCurve.setData(self.time, self.data)
    
            self.ptr += 1
            
        else:
            
            self.i += 1

if __name__ == '__main__':

    app = QtGui.QApplication([])
    cam = uc480.UC480_Camera()


    win = FocusWidget(cam)
    win.show()
    app.exec_()
        