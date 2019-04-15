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
from datetime import date
import os

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import Dock, DockArea
import pyqtgraph.ptime as ptime
import qdarkstyle # see https://stackoverflow.com/questions/48256772/dark-theme-for-in-qt-widgets

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot

import sys
sys.path.append('C:\Program Files\Thorlabs\Scientific Imaging\ThorCam')
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

    adwin.Set_Par(33, n_pixels_z)
    adwin.Set_FPar(35, z_f)
    adwin.Set_FPar(36, tools.timeToADwin(pixeltime))

def zMoveTo(adwin, z_f):

    actuatorParameters(adwin, z_f)
    adwin.Start_Process(3)

class Frontend(QtGui.QFrame):
    
    liveviewSignal = pyqtSignal(bool)
    changedROI = pyqtSignal(np.ndarray)  # oass new roi size
    closeSignal = pyqtSignal()
    lockFocusSignal = pyqtSignal(bool)
    saveDataSignal = pyqtSignal(bool)
    
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        self.roi = None
        self.cropped = False

        self.setup_gui()

    def ROImethod(self):
        
        if self.cropped is True:  # code to go back to the (1280 x 1024) ROI
            
            x0 = 0
            y0 = 0
            x1 = 1280 
            y1 = 1024 
            
            value = np.array([x0, y0, x1, y1])
            self.changedROI.emit(value)
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
    
        ROIpos = np.array(self.roi.pos())
        roisize = np.array(self.roi.size())
        
        y0 = int(ROIpos[0])
        x0 = int(ROIpos[1])
        y1 = int(ROIpos[0] + roisize[0])
        x1 = int(ROIpos[1] + roisize[1])
        
        value = np.array([x0, y0, x1, y1])
        
        self.changedROI.emit(value)
    
        self.vb.removeItem(self.roi)
        self.roi.hide()
        self.roi = None
        
#    def toggleFocus(self):
#        
#        if self.lockButton.isChecked():
#            
#            self.lockFocusSignal.emit(True)
#
##            self.setpointLine = self.focusGraph.zPlot.addLine(y=self.setPoint, pen='r')
#            
#        else:
#            
#            self.lockFocusSignal.emit(False)
            
    def toggle_liveview(self):
        
        if self.liveviewButton.isChecked():
            
            self.liveviewSignal.emit(True)
            print('focus live view started')
        
        else:
            
            self.liveviewSignal.emit(False)
            self.liveviewButton.setChecked(False)
            self.img.setImage(np.zeros((512,512)), autoLevels=False)
            print('focus live view stopped')
            
    def emit_save_data_state(self):
        
        if self.saveDataBox.isChecked():
            
            self.saveDataSignal.emit(True)
            
        else:
            
            self.saveDataSignal.emit(False)
            
#    def toggle_stats(self):
#        
#        if self.feedbackLoopBox.isChecked():
#        
#            self.focusMean = self.focusGraph.plot.addLine(y=self.setPoint,
#                                                          pen='c')
#        
#        else:
#            
#            self.focusGraph.removeItem(self.focusMean)
        
    @pyqtSlot(np.ndarray)
    def get_image(self, img):
        
        #  The croppingis done because otherwise the displayed image will be
        #  300 x 1024. It doesn't affect the performance of the system
        
        if self.cropped is False: 
            
            self.img.setImage(img, autoLevels=False)
        
        else:

            croppedimg = img[0:300, 0:300]
            self.img.setImage(croppedimg)
            
    @pyqtSlot(np.ndarray, np.ndarray)
    def get_data(self, time, position):
        
        self.focusCurve.setData(time, position)
             
        if self.feedbackLoopBox.isChecked():
            
            if len(position) > 2:
        
                zMean = np.mean(position)
                zStDev = np.std(position)
                
                self.focusMean.setValue(zMean)
                self.focusStDev0.setValue(zMean - zStDev)
                self.focusStDev1.setValue(zMean + zStDev)
      
    @pyqtSlot(float)          
    def get_setpoint(self, value):
        
        self.setPoint = value
        
        self.focusSetPoint = self.focusGraph.zPlot.addLine(y=self.setPoint,
                                                           pen=pg.mkPen('r', width=2))
        self.focusMean = self.focusGraph.zPlot.addLine(y=self.setPoint,
                                                       pen='c')
        self.focusStDev0 = self.focusGraph.zPlot.addLine(y=self.setPoint,
                                                         pen='c')
        
        self.focusStDev1 = self.focusGraph.zPlot.addLine(y=self.setPoint,
                                                         pen='c')
        
    def clear_graph(self):
        
        self.focusGraph.zPlot.removeItem(self.focusSetPoint)
        self.focusGraph.zPlot.removeItem(self.focusMean)
        self.focusGraph.zPlot.removeItem(self.focusStDev0)
        self.focusGraph.zPlot.removeItem(self.focusStDev1)
            
    def make_connection(self, backend):
            
        backend.changedImage.connect(self.get_image)
        backend.changedData.connect(self.get_data)
        backend.changedSetPoint.connect(self.get_setpoint)
        
    def setup_gui(self):
        
         # Focus lock widget
         
        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.setMinimumSize(2, 200)
        
        # LiveView Button

        self.liveviewButton = QtGui.QPushButton('Camera LIVEVIEW')
        self.liveviewButton.setCheckable(True)

        # turn ON/OFF feedback loop
        
        self.feedbackLoopBox = QtGui.QCheckBox('Feedback loop')
        
        # ROI button

        self.ROIbutton = QtGui.QPushButton('ROI')
        self.selectROIbutton = QtGui.QPushButton('Select ROI')
        self.calibrationButton = QtGui.QPushButton('Calibrate')
        
        self.exportDataButton = QtGui.QPushButton('Export data')
        self.saveDataBox = QtGui.QCheckBox("Save data")
                
        self.clearDataButton = QtGui.QPushButton('Clear data')
        
        self.ROIbutton.clicked.connect(self.ROImethod)

        self.focusPropertiesDisplay = QtGui.QLabel(' st_dev = 0  max_dev = 0')
        
        # gui connections
        
        self.liveviewButton.clicked.connect(self.toggle_liveview)
        self.saveDataBox.stateChanged.connect(self.emit_save_data_state)
        self.selectROIbutton.clicked.connect(self.selectROI)
        self.clearDataButton.clicked.connect(self.clear_graph)

        # focus camera display
        
        self.camDisplay = pg.GraphicsLayoutWidget()
        self.camDisplay.setMinimumHeight(200)
        self.camDisplay.setMinimumWidth(200)
        
        self.vb = self.camDisplay.addViewBox(row=0, col=0)
        self.vb.setAspectLocked(True)
        self.vb.setMouseMode(pg.ViewBox.RectMode)
        self.img = pg.ImageItem()
        self.img.translate(-0.5, -0.5)
        self.vb.addItem(self.img)

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
 
        
#        self.focusSetPoint = self.focusGraph.plot.addLine(y=self.setPoint, pen='r')



        # GUI layout
        
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        
        # parameters widget
        
        self.paramWidget = QtGui.QFrame()
        self.paramWidget.setFrameStyle(QtGui.QFrame.Panel |
                                       QtGui.QFrame.Raised)
        
        self.paramWidget.setFixedHeight(230)
        self.paramWidget.setFixedWidth(150)
        
        subgrid = QtGui.QGridLayout()
        self.paramWidget.setLayout(subgrid)
        
        subgrid.addWidget(self.calibrationButton, 6, 0)
        subgrid.addWidget(self.exportDataButton, 4, 0)
        subgrid.addWidget(self.clearDataButton, 5, 0)
        
        subgrid.addWidget(self.feedbackLoopBox, 7, 0)
        subgrid.addWidget(self.saveDataBox, 8, 0)
        
        subgrid.addWidget(self.liveviewButton, 1, 0)
        subgrid.addWidget(self.ROIbutton, 2, 0)
        subgrid.addWidget(self.selectROIbutton, 3, 0)
        
        grid.addWidget(self.paramWidget, 0, 0)
        grid.addWidget(self.focusGraph, 0, 1)
        grid.addWidget(self.camDisplay, 0, 2)

        
    def closeEvent(self, *args, **kwargs):
        
        self.closeSignal.emit()
        
        super().closeEvent(*args, **kwargs)
        
        
class Backend(QtCore.QObject):
    
    changedImage = pyqtSignal(np.ndarray)
    changedData = pyqtSignal(np.ndarray, np.ndarray)
    changedSetPoint = pyqtSignal(float)
    
    
    ZdriftCorrectionIsDone = pyqtSignal(float, float, float)
    ZtcspcIsDone = pyqtSignal()

    def __init__(self, camera, adw, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.camera = camera
        self.adw = adw
        self.feedback_active = False
        self.cropped = False
        self.standAlone = False
        
        today = str(date.today()).replace('-', '') # TO DO: change to get folder from microscope
        root = r'C:\\Data\\'
        folder = root + today
        
        filename = r'zdata.txt'
        self.filename = os.path.join(folder, filename)
        
        self.save_data_state = False
    
        self.npoints = 400
        
        # checks image size
        
        rawimage = self.camera.latest_frame()
        image = np.sum(rawimage, axis=2)
        
        self.pxSize = 10   # in nm, TO DO: change for an input from the user
        
        self.sensorSize = np.array(image.shape)
        self.focusSignal = 0
        
        # set focus update rate
        
        self.scansPerS = 20

        self.focusTime = 1000 / self.scansPerS
        self.focusTimer = QtCore.QTimer()

#        self.currentZ = tools.convert(self.actuator.Get_FPar(72), 'UtoX')
#        self.currentX = tools.convert(self.actuator.Get_FPar(70), 'UtoX')
#        self.currentY = tools.convert(self.actuator.Get_FPar(71), 'UtoX')
        
        self.reset()
        self.reset_data_arrays()
        
    def set_actuator_param(self, pixeltime=1000):

        self.adw.Set_FPar(36, tools.timeToADwin(pixeltime))
        
        # set-up actuator initial param
    
        z_f = tools.convert(10, 'XtoU') # TO DO: make this more robust
        
        self.adw.Set_FPar(32, z_f)

        self.adw.Set_Par(30, 1)
        
    def actuator_z(self, z_f):
        
        z_f = tools.convert(z_f, 'XtoU')
          
        self.adw.Set_FPar(32, z_f)
        
        self.adw.Set_Par(30, 1)
        
    @pyqtSlot(bool)
    def liveview(self, value):

        if value:
            self.liveview_start()

        else:
            self.liveview_stop()

        
    def liveview_start(self):
        
        try:
            self.camera.stop_live_video()
        except:
            pass
           
        self.camera.start_live_video(framerate='20 Hz')

        self.focusTimer.start(self.focusTime)

    def liveview_stop(self):
        
        self.focusTimer.stop()
        
        x0 = 0
        y0 = 0
        x1 = 1280 
        y1 = 1024 
            
        val = np.array([x0, y0, x1, y1])
        self.camera._set_AOI(*val)
        
    @pyqtSlot(bool)
    def toggle_feedback(self, val):
        ''' Toggles ON/OFF feedback for either continous (TCSPC) 
        or discrete (confocal imaging) correction'''
        
        if val is True:
            
            self.reset()
            self.setup_feedback()
            self.update()
            self.feedback_active = True
            
            # set up and start actuator process
            
            self.set_actuator_param()
            self.adw.Start_Process(3)
            
            print('Feedback loop ON')
            
        if val is False:
            
            self.feedback_active = False
            print('Feedback loop OFF')
    
    @pyqtSlot()    
    def setup_feedback(self):
        
        ''' set up on/off feedback loop'''
        
        print('set point is', self.focusSignal)
        self.setPoint = self.focusSignal * self.pxSize
        initialZ = tools.convert(self.adw.Get_FPar(72), 'UtoX') # current Z position of the piezo
        self.currentZ = initialZ # set initialZ as currentZ
        
        self.changedSetPoint.emit(self.focusSignal)

        # TO DO: implement calibrated version of this
    
    def update_feedback(self):
        
        dz = self.focusSignal * self.pxSize - self.setPoint

#        print('dz', dz, ' nm')
        
        threshold = 7 # in nm
        far_threshold = 20 # in nm
        correct_factor = 1
        security_thr = 200 # in nm
        
        if np.abs(dz) > threshold:
            
            if np.abs(dz) < far_threshold:
                
                dz = correct_factor * dz
    
        if np.abs(dz) > security_thr:
            
            print('Correction movement larger than 200 nm, active correction turned OFF')
            
        else:
            
            self.currentZ = self.currentZ + dz/1000  # conversion to µm
            self.actuator_z(self.currentZ)
            
    def update_graph_data(self):
        
        ''' update of the data displayed in the gui graph '''

        if self.ptr < self.npoints:
            self.data[self.ptr] = self.focusSignal
            self.time[self.ptr] = self.currentTime
            
            self.changedData.emit(self.time[0:self.ptr + 1],
                                  self.data[0:self.ptr + 1])

        else:
            self.data[:-1] = self.data[1:]
            self.data[-1] = self.focusSignal
            self.time[:-1] = self.time[1:]
            self.time[-1] = self.currentTime

            self.changedData.emit(self.time, self.data)

        self.ptr += 1
            
    def update_stats(self):
        
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
        
    def update(self):
        
        self.acquire_data()
        self.update_graph_data()
        
        #  if locked, correct position
        
        if self.feedback_active:
            
#            self.updateStats()
            self.update_feedback()
            
        if self.save_data_state:
                        
            self.time_array.append(self.currentTime)
            self.z_array.append(self.focusSignal)
            
    def acquire_data(self):
        
        # acquire image
    
        raw_image = self.camera.latest_frame()
        image = np.sum(raw_image, axis=2)   # sum the R, G, B images

        # send image to gui

        self.changedImage.emit(image)
                
        # get mass center
                
        self.massCenter = np.array(ndi.measurements.center_of_mass(image))
        self.focusSignal = self.massCenter[0]
        self.currentTime = ptime.time() - self.startTime
        
#    @pyqtSlot()
#    def setup_discrete_feedback_loop(self):
#        
#        self.setup_feedback()
        
    @pyqtSlot(float, float, float)
    def confocal_drift_correction(self, x, y, z):
        
        initialPos = np.array([x, y, z])
        
#        self.camera.start_live_video(framerate='20 Hz')
#        time.sleep(0.050)
        
        self.acquire_data()
        self.update_graph_data()
        self.update_feedback()
        
        initialPos[2] = self.currentZ
        self.ZdriftCorrectionIsDone.emit(*initialPos)
        
        if self.save_data_state:
            
            self.time_array.append(self.time[-1])
            self.z_array.append(self.data[-1])

    def calibrate(self):
        
        # TO DO: fix calibration function
        
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
        
    def reset_data_arrays(self):
        
        self.time_array = []
        self.z_array = []
        
    def export_data(self):
        
        fname = self.filename
        filename = tools.getUniqueName(fname)
        filename = filename + '_zdata.txt'

        size = np.size(self.z_array)
        savedData = np.zeros((2, size))

        savedData[0, :] = np.array(self.time_array)
        savedData[1, :] = np.array(self.z_array)
        
        np.savetxt(filename, savedData.T, header='t (s), z (px)')
        
        print('z data exported to', filename)
        
    @pyqtSlot(bool)
    def get_lock(self, lockbool):
        self.lock_focus(lockbool)
            
    @pyqtSlot(np.ndarray)
    def get_newROI(self, val):

        self.cropped = True
        self.camera._set_AOI(*val)
        print('focus lock ROI changed to', self.camera._get_AOI())
        
    @pyqtSlot(bool, str)   
    def get_tcspc_signal(self, val, fname):
        
        """ 
        Get signal to start/stop xy position tracking and lock during 
        tcspc acquisition. It also gets the name of the tcspc file to produce
        the corresponding xy_data file
        
        bool val
        True: starts the tracking and feedback loop
        False: stops saving the data and exports the data during tcspc measurement
        tracking and feedback are not stopped automatically 
        
        """
        
        self.filename = fname
         
        if val is True:
            
            self.reset()
            self.reset_data_arrays()
            
            self.save_data_state = True
            self.toggle_feedback(True)
            self.save_data_state = True
            
        else:
            
            self.export_data()
            self.save_data_state = False
            
        # TO DO: fix updateGUIcheckboxSignal    
        
#        self.updateGUIcheckboxSignal.emit(self.tracking_value, 
#                                          self.feedback_active, 
#                                          self.save_data_state)
            
    @pyqtSlot(bool, str)   
    def get_scan_signal(self, val, fname):
        
        """ 
        Get signal to stop continous xy tracking/feedback if active and to
        go to discrete xy tracking/feedback mode if required
        """
        
    @pyqtSlot(bool)
    def get_save_data_state(self, val):
        
        self.save_data_state = val
        print('save_data_state = {}'.format(val))
        
    def make_connection(self, frontend):
          
        frontend.liveviewSignal.connect(self.liveview)
        frontend.changedROI.connect(self.get_newROI)
        frontend.closeSignal.connect(self.stop)
#        frontend.lockFocusSignal.connect(self.lock_focus)
        frontend.feedbackLoopBox.stateChanged.connect(lambda: self.toggle_feedback(frontend.feedbackLoopBox.isChecked()))
        frontend.saveDataSignal.connect(self.get_save_data_state)
        frontend.exportDataButton.clicked.connect(self.export_data)
        frontend.clearDataButton.clicked.connect(self.reset)
        frontend.calibrationButton.clicked.connect(self.calibrate)
        
    def set_aux_moveTo_param(self, x_f, y_f, z_f, n_pixels_x=128, n_pixels_y=128,
                         n_pixels_z=128, pixeltime=2000):

        x_f = tools.convert(x_f, 'XtoU')
        y_f = tools.convert(y_f, 'XtoU')
        z_f = tools.convert(z_f, 'XtoU')

        self.adw.Set_Par(21, n_pixels_x)
        self.adw.Set_Par(22, n_pixels_y)
        self.adw.Set_Par(23, n_pixels_z)

        self.adw.Set_FPar(23, x_f)
        self.adw.Set_FPar(24, y_f)
        self.adw.Set_FPar(25, z_f)

        self.adw.Set_FPar(26, tools.timeToADwin(pixeltime))

    def aux_moveTo(self, x_f, y_f, z_f): # TO DO: delete this two functions, only useful for the initial and final movement in stand-alone mode

        self.set_aux_moveTo_param(x_f, y_f, z_f)
        self.adw.Start_Process(2)
        
    def stop(self):
        
        self.focusTimer.stop()
        self.camera.close()
        
        if self.standAlone is True:
            
            # Go back to 0 position
    
            x_0 = 0
            y_0 = 0
            z_0 = 0
    
            self.aux_moveTo(x_0, y_0, z_0)
            
        print('Focus lock stopped')
        
        # clean up aux files from NiceLib
        
        os.remove(r'C:\Users\USUARIO\Documents\GitHub\pyflux\lextab.py')
        os.remove(r'C:\Users\USUARIO\Documents\GitHub\pyflux\yacctab.py')
            
    

if __name__ == '__main__':
    
    app = QtGui.QApplication([])
    app.setStyle(QtGui.QStyleFactory.create('fusion'))
#    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    
    print('Focus lock module running in stand-alone mode')
    
    # initialize devices
    
    cam = uc480.UC480_Camera()
    
    DEVICENUMBER = 0x1
    adw = ADwin.ADwin(DEVICENUMBER, 1)
    scan.setupDevice(adw)
    
    gui = Frontend()   
    worker = Backend(cam, adw)
    worker.standAlone = True
    
    worker.make_connection(gui)
    gui.make_connection(worker)
    
    focusThread = QtCore.QThread()
    worker.moveToThread(focusThread)
    worker.focusTimer.moveToThread(focusThread)
    worker.focusTimer.timeout.connect(worker.update)
    
    focusThread.start()
    
    # initialize fpar_70, fpar_71, fpar_72 ADwin position parameters
        
    pos_zero = tools.convert(0, 'XtoU')
        
    worker.adw.Set_FPar(70, pos_zero)
    worker.adw.Set_FPar(71, pos_zero)
    worker.adw.Set_FPar(72, pos_zero)
    
    worker.aux_moveTo(10, 10, 10) # in µm
    
    gui.setWindowTitle('Focus lock')
    gui.resize(1500, 500)

    gui.show()
    app.exec_()
        