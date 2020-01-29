# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 11:59:13 2019

@author: Luciano A. Masullo
"""

import numpy as np
import time
import ctypes as ct
from datetime import date, datetime

import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import Dock, DockArea
import pyqtgraph.ptime as ptime
from scipy import optimize as opt
from PIL import Image

import tools.viewbox_tools as viewbox_tools
import tools.colormaps as cmaps
import tools.PSF as PSF
import tools.tools as tools
import scan

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QThread
import qdarkstyle

from lantz.drivers.andor import ccd 
import drivers.ADwin as ADwin

DEBUG = True

PX_SIZE = 80.0 #px size of camera in nm

class Frontend(QtGui.QFrame):
    
    roiInfoSignal = pyqtSignal(int, np.ndarray)
    closeSignal = pyqtSignal()
    saveDataSignal = pyqtSignal(bool)
    
    """
    Signals
             
    - roiInfoSignal:
         To: [backend] get_roi_info
        
    - closeSignal:
         To: [backend] stop
        
    - saveDataSignal:
         To: [backend] get_save_data_state
        
    """
    
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        self.setup_gui()
        
        # initial ROI parameters        
        
        self.NofPixels = 200
        self.roi = None
        self.ROInumber = 0
        self.roilist = []

    def craete_roi(self):
        
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
    
    def emit_roi_info(self):
        
#        print('Set ROIs function')
        
        roinumber = len(self.roilist)
        
        if roinumber == 0:
            
            print(datetime.now(), '[xy_tracking] Please select a valid ROI for beads tracking')
            
        else:
            
            coordinates = np.zeros((4))
            
            for i in range(len(self.roilist)):
                
    #            print(self.roilist[i].pos())
    #            print(self.roilist[i].size())
                xmin, ymin = self.roilist[i].pos()
                xmax, ymax = self.roilist[i].pos() + self.roilist[i].size()
        
                coordinates = np.array([xmin, xmax, ymin, ymax])  
#            roicoordinates[i] = coordinates
                
#            self.roiInfoSignal.emit(roinumber, roicoordinates)
            self.roiInfoSignal.emit(roinumber, coordinates)

    def delete_roi(self):
        
        for i in range(len(self.roilist)):
            
            self.vb.removeItem(self.roilist[i])
            self.roilist[i].hide()
            
        self.roilist = []
        self.delete_roiButton.setChecked(False)
        self.ROInumber = 0
     
    @pyqtSlot(bool)
    def toggle_liveview(self, on):
        
        if on:
            self.liveviewButton.setChecked(True)
            print(datetime.now(), '[xy_tracking] Live view started')
        else:
            self.liveviewButton.setChecked(False)
            self.emit_roi_info()
            self.img.setImage(np.zeros((512,512)), autoLevels=False)
            print(datetime.now(), '[xy_tracking] Live view stopped')
        
    @pyqtSlot()  
    def get_roi_request(self):
        
        print(datetime.now(), '[xy_tracking] got ROI request')
        
        self.emit_roi_info()
        
    @pyqtSlot(np.ndarray)
    def get_image(self, img):
        
#        if DEBUG:
#            print(datetime.now(),'[xy_tracking-frontend] got image signal')

        self.img.setImage(img, autoLevels=False)
        
        self.xaxis.setScale(scale=PX_SIZE/1000) #scale to µm
        self.yaxis.setScale(scale=PX_SIZE/1000) #scale to µm
        
        
    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def get_data(self, time, xData, yData):
        
        self.xCurve.setData(time, xData)
        self.yCurve.setData(time, yData)
        
        self.xyDataItem.setData(xData, yData)
        
        if len(xData) > 2:
            
            self.plot_ellipse(xData, yData)
        
    def plot_ellipse(self, x_array, y_array):
        
        pass
        
#            cov = np.cov(x_array, y_array)
#            
#            a, b, theta = tools.cov_ellipse(cov, q=.683)
#            
#            theta = theta + np.pi/2            
##            print(a, b, theta)
#            
#            xmean = np.mean(xData)
#            ymean = np.mean(yData)
#            
#            t = np.linspace(0, 2 * np.pi, 1000)
#            
#            c, s = np.cos(theta), np.sin(theta)
#            R = np.array(((c, -s), (s, c)))
#            
#            coord = np.array([a * np.cos(t), b * np.sin(t)])
#            
#            coord_rot = np.dot(R, coord)
#            
#            x = coord_rot[0] + xmean
#            y = coord_rot[1] + ymean
            
            # TO DO: fix plot of ellipse
            
#            self.xyDataEllipse.setData(x, y)
#            self.xyDataMean.setData([xmean], [ymean])

    @pyqtSlot(int, bool)    
    def update_shutter(self, num, on):
        
        '''
        setting of num-value:
            0 - signal send by scan-gui-button --> change state of all minflux shutters
            1...6 - shutter 1-6 will be set according to on-variable, i.e. either true or false; only 1-4 controlled from here
            7 - set all minflux shutters according to on-variable
            8 - set all shutters according to on-variable
        for handling of shutters 1-5 see [scan] and [focus]
        '''
        
        if (num == 6)  or (num == 8):
            self.shutterCheckbox.setChecked(on)
                    
    @pyqtSlot(bool, bool, bool)
    def get_backend_states(self, tracking, feedback, savedata):
        
#        print(datetime.now(), '[xy_tracking] Got backend states')
        
        if tracking is True:
            
            self.trackingBeadsBox.setChecked(True)
        
        if tracking is False:
            
            self.trackingBeadsBox.setChecked(False)
            
        if feedback is True:
            
            self.feedbackLoopBox.setChecked(True)
            
        if feedback is False:
            
            self.feedbackLoopBox.setChecked(False)
            
        if savedata is True:
            
            self.saveDataBox.setChecked(True)
            
        if savedata is False:
            
            self.saveDataBox.setChecked(False)

    def emit_save_data_state(self):
        
        if self.saveDataBox.isChecked():
            
            self.saveDataSignal.emit(True)
            self.emit_roi_info()
            
        else:
            
            self.saveDataSignal.emit(False)
        
    def make_connection(self, backend):
            
        backend.changedImage.connect(self.get_image)
        backend.changedData.connect(self.get_data)
        backend.updateGUIcheckboxSignal.connect(self.get_backend_states)
        backend.shuttermodeSignal.connect(self.update_shutter)
        backend.liveviewSignal.connect(self.toggle_liveview)
        
    def setup_gui(self):
        
        # GUI layout
        
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        
        # parameters widget
        
        self.paramWidget = QtGui.QFrame()
        self.paramWidget.setFrameStyle(QtGui.QFrame.Panel |
                                       QtGui.QFrame.Raised)
        
        self.paramWidget.setFixedHeight(200)
        self.paramWidget.setFixedWidth(250)
        
        grid.addWidget(self.paramWidget, 0, 1)
        
        # image widget layout
        
        imageWidget = pg.GraphicsLayoutWidget()
        imageWidget.setMinimumHeight(350)
        imageWidget.setMinimumWidth(350)
        
        # setup axis, for scaling see get_image()
        self.xaxis = pg.AxisItem(orientation='bottom', maxTickLength=5)
        self.xaxis.showLabel(show=True)
        self.xaxis.setLabel('x', units='µm')
        
        self.yaxis = pg.AxisItem(orientation='left', maxTickLength=5)
        self.yaxis.showLabel(show=True)
        self.yaxis.setLabel('y', units='µm')
        
        self.vb = imageWidget.addPlot(axisItems={'bottom': self.xaxis, 
                                                 'left': self.yaxis})
    
        self.vb.setAspectLocked(True)
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
    
#        self.xyGraph.resize(200, 300)
        self.xyGraph.setAntialiasing(True)
        
        self.xyGraph.statistics = pg.LabelItem(justify='right')
        self.xyGraph.addItem(self.xyGraph.statistics)
        self.xyGraph.statistics.setText('---')
        
        self.xyGraph.xPlot = self.xyGraph.addPlot(row=1, col=0)
        self.xyGraph.xPlot.setLabels(bottom=('Time', 's'),
                            left=('Y position', 'nm'))   # TO DO: clean-up the x-y mess (they're interchanged)
        self.xyGraph.xPlot.showGrid(x=True, y=True)
        self.xCurve = self.xyGraph.xPlot.plot(pen='b')
        

        
        self.xyGraph.yPlot = self.xyGraph.addPlot(row=0, col=0)
        self.xyGraph.yPlot.setLabels(bottom=('Time', 's'),
                                     left=('X position', 'nm'))
        self.xyGraph.yPlot.showGrid(x=True, y=True)
        self.yCurve = self.xyGraph.yPlot.plot(pen='r')
        
        # xy drift graph (2D point plot)
        
        self.xyPoint = pg.GraphicsWindow()
        self.xyPoint.resize(400, 400)
        self.xyPoint.setAntialiasing(False)
        
#        self.xyPoint.xyPointPlot = self.xyGraph.addPlot(col=1)
#        self.xyPoint.xyPointPlot.showGrid(x=True, y=True)
        
        self.xyplotItem = self.xyPoint.addPlot()
        self.xyplotItem.showGrid(x=True, y=True)
        self.xyplotItem.setLabels(bottom=('X position', 'nm'),
                                  left=('Y position', 'nm'))
        
        self.xyDataItem = self.xyplotItem.plot([], pen=None, symbolBrush=(255,0,0), 
                                               symbolSize=5, symbolPen=None)
        
        self.xyDataMean = self.xyplotItem.plot([], pen=None, symbolBrush=(117, 184, 200), 
                                               symbolSize=5, symbolPen=None)
        
        self.xyDataEllipse = self.xyplotItem.plot(pen=(117, 184, 200))

        
        # LiveView Button

        self.liveviewButton = QtGui.QPushButton('Camera LIVEVIEW')
        self.liveviewButton.setCheckable(True)
        
        # create ROI button
    
        self.ROIButton = QtGui.QPushButton('ROI')
        self.ROIButton.setCheckable(True)
        self.ROIButton.clicked.connect(self.craete_roi)
        
        # select ROI
        
        self.selectROIbutton = QtGui.QPushButton('Select ROI')
        self.selectROIbutton.clicked.connect(self.emit_roi_info)
        
        # delete ROI button
        
        self.delete_roiButton = QtGui.QPushButton('delete ROIs')
        self.delete_roiButton.clicked.connect(self.delete_roi)
        
        # position tracking checkbox
        
        self.exportDataButton = QtGui.QPushButton('export current data')

        # position tracking checkbox
        
        self.trackingBeadsBox = QtGui.QCheckBox('Track beads')
        self.trackingBeadsBox.stateChanged.connect(self.emit_roi_info)
        
        # turn ON/OFF feedback loop
        
        self.feedbackLoopBox = QtGui.QCheckBox('Feedback loop')

        # save data signal
        
        self.saveDataBox = QtGui.QCheckBox("Save data")
        self.saveDataBox.stateChanged.connect(self.emit_save_data_state)
        
        
        # button to clear the data
        
        self.clearDataButton = QtGui.QPushButton('Clear data')
        
        #shutter button and label
        self.shutterLabel = QtGui.QLabel('Shutter open?')
        self.shutterCheckbox = QtGui.QCheckBox('473 nm laser')
        
        # buttons and param layout
        
        subgrid = QtGui.QGridLayout()
        self.paramWidget.setLayout(subgrid)

        subgrid.addWidget(self.liveviewButton, 0, 0)
        subgrid.addWidget(self.ROIButton, 1, 0)
        subgrid.addWidget(self.selectROIbutton, 2, 0)
        subgrid.addWidget(self.delete_roiButton, 3, 0)
        subgrid.addWidget(self.exportDataButton, 4, 0)
        subgrid.addWidget(self.clearDataButton, 5, 0)
        subgrid.addWidget(self.trackingBeadsBox, 1, 1)
        subgrid.addWidget(self.feedbackLoopBox, 2, 1)
        subgrid.addWidget(self.saveDataBox, 3, 1)
        subgrid.addWidget(self.shutterLabel, 7, 0)
        subgrid.addWidget(self.shutterCheckbox, 7, 1)
        
        grid.addWidget(self.xyGraph, 1, 0)
        grid.addWidget(self.xyPoint, 1, 1)
        
        self.liveviewButton.clicked.connect(lambda: self.toggle_liveview(self.liveviewButton.isChecked()))
        
    def closeEvent(self, *args, **kwargs):
        
        self.closeSignal.emit()
        super().closeEvent(*args, **kwargs)
        app.quit()
        
class Backend(QtCore.QObject):
    
    changedImage = pyqtSignal(np.ndarray)
    changedData = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    updateGUIcheckboxSignal = pyqtSignal(bool, bool, bool)

    xyIsDone = pyqtSignal(bool, float, float)  # signal to emit new piezo position after drift correction
    shuttermodeSignal = pyqtSignal(int, bool)
    liveviewSignal = pyqtSignal(bool)
    """
    Signals
    
    - changedImage:
        To: [frontend] get_image
             
    - changedData:
        To: [frontend] get_data
        
    - updateGUIcheckboxSignal:
        To: [frontend] get_backend_states
        
    - xyIsDone:
        To: [psf] get_xy_is_done
        
    - shuttermodeSignal:
        To: [frontend] update_shutter

    """

    def __init__(self, andor, adw, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.andor = andor
        self.adw = adw
        self.initialize_camera()
        self.setup_camera()
        
        # folder
        
        today = str(date.today()).replace('-', '')  # TO DO: change to get folder from microscope
        root = r'C:\\Data\\'
        folder = root + today
        
        filename = r'\xydata'
        self.filename = folder + filename
        
        self.viewtimer = QtCore.QTimer()
        self.viewtimer.timeout.connect(self.update)
        
        self.tracking_value = False
        self.save_data_state = False
        self.feedback_active = False
        self.camON = False

        self.npoints = 1200
        self.buffersize = 30000
        
        self.currentx = 0
        self.currenty = 0
        
        self.reset()
        self.reset_data_arrays()
        
        self.counter = 0
        
    def setup_camera(self):
        
        #self.pxSize = 80  # in nm #moved infront of Frontend to allow access
        self.shape = (512, 512) # TO DO: change to 256 x 256
        self.expTime = 0.300   # in sec
        
        self.andor.set_exposure_time(self.expTime)
        self.andor.set_image(shape=self.shape)
        
        print(datetime.now(), '[xy_tracking] FOV size = {}'.format(self.shape))

        # Temperature

        self.andor.cooler_on = True
        self.andor.temperature_setpoint = -20   # in °C
        
        # Frame transfer mode
        
        self.andor.frame_transfer_mode = True
        print(datetime.now(), '[xy_tracking] Frame transfer mode =', self.andor.frame_transfer_mode)

        # Horizontal readout speed

        ad = 1   # 16-bit DAC
        typ = 0   # EM mode
        index = 0   # 1 MHz
        self.andor.lib.SetHSSpeed(ct.c_int(ad), ct.c_int(typ), ct.c_int(index))
        
        hrate = self.andor.true_horiz_shift_speed(index=0, typ=0, ad=1)
        print(datetime.now(), '[xy_tracking] Horizontal readout rate = {} MHz'.format(hrate.magnitude))
        
        # pre-amp GAIN

        self.andor.preamp = 2  # index 2 for preamp gain = 4.7 
        
        gain = self.andor.true_preamp(2)
        print(datetime.now(), '[xy_tracking] PreAmp gain = {}'.format(np.round(gain, 1)))

        # EM GAIN
        
        self.andor.EM_gain_mode = 'DAC255'
        self.andor.EM_gain = 1  # EM gain set to 100

        print(datetime.now(), '[xy_tracking] EM gain = {}'.format(self.andor.EM_gain))
    
        # Vertical shift speed
        
        self.andor.vert_shift_speed = 4
        
        vspeed = self.andor.true_vert_shift_speed(4)
        print(datetime.now(), '[xy_tracking] Vertical shift speed = {} µs'.format(np.round(vspeed.magnitude,
                                                             1)))
        
    def initialize_camera(self):
        
        cam = 0
        self.andor.current_camera = self.andor.camera_handle(cam)
        self.andor.lib.Initialize()
        print(datetime.now(), '[xy_tracking] idn:', self.andor.idn)

    @pyqtSlot(int, bool)
    def toggle_tracking_shutter(self, num, val):
        #TODO: change code to also update checkboxes in case of minflux measurement
        if (num == 6)  or (num == 8):
            if val:
                tools.toggle_shutter(self.adw, 6, True)
                print(datetime.now(), '[xy_tracking] Tracking shutter opened')
            else:
                tools.toggle_shutter(self.adw, 6, False)
                print(datetime.now(), '[xy_tracking] Tracking shutter closed')
   
    @pyqtSlot(int, bool)
    def shutter_handler(self, num, on):
        self.shuttermodeSignal.emit(num, on)
        
    @pyqtSlot(bool)
    def liveview(self, value):
        
        '''
        Connection: [frontend] liveviewSignal
        Description: toggles start/stop the liveview of the camera.
        
        '''
        if value:
            self.camON = True
            self.liveview_start()
            
        else:
            self.liveview_stop()
            self.camON = False

        
    def liveview_start(self):
        
        self.initial = True
        
        print(datetime.now(), '[xy_tracking] Temperature = {} °C'.format(self.andor.temperature))
        print(datetime.now(), '[xy_tracking] Andor temperature status:', self.andor.temperature_status)
        
        # Initial image
        
        self.andor.acquisition_mode = 'Run till abort'
        print(datetime.now(), '[xy_tracking] Acquisition mode:', self.andor.acquisition_mode)
        self.andor.shutter(0, 1, 0, 0, 0)
        self.andor.start_acquisition()
        
        time.sleep(self.expTime * 2)
          
        self.image = self.andor.most_recent_image16(self.shape)

        self.changedImage.emit(self.image)

        self.viewtimer.start(400) # DON'T USE time.sleep() inside the update()
                                  # 400 ms ~ acq time + gaussian fit time
    
    def liveview_stop(self):
        
        self.viewtimer.stop()
        
        self.andor.abort_acquisition()
            
#        self.andor.shutter(0, 2, 0, 0, 0)  # TO DO: implement toggle shutter
                    
    def update(self):
        """ General update method """
        
#        print(datetime.now(), '[xy_tracking] entered update')
        
        self.update_view()

        if self.tracking_value:
                
            self.track()
            self.update_graph_data()
            
            if self.feedback_active:
                    
                self.correct()
                         
        self.counter += 1  # counter to check how many times this function is executed

            
    def update_view(self):
        """ Image update while in Liveview mode """

        self.image = self.andor.most_recent_image16(self.shape)
        self.changedImage.emit(self.image)
            
    def update_graph_data(self):
        """ Update the data displayed in the graphs """
        
        if self.ptr < self.npoints:
            self.xData[self.ptr] = self.x
            self.yData[self.ptr] = self.y
            self.time[self.ptr] = self.currentTime
            
            self.changedData.emit(self.time[0:self.ptr + 1],
                                  self.xData[0:self.ptr + 1],
                                  self.yData[0:self.ptr + 1])

        else:
            self.xData[:-1] = self.xData[1:]
            self.xData[-1] = self.x
            self.yData[:-1] = self.yData[1:]
            self.yData[-1] = self.y
            self.time[:-1] = self.time[1:]
            self.time[-1] = self.currentTime
            
            self.changedData.emit(self.time, self.xData, self.yData)

        self.ptr += 1
    
    @pyqtSlot(bool)
    def toggle_tracking(self, val):
        
        '''
        Connection: [frontend] trackingBeadsBox.stateChanged
        Description: toggles ON/OFF tracking of fiducial fluorescent beads. 
        Drift correction feedback loop is not automatically started.
        
        '''

        
        self.startTime = time.time()
        
        if val is True:
            
            self.reset()
            self.reset_data_arrays()
            
            self.tracking_value = True
            self.counter = 0
                    
        if val is False:
        
            self.tracking_value = False
            
    @pyqtSlot(bool)
    def toggle_feedback(self, val, mode='continous'):
        ''' 
        Connection: [frontend] feedbackLoopBox.stateChanged
        Description: toggles ON/OFF feedback for either continous (TCSPC) 
        or discrete (scan imaging) correction
        '''
        
        if val is True:
            
            self.feedback_active = True

            # set up and start actuator process
            
            if mode == 'continous':
            
                self.set_actuator_param()
                self.adw.Start_Process(4)
            
            if DEBUG:
                print(datetime.now(), '[xy_tracking] Feedback loop ON')
            
        if val is False:
            
            self.feedback_active = False
            
            if mode == 'continous':

#                self.adw.Stop_Process(4)
                print(datetime.now(), '[xy_tracking] Process 4 stopped')
            
            if DEBUG:
                print(datetime.now(), '[xy_tracking] Feedback loop OFF')
            
        self.updateGUIcheckboxSignal.emit(self.tracking_value, 
                                          self.feedback_active, 
                                          self.save_data_state)
            
    def gaussian_fit(self):
        
        # set main reference frame
        
        xmin, xmax, ymin, ymax = self.ROIcoordinates
        xmin_nm, xmax_nm, ymin_nm, ymax_nm = self.ROIcoordinates * PX_SIZE
        
        # select the data of the image corresponding to the ROI

        array = self.image[xmin:xmax, ymin:ymax]
        
        # set new reference frame
        
        xrange_nm = xmax_nm - xmin_nm
        yrange_nm = ymax_nm - ymin_nm
             
        x_nm = np.arange(0, xrange_nm, PX_SIZE)
        y_nm = np.arange(0, yrange_nm, PX_SIZE)
        
        (Mx_nm, My_nm) = np.meshgrid(x_nm, y_nm)
        
        # find max 
        
        argmax = np.unravel_index(np.argmax(array, axis=None), array.shape)
        
        x_center_id = argmax[0]
        y_center_id = argmax[1]
        
        # define area around maximum
    
        xrange = 10 # in px
        yrange = 10 # in px
        
        xmin_id = int(x_center_id-xrange)
        xmax_id = int(x_center_id+xrange)
        
        ymin_id = int(y_center_id-yrange)
        ymax_id = int(y_center_id+yrange)
        
        array_sub = array[xmin_id:xmax_id, ymin_id:ymax_id]
                
        xsubsize = 2 * xrange
        ysubsize = 2 * yrange
        
#        plt.imshow(array_sub, cmap=cmaps.parula, interpolation='None')
        
        x_sub_nm = np.arange(0, xsubsize) * PX_SIZE
        y_sub_nm = np.arange(0, ysubsize) * PX_SIZE

        [Mx_sub, My_sub] = np.meshgrid(x_sub_nm, y_sub_nm)
        
        # make initial guess for parameters
        
        bkg = np.min(array)
        A = np.max(array) - bkg
        σ = 130 # nm
        x0 = x_sub_nm[int(xsubsize/2)]
        y0 = y_sub_nm[int(ysubsize/2)]
        
        initial_guess_G = [A, x0, y0, σ, σ, bkg]
         
        poptG, pcovG = opt.curve_fit(PSF.gaussian2D, (Mx_sub, My_sub), 
                                     array_sub.ravel(), p0=initial_guess_G)
        
        # retrieve results

        poptG = np.around(poptG, 2)
    
        A, x0, y0, σ_x, σ_y, bkg = poptG
        
        x = x0 + Mx_nm[xmin_id, ymin_id]
        y = y0 + My_nm[xmin_id, ymin_id]
        
#        self.currentx = x
#        self.currenty = y
        
        # if to avoid (probably) false localizations
        
        maxdist = 200 # in nm
        
        if self.initial is False:
        
            if np.abs(x - self.currentx) < maxdist and np.abs(y - self.currenty) < maxdist:
        
                self.currentx = x
                self.currenty = y
                
#                print(datetime.now(), '[xy_tracking] normal')
                
            else:
                
                pass
                
                print(datetime.now(), '[xy_tracking] max dist exceeded')
        
        else:
            
            self.currentx = x
            self.currenty = y
            
#            print(datetime.now(), '[xy_tracking] else')
        
            
    def track(self):
        
        """ 
        Function to track fiducial markers (fluorescent beads) from the selected ROI.
        The position of the beads is calculated through a guassian fit. 
        If feedback_active = True it also corrects for drifts in xy
        If save_data_state = True it saves the xy data
        
        """
        
        try:
            self.gaussian_fit()
            
        except(RuntimeError, ValueError):
            
            print(datetime.now(), '[xy_tracking] Gaussian fit did not work')
            self.toggle_feedback(False)
               
        if self.initial is True:
            
            self.initialx = self.currentx
            self.initialy = self.currenty
            
            self.initial = False
            
        self.x = self.currentx - self.initialx
        self.y = self.currenty - self.initialy
        
        self.currentTime = time.time() - self.startTime
        
        if self.save_data_state:
            
            self.time_array[self.j] = self.currentTime
            self.x_array[self.j] = self.x
            self.y_array[self.j] = self.y
            
            self.j += 1
                        
            if self.j >= (self.buffersize - 5):    # TO DO: -5, arbitrary bad fix
                
                self.export_data()
                self.reset_data_arrays()
                
                print(datetime.now(), '[xy_tracking] Data array, longer than buffer size, data_array reset')
                
    def correct(self, mode='continous'):

        dx = 0
        dy = 0
        threshold = 7
        far_threshold = 15
        correct_factor = 0.6
        security_thr = 0.15 # in µm
        
        if np.abs(self.x) > threshold:
            
            if dx < far_threshold:
                
                dx = correct_factor * dx
            
            dx = - (self.x)/1000 # conversion to µm

#                print('dx', dx)
            
        if np.abs(self.y) > threshold:
            
            if dy < far_threshold:
                
                dy = correct_factor * dy
            
            dy = - (self.y)/1000 # conversion to µm
            
#                print('dy', dy)
    
        if dx > security_thr or dy > security_thr:
            
            print(datetime.now(), '[xy_tracking] Correction movement larger than 200 nm, active correction turned OFF')
            self.toggle_feedback(False)
            
        else:
            
            # compensate for the mismatch between camera/piezo system of reference
            
            theta = np.radians(-3.7)   # 86.3 (or 3.7) is the angle between camera and piezo (measured)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c,-s), (s, c)))
            
            dy, dx = np.dot(R, np.asarray([dx, dy]))
            
            # add correction to piezo position
            
            currentXposition = tools.convert(self.adw.Get_FPar(70), 'UtoX')
            currentYposition = tools.convert(self.adw.Get_FPar(71), 'UtoX')

            targetXposition = currentXposition + dx  
            targetYposition = currentYposition + dy  
            
            if mode == 'continous':
            
                self.actuator_xy(targetXposition, targetYposition)
                
            if mode == 'discrete':
                
#                self.moveTo(targetXposition, targetYposition, 
#                            currentZposition, pixeltime=10)
                
                self.target_x = targetXposition
                self.target_y = targetYposition
            
    @pyqtSlot(bool, bool)
    def single_xy_correction(self, feedback_val, initial): 
        
        """
        From: [psf] xySignal
        Description: Starts acquisition of the camera and makes one single xy
        track and, if feedback_val is True, corrects for the drift
        """
#        if DEBUG:
#            print(datetime.now(), '[xy_tracking] Feedback {}'.format(feedback_val))
        
        if initial:
            self.toggle_feedback(True, mode='discrete')
            self.initial = initial
            print(datetime.now(), '[xy_tracking] initial', initial)
        
        if not self.camON:
            print(datetime.now(), 'liveview started')
            self.camON = True
            self.andor.start_acquisition()
            
        time.sleep(self.expTime * 3)

        self.image = self.andor.most_recent_image16(self.shape)
        self.changedImage.emit(self.image)
            
        self.andor.abort_acquisition()
        self.camON = False
        
        self.track()
        self.update_graph_data()
        self.correct(mode='discrete')
                
        target_x = np.round(self.target_x, 3)
        target_y = np.round(self.target_y, 3)
        
        print(datetime.now(), '[xy_tracking] discrete correction to', 
              target_x, target_y)
    
        self.xyIsDone.emit(True, target_x, target_y)
        
        if DEBUG:
            print(datetime.now(), '[xy_tracking] single xy correction ended')  
        
    def set_actuator_param(self, pixeltime=1000):

        self.adw.Set_FPar(46, tools.timeToADwin(pixeltime))
        
        # set-up actuator initial param
        
        currentXposition = tools.convert(self.adw.Get_FPar(70), 'UtoX')
        currentYposition = tools.convert(self.adw.Get_FPar(71), 'UtoX')
    
        x_f = tools.convert(currentXposition, 'XtoU')
        y_f = tools.convert(currentYposition, 'XtoU')
        
        self.adw.Set_FPar(40, x_f)
        self.adw.Set_FPar(41, y_f)
            
        self.adw.Set_Par(40, 1)
        
    def actuator_xy(self, x_f, y_f):
        
#        print(datetime.now(), '[xy_tracking] actuator x, y =', x_f, y_f)
        
        x_f = tools.convert(x_f, 'XtoU')
        y_f = tools.convert(y_f, 'XtoU')
        
        self.adw.Set_FPar(40, x_f)
        self.adw.Set_FPar(41, y_f)
        
        self.adw.Set_Par(40, 1)    
        
    def set_moveTo_param(self, x_f, y_f, z_f, n_pixels_x=128, n_pixels_y=128,
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

    def moveTo(self, x_f, y_f, z_f, pixeltime=2000): 

        self.set_moveTo_param(x_f, y_f, z_f, pixeltime=2000)
        self.adw.Start_Process(2)
            
    def reset(self):
        
        self.initial = True
        self.xData = np.zeros(self.npoints)
        self.yData = np.zeros(self.npoints)
        self.time = np.zeros(self.npoints)
        self.ptr = 0
        self.startTime = time.time()
        self.j = 0  # iterator on the data array
        
        self.changedData.emit(self.time, self.xData, self.yData)
        
    def reset_data_arrays(self):
        
        self.time_array = np.zeros(self.buffersize, dtype=np.float16)
        self.x_array = np.zeros(self.buffersize, dtype=np.float16)
        self.y_array = np.zeros(self.buffersize, dtype=np.float16)
        
        
    def export_data(self):
        
        """
        Exports the x, y and t data into a .txt file
        """

#        fname = self.filename
##        filename = tools.getUniqueName(fname)    # TO DO: make compatible with psf measurement and stand alone
#        filename = fname + '_xydata.txt'
        
        fname = self.filename
        #case distinction to prevent wrong filenaming when starting minflux or psf measurement
        if fname[0] == '!':
            filename = fname[1:]
        else:
            filename = tools.getUniqueName(fname)
        filename = filename + '_xydata.txt'
        
        size = self.j
        savedData = np.zeros((3, size))

        savedData[0, :] = self.time_array[0:self.j]
        savedData[1, :] = self.x_array[0:self.j]
        savedData[2, :] = self.y_array[0:self.j]
        
        np.savetxt(filename, savedData.T,  header='t (s), x (nm), y(nm)') # transpose for easier loading
        
        print(datetime.now(), '[xy_tracking] xy data exported to', filename)

    @pyqtSlot(bool)    
    def get_stop_signal(self, stoplive):
        
        """
        Connection: [psf] xyStopSignal
        Description: stops liveview, tracking, feedback if they where running to
        start the psf measurement with discrete xy - z corrections
        """
        
        
        self.toggle_feedback(False)
        self.toggle_tracking(False)
        
        self.reset()
        self.reset_data_arrays()
        
        self.save_data_state = True  # TO DO: sync this with GUI checkboxes (Lantz typedfeat?)
                
        if not self.camON:
            self.liveviewSignal.emit(False)
        else:
            self.viewtimer.stop()
    
    @pyqtSlot(bool)
    def get_save_data_state(self, val):
        
        '''
        Connection: [frontend] saveDataSignal
        Description: gets value of the save_data_state variable, True -> save,
        Fals -> don't save
        
        '''
        
        self.save_data_state = val
        
        if DEBUG:
            print(datetime.now(), '[xy_tracking] save_data_state = {}'.format(val))
    
    @pyqtSlot(int, np.ndarray)
    def get_roi_info(self, N, coordinates_array):
        
        '''
        Connection: [frontend] roiInfoSignal
        Description: gets coordinates of the ROI in the GUI
        
        '''
        
        # TO DO: generalize to N ROIs
        
        self.ROIcoordinates = coordinates_array.astype(int)
        
        if DEBUG:
            print(datetime.now(), '[xy_tracking] got ROI coordinates')
     
    @pyqtSlot()    
    def get_lock_signal(self):
        
        '''
        Connection: [minflux] xyzStartSignal
        Description: activates tracking and feedback
        
        '''
        if self.camON:
            self.liveviewSignal.emit(False)
        
        self.toggle_tracking(True)
        self.toggle_feedback(True)
        self.save_data_state = True
        
        self.updateGUIcheckboxSignal.emit(self.tracking_value, 
                                          self.feedback_active, 
                                          self.save_data_state)
        
        if DEBUG:
            print(datetime.now(), '[xy_tracking] System xy locked')

    @pyqtSlot(np.ndarray, np.ndarray) 
    def get_move_signal(self, r, r_rel):            
        
        self.toggle_feedback(False)
#        self.toggle_tracking(True)
        
        self.updateGUIcheckboxSignal.emit(self.tracking_value, 
                                          self.feedback_active, 
                                          self.save_data_state)
        
        x_f, y_f = r

        self.actuator_xy(x_f, y_f)
         
        if DEBUG:
            print(datetime.now(), '[xy_tracking] Moved to', r)
        
#        # Lock again
        
#        print(datetime.now(), '[xy_tracking] initial x and y', self.initialx, self.initialy)
#        print(datetime.now(), '[xy_tracking] dx, dy', r_rel)
##        self.initial = True # to lock at a new position, TO DO: fix relative position tracking
#        self.initialx = self.currentx - r_rel[0] * 1000 # r_rel to nm
#        self.initialy = self.currenty - r_rel[1] * 1000 # r_rel to nm
#        print(datetime.now(), '[xy_tracking] initial x and y', self.initialx, self.initialy)
        
#        self.toggle_feedback(True) # TO DO: fix each position lock
        

        
    @pyqtSlot(str)    
    def get_end_measurement_signal(self, fname):
        
        '''
        From: [minflux] xyzEndSignal or [psf] endSignal
        Description: at the end of the measurement exports the xy data

        '''
        
        self.filename = fname
        self.export_data()
        
        self.toggle_feedback(False) # TO DO: decide wether I want feedback ON/OFF at the end of measurement
        #check
        self.toggle_tracking(False)
        
        self.reset()
        self.reset_data_arrays()
        
        #TODO: check whether actually necessary
        #self.liveview_start()
        
    def make_connection(self, frontend):
            
        frontend.roiInfoSignal.connect(self.get_roi_info)
        frontend.closeSignal.connect(self.stop)
        frontend.saveDataSignal.connect(self.get_save_data_state)
        frontend.exportDataButton.clicked.connect(self.export_data)
        frontend.clearDataButton.clicked.connect(self.reset)
        frontend.clearDataButton.clicked.connect(self.reset_data_arrays)
        frontend.trackingBeadsBox.stateChanged.connect(lambda: self.toggle_tracking(frontend.trackingBeadsBox.isChecked()))
        frontend.shutterCheckbox.stateChanged.connect(lambda: self.toggle_tracking_shutter(8, frontend.shutterCheckbox.isChecked()))
        frontend.liveviewButton.clicked.connect(self.liveview)

#        frontend.feedbackLoopBox.stateChanged.connect(lambda: self.toggle_feedback(frontend.feedbackLoopBox.isChecked()))
        
        # TO DO: clean-up checkbox create continous and discrete feedback loop
        
        # lambda function and gui_###_state are used to toggle both backend
        # states and checkbox status so that they always correspond 
        # (checked <-> active, not checked <-> inactive)
        
    @pyqtSlot()    
    def stop(self):
        
        self.viewtimer.stop()
        
        if self.camON:
            self.andor.abort_acquisition()
                
        self.andor.shutter(0, 2, 0, 0, 0)
            
        self.andor.finalize()
        print(datetime.now(), '[xy_tracking] Andor camera shut down')
        
        # Go back to 0 position

        x_0 = 0
        y_0 = 0
        z_0 = 0

        self.moveTo(x_0, y_0, z_0)
        
        self.toggle_tracking_shutter(8, False)
        

if __name__ == '__main__':

    if not QtGui.QApplication.instance():
        app = QtGui.QApplication([])
    else:
        app = QtGui.QApplication.instance()
        
    #app.setStyle(QtGui.QStyleFactory.create('fusion'))
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    
    andor = ccd.CCD()
    
    DEVICENUMBER = 0x1
    adw = ADwin.ADwin(DEVICENUMBER, 1)
    scan.setupDevice(adw)
    
    gui = Frontend()
    worker = Backend(andor, adw)
    
    gui.make_connection(worker)
    worker.make_connection(gui)
    
    # initialize fpar_70, fpar_71, fpar_72 ADwin position parameters
        
    pos_zero = tools.convert(0, 'XtoU')
        
    worker.adw.Set_FPar(70, pos_zero)
    worker.adw.Set_FPar(71, pos_zero)
    worker.adw.Set_FPar(72, pos_zero)
    
    worker.moveTo(10, 10, 10) # in µm
    
    time.sleep(0.200)
        
    worker.piezoXposition = 10.0 # in µm
    worker.piezoYposition = 10.0 # in µm
    worker.piezoZposition = 10.0 # in µm

    gui.setWindowTitle('xy drift correction')
    gui.show()
    app.exec_()
        