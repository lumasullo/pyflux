# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 11:19:30 2021

@author: Luciano A. Masullo
"""

import numpy as np
import time
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from datetime import date, datetime
import os

from scipy import optimize as opt
from PIL import Image

import tools.viewbox_tools as viewbox_tools
import tools.colormaps as cmaps
import tools.PSF as PSF
import tools.tools as tools

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QThread
from PyQt5.QtWidgets import QGroupBox

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import Dock, DockArea
import pyqtgraph.ptime as ptime
import qdarkstyle # see https://stackoverflow.com/questions/48256772/dark-theme-for-in-qt-widgets

import sys
sys.path.append('C:\Program Files\Thorlabs\Scientific Imaging\ThorCam')
# install from https://instrumental-lib.readthedocs.io/en/stable/install.html

from instrumental.drivers.cameras import uc480
from instrumental import Q_
from drivers import bpc_piezo as bpc



DEBUG = True

PX_SIZE = 35.0 # px size of camera in nm
PX_Z = 15.0 # px size for z in nm

class Frontend(QtGui.QFrame):
    
    roiInfoSignal = pyqtSignal(str, int, np.ndarray)
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

    def craete_roi(self, roi_type):
        
        if roi_type == 'xy':
        
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
            self.xyROIButton.setChecked(False)
            
        if roi_type == 'z':
            
            ROIpen = pg.mkPen(color='y')
            
            ROIpos = (0.5 * self.NofPixels - 64, 0.5 * self.NofPixels - 64)
            self.roi_z = viewbox_tools.ROI2(self.NofPixels/2, self.vb, ROIpos,
                                            handlePos=(1, 0),
                                            handleCenter=(0, 1),
                                            scaleSnap=True,
                                            translateSnap=True,
                                            pen=ROIpen, number=self.ROInumber)
            
            self.zROIButton.setChecked(False)

    def emit_roi_info(self, roi_type):
        
        if roi_type == 'xy':
        
            roinumber = len(self.roilist)
            
            if roinumber == 0:
                
                print(datetime.now(), '[xy_tracking] Please select a valid ROI for fiducial NPs tracking')
                
            else:
                
                coordinates = np.zeros((4))
                
                for i in range(len(self.roilist)):
                    
                    xmin, ymin = self.roilist[i].pos()
                    xmax, ymax = self.roilist[i].pos() + self.roilist[i].size()
            
                    coordinates = np.array([xmin, xmax, ymin, ymax])  
    #            roicoordinates[i] = coordinates
    # TODO: generalize to N rois, now it's only working for one roi (the last one)
                    
    #            self.roiInfoSignal.emit(roinumber, roicoordinates)
                self.roiInfoSignal.emit('xy', roinumber, coordinates)
                    
        if roi_type == 'z':
            
            xmin, ymin = self.roi_z.pos()
            xmax, ymax = self.roi_z.pos() + self.roi_z.size()
            
            coordinates = np.array([xmin, xmax, ymin, ymax]) 
            
            self.roiInfoSignal.emit('z', 0, coordinates)

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
        
    # @pyqtSlot()  
    # def get_roi_request(self):
        
    #     print(datetime.now(), '[xy_tracking] got ROI request')
        
    #     self.emit_roi_info()
        
    @pyqtSlot(np.ndarray)
    def get_image(self, img):
                
        self.img.setImage(img, autoLevels=False)
                
        # self.xaxis.setScale(scale=PX_SIZE/1000) #scale to µm
        # self.yaxis.setScale(scale=PX_SIZE/1000) #scale to µm
        
        
    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    def get_data(self, time, xData, yData, zData):
        
        self.xCurve.setData(time, xData)
        self.yCurve.setData(time, yData)
        self.zCurve.setData(time, zData)
        
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
            
            # TODO: fix plot of ellipse
            
#            self.xyDataEllipse.setData(x, y)
#            self.xyDataMean.setData([xmean], [ymean])

    @pyqtSlot(bool, bool, bool)
    def get_backend_states(self, tracking, feedback, savedata):

        self.trackingBeadsBox.setChecked(tracking)
        self.feedbackLoopBox.setChecked(feedback)
        self.saveDataBox.setChecked(savedata)            

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
        # backend.liveviewSignal.connect(self.toggle_liveview)
        
    def setup_gui(self):
        
        # GUI layout
        
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        
        # parameters widget
        
        self.paramWidget = QGroupBox('XYZ-Tracking parameter')   
        self.paramWidget.setFixedHeight(260)
        self.paramWidget.setFixedWidth(240)
        
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
        # lut = viewbox_tools.generatePgColormap(cmaps.parula)
        # self.hist.gradient.setColorMap(lut)
#        self.hist.vb.setLimits(yMin=800, yMax=3000)

        ## TODO: fix histogram range

        for tick in self.hist.gradient.ticks:
            tick.hide()
        imageWidget.addItem(self.hist, row=0, col=1)
        
        # xy drift graph (graph without a fixed range)
        
        self.xyzGraph = pg.GraphicsWindow()
    
#        self.xyzGraph.resize(200, 300)
        self.xyzGraph.setAntialiasing(True)
        
        self.xyzGraph.statistics = pg.LabelItem(justify='right')
        self.xyzGraph.addItem(self.xyzGraph.statistics)
        self.xyzGraph.statistics.setText('---')
        
        self.xyzGraph.xPlot = self.xyzGraph.addPlot(row=0, col=0)
        self.xyzGraph.xPlot.setLabels(bottom=('Time', 's'),
                                      left=('X position', 'nm'))
        self.xyzGraph.xPlot.showGrid(x=True, y=True)
        self.xCurve = self.xyzGraph.xPlot.plot(pen='b')
        
        self.xyzGraph.yPlot = self.xyzGraph.addPlot(row=1, col=0)
        self.xyzGraph.yPlot.setLabels(bottom=('Time', 's'),
                                     left=('Y position', 'nm'))
        self.xyzGraph.yPlot.showGrid(x=True, y=True)
        self.yCurve = self.xyzGraph.yPlot.plot(pen='r')
        
        self.xyzGraph.zPlot = self.xyzGraph.addPlot(row=2, col=0)
        self.xyzGraph.zPlot.setLabels(bottom=('Time', 's'),
                                     left=('Z position', 'nm'))
        self.xyzGraph.zPlot.showGrid(x=True, y=True)
        self.zCurve = self.xyzGraph.zPlot.plot(pen='y')
        
        # xy drift graph (2D point plot)
        
        self.xyPoint = pg.GraphicsWindow()
        self.xyPoint.resize(400, 400)
        self.xyPoint.setAntialiasing(False)
        
#        self.xyPoint.xyPointPlot = self.xyzGraph.addPlot(col=1)
#        self.xyPoint.xyPointPlot.showGrid(x=True, y=True)
        
        self.xyplotItem = self.xyPoint.addPlot()
        self.xyplotItem.showGrid(x=True, y=True)
        self.xyplotItem.setLabels(bottom=('X position', 'nm'),
                                  left=('Y position', 'nm'))
        self.xyplotItem.setAspectLocked(True)
        
        self.xyDataItem = self.xyplotItem.plot([], pen=None, symbolBrush=(255,0,0), 
                                               symbolSize=5, symbolPen=None)
        
        self.xyDataMean = self.xyplotItem.plot([], pen=None, symbolBrush=(117, 184, 200), 
                                               symbolSize=5, symbolPen=None)
        
        self.xyDataEllipse = self.xyplotItem.plot(pen=(117, 184, 200))

        
        # LiveView Button

        self.liveviewButton = QtGui.QPushButton('Camera LIVEVIEW')
        self.liveviewButton.setCheckable(True)
        
        # create xy ROI button
    
        self.xyROIButton = QtGui.QPushButton('xy ROI')
        self.xyROIButton.setCheckable(True)
        self.xyROIButton.clicked.connect(lambda: self.craete_roi(roi_type='xy'))
        
        # create z ROI button
    
        self.zROIButton = QtGui.QPushButton('z ROI')
        self.zROIButton.setCheckable(True)
        self.zROIButton.clicked.connect(lambda: self.craete_roi(roi_type='z'))
        
        # select xy ROI
        
        self.selectxyROIbutton = QtGui.QPushButton('Select xy ROI')
        self.selectxyROIbutton.clicked.connect(lambda: self.emit_roi_info(roi_type='xy'))
        
        # select z ROI
        
        self.selectzROIbutton = QtGui.QPushButton('Select z ROI')
        self.selectzROIbutton.clicked.connect(lambda: self.emit_roi_info(roi_type='z'))
        
        # delete ROI button
        
        self.delete_roiButton = QtGui.QPushButton('Delete ROIs')
        self.delete_roiButton.clicked.connect(self.delete_roi)
        
        # position tracking checkbox
        
        self.exportDataButton = QtGui.QPushButton('Export current data')

        # position tracking checkbox
        
        self.trackingBeadsBox = QtGui.QCheckBox('Track xy fiducials')
        self.trackingBeadsBox.stateChanged.connect(self.emit_roi_info)
        
        # position tracking checkbox
        
        # self.trackZbeamBox = QtGui.QCheckBox('Track z beam')
        # self.trackZbeamBox.stateChanged.connect(self.emit_roi_info)
        
        # turn ON/OFF feedback loop
        
        self.feedbackLoopBox = QtGui.QCheckBox('Feedback loop')

        # save data signal
        
        self.saveDataBox = QtGui.QCheckBox("Save data")
        self.saveDataBox.stateChanged.connect(self.emit_save_data_state)
        
        
        # button to clear the data
        
        self.clearDataButton = QtGui.QPushButton('Clear data')
        
        # button to reset piezo position
        
        self.resetPiezoPosButton = QtGui.QPushButton('Reset piezo position')
                
        # buttons and param layout
        
        subgrid = QtGui.QGridLayout()
        self.paramWidget.setLayout(subgrid)

        subgrid.addWidget(self.liveviewButton, 0, 0)
        subgrid.addWidget(self.xyROIButton, 1, 0)
        subgrid.addWidget(self.zROIButton, 2, 0)
        subgrid.addWidget(self.selectxyROIbutton, 3, 0)
        subgrid.addWidget(self.selectzROIbutton, 4, 0)
        subgrid.addWidget(self.delete_roiButton, 5, 0)
        subgrid.addWidget(self.exportDataButton, 6, 0)
        subgrid.addWidget(self.clearDataButton, 7, 0)
        subgrid.addWidget(self.resetPiezoPosButton, 8, 0)
        subgrid.addWidget(self.trackingBeadsBox, 1, 1)
        # subgrid.addWidget(self.trackZbeamBox, 2, 1)
        subgrid.addWidget(self.feedbackLoopBox, 2, 1)
        subgrid.addWidget(self.saveDataBox, 3, 1)
        
        grid.addWidget(self.xyzGraph, 1, 0)
        grid.addWidget(self.xyPoint, 1, 1)
        
        self.liveviewButton.clicked.connect(lambda: self.toggle_liveview(self.liveviewButton.isChecked()))
        
    def close_event(self, *args, **kwargs):
        
        self.closeSignal.emit()
        super().close_event(*args, **kwargs)
        app.quit()
        
        
class Backend(QtCore.QObject):
    
    changedImage = pyqtSignal(np.ndarray)
    changedData = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    updateGUIcheckboxSignal = pyqtSignal(bool, bool, bool)

    xyIsDone = pyqtSignal(bool, float, float)  # signal to emit new piezo position after drift correction
    # liveviewSignal = pyqtSignal(bool)
    
    """
    Signals
    
    - changedImage:
        To: [frontend] get_image
             
    - changedData:
        To: [frontend] get_data
        
    - updateGUIcheckboxSignal:
        To: [frontend] get_backend_states
        
    """

    def __init__(self, thorcam, piezo, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.camera = thorcam # no need to setup or initialize camera
        self.camera.master_gain = 4
        self.camera.auto_blacklevel = True
        self.camera.gain_boost = True
        
        self.pz = piezo
        self.pz.connect()
        self.pz.set_zero() # important for internal piezo calibration
                
        # folder
        
        today = str(date.today()).replace('-', '')  # TODO: change to get folder from microscope
        root = r'C:\\Data\\'
        folder = root + today
        
        filename = r'\xydata'
        self.filename = folder + filename
        
        self.view_timer = QtCore.QTimer()
        # self.view_timer.timeout.connect(self.update)
        self.xyz_time = 200 # 200 ms per acquisition + fit + correction
        
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
        self.pattern = False
        
    @pyqtSlot(bool)
    def liveview(self, value):

        if value:
            self.camON = True
            self.liveview_start()

        else:
            self.liveview_stop()
            self.camON = False

        
    def liveview_start(self):
        
        if self.camON:
            self.camera.stop_live_video()
            self.camON = False
        
        self.camON = True
        self.camera.start_live_video()
        self.camera._set_exposure(Q_('50 ms')) # ms


        self.view_timer.start(self.xyz_time)

        
    def liveview_stop(self):
        
        self.view_timer.stop()
        self.camON = False
        
        x0 = 0
        y0 = 0
        x1 = 1280 
        y1 = 1024 
            
        val = np.array([x0, y0, x1, y1])
        # self.get_new_roi(val)
                    
    def update(self):
        """ General update method """
        
#        print(datetime.now(), '[xy_tracking] entered update')
        
        self.update_view()

        if self.tracking_value:
                
            self.track('xy')
            self.track('z')
            self.update_graph_data()
            
            if self.feedback_active:
                    
                self.correct()
                    
        self.counter += 1  # counter to check how many times this function is executed


    def update_view(self):
        """ Image update while in Liveview mode """

        print(self.camera._get_exposure())

        # acquire image
    
        raw_image = self.camera.latest_frame()
        # self.image = np.sum(raw_image, axis=2)   # sum the R, G, B images
        self.image = raw_image[:, :, 0] # take only R channel
                
        # send image to the Frontend
        
        self.changedImage.emit(self.image)
            
    def update_graph_data(self):
        """ Update the data displayed in the graphs """
        
        if self.ptr < self.npoints:
            self.xData[self.ptr] = self.x
            self.yData[self.ptr] = self.y
            self.zData[self.ptr] = self.z
            self.time[self.ptr] = self.currentTime
            
            self.changedData.emit(self.time[0:self.ptr + 1],
                                  self.xData[0:self.ptr + 1],
                                  self.yData[0:self.ptr + 1],
                                  self.zData[0:self.ptr + 1])

        else:
            self.xData[:-1] = self.xData[1:]
            self.xData[-1] = self.x
            self.yData[:-1] = self.yData[1:]
            self.yData[-1] = self.y
            self.zData[:-1] = self.zData[1:]
            self.zData[-1] = self.z
            self.time[:-1] = self.time[1:]
            self.time[-1] = self.currentTime
            
            self.changedData.emit(self.time, self.xData, self.yData, self.zData)

        self.ptr += 1
    
    @pyqtSlot(bool)
    def toggle_tracking(self, val):
        
        '''
        Connection: [frontend] trackingBeadsBox.stateChanged
        Description: toggles ON/OFF tracking of fiducial NPs. 
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
        Description: toggles ON/OFF feedback for continous active correction
        '''
        
        if val is True:
            
            self.feedback_active = True

            if DEBUG:
                print(datetime.now(), '[xy_tracking] Feedback loop ON')
            
            
        if val is False:
            self.feedback_active = False
            
            if DEBUG:
                print(datetime.now(), '[xy_tracking] Feedback loop OFF')
#            
#        self.updateGUIcheckboxSignal.emit(self.tracking_value, 
#                                          self.feedback_active, 
#                                          self.save_data_state)
     
    def center_of_mass(self):
        
        # set main reference frame
        
        xmin, xmax, ymin, ymax = self.zROIcoordinates
        
        # select the data of the image corresponding to the ROI

        zimage = self.image[xmin:xmax, ymin:ymax]
        self.m_center = np.array(ndi.measurements.center_of_mass(zimage))
        
        #
        
        self.currentz = np.sqrt(self.m_center[0]**2 + self.m_center[1]**2)
          
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
    
        xrange = 15 # in px
        yrange = 15 # in px
        
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
        
                self.currentx = x  # self.currentx is the absolute x position
                self.currenty = y
                
                print(datetime.now(), '[xy_tracking] normal')
                
            else:
                                
                print(datetime.now(), '[xy_tracking] max dist exceeded')
        
        else:
            
            self.currentx = x
            self.currenty = y
            
#            print(datetime.now(), '[xy_tracking] else')
        
            
    def track(self, track_type):
        
        """ 
        Function to track fiducial markers (Au NPs) from the selected ROI.
        The position of the NPs is calculated through an xy gaussian fit 
        If feedback_active = True it also corrects for drifts in xy
        If save_data_state = True it saves the xy data
        
        """
        
        if track_type == 'xy':
        
            try:
                self.gaussian_fit()
                
            except(RuntimeError, ValueError):
                
                print(datetime.now(), '[xy_tracking] Gaussian fit did not work')
                self.toggle_feedback(False)
                   
            if self.initial is True:
                
                self.initialx = self.currentx
                self.initialy = self.currenty
                
                self.initial = False
                
            self.x = self.currentx - self.initialx  # self.x is relative to initial pos
            self.y = self.currenty - self.initialy
            
            self.currentTime = time.time() - self.startTime
            
            if self.save_data_state:
                
                self.time_array[self.j] = self.currentTime
                self.x_array[self.j] = self.x 
                self.y_array[self.j] = self.y
                
                self.j += 1
                            
                if self.j >= (self.buffersize - 5):    # TODO: -5, arbitrary bad fix
                    
                    self.export_data()
                    self.reset_data_arrays()
                    
                    print(datetime.now(), '[xy_tracking] Data array, longer than buffer size, data_array reset')
                    
        if track_type == 'z':
            
            self.center_of_mass()
            
            if self.initial_focus is True:
                
                self.initialz = self.currentz
                
                self.initial_focus = False
            
            self.z = (self.currentz - self.initialz) * PX_Z
            
                
    def correct(self, mode='continous'):

        dx = 0
        dy = 0
        dz = 0
        threshold = 5
        far_threshold = 12
        correct_factor = 0.6
        security_thr = 0.35 # in µm
        
        if np.abs(self.x) > threshold:
            
            dx = - (self.x)/1000 # conversion to µm
            
            if dx < far_threshold: #TODO: double check this conditions (do they work?)
                
                dx = correct_factor * dx #TODO: double check this conditions (do they work?)

#                print('dx', dx)
            
        if np.abs(self.y) > threshold:
                        
            dy = - (self.y)/1000 # conversion to µm
            
            if dy < far_threshold:
                
                dy = correct_factor * dy
            
#                print('dy', dy)

        if np.abs(self.z) > threshold:
                        
            dz = - (self.z)/1000 # conversion to µm
            
            if dz < far_threshold:
                
                dz = correct_factor * dz
            
#                print('dy', dy)
    
        if dx > security_thr or dy > security_thr or dz > 2 * security_thr:
            
            print(datetime.now(), '[xy_tracking] Correction movement larger than 200 nm, active correction turned OFF')
            self.toggle_feedback(False)
            
        else:
            
            # compensate for the mismatch between camera/piezo system of reference
            
            # theta = np.radians(-3.7)   # 86.3 (or 3.7) is the angle between camera and piezo (measured)
            # c, s = np.cos(theta), np.sin(theta)
            # R = np.array(((c,-s), (s, c)))
            
            # dy, dx = np.dot(R, np.asarray([dx, dy]))
            
            print('dx, dy', dx, dy)
            
            # # add correction to piezo position
            
            current_piezo_pos = self.pz.get_positions()
            
            currentx_piezo = current_piezo_pos[0] # in µm
            currenty_piezo = current_piezo_pos[1] # in µm
            currentz_piezo = current_piezo_pos[2] # in µm
            
            print('current x, y, z', currentx_piezo, currenty_piezo, currentz_piezo)
            
            targetx_piezo = currentx_piezo - dx # in µm
            targety_piezo = currenty_piezo - dy # in µm
            targetz_piezo = currentz_piezo - dz # in µm
            
            print('target x y z', targetx_piezo, targety_piezo, targetz_piezo)
                        
            self.actuator_xyz(targetx_piezo, targety_piezo, targetz_piezo)
                         
    def actuator_xyz(self, x, y, z=None):
        
        self.pz.set_positions([float(x), float(y), float(z)]) # in µm
        
    def reset_piezo_position(self):
        
        for i in range(10):
            
            self.pz.set_positions([i+1, i+1, i+1]) # go to the middle of the piezo range
            time.sleep(.1)

    def reset(self):
        
        self.initial = True
        self.initial_focus = True
        self.xData = np.zeros(self.npoints)
        self.yData = np.zeros(self.npoints)
        self.zData = np.zeros(self.npoints)
        self.time = np.zeros(self.npoints)
        self.ptr = 0
        self.startTime = time.time()
        self.j = 0  # iterator on the data array
        
        self.changedData.emit(self.time, self.xData, self.yData, self.zData)
        
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
    def get_save_data_state(self, val):
        '''
        Connection: [frontend] saveDataSignal
        Description: gets value of the save_data_state variable, True -> save,
        False -> don't save
        '''

        self.save_data_state = val

        if DEBUG:
            print(datetime.now(), '[xy_tracking] save_data_state = {}'.format(val))

    @pyqtSlot(str, int, np.ndarray)
    def get_roi_info(self, roi_type, N, coordinates_array):
        
        '''
        Connection: [frontend] roiInfoSignal
        Description: gets coordinates of the ROI in the GUI
        
        '''
                
        if roi_type == 'xy':
            
        # TODO: generalize to N ROIs
        
            self.ROIcoordinates = coordinates_array.astype(int)
        
            if DEBUG:
                print(datetime.now(), '[xy_tracking] got ROI coordinates')
                
        if roi_type == 'z':
            
            self.zROIcoordinates = coordinates_array.astype(int)
            
            print(self.zROIcoordinates)
     
    def make_connection(self, frontend):
            
        frontend.roiInfoSignal.connect(self.get_roi_info)
        frontend.closeSignal.connect(self.stop)
        frontend.saveDataSignal.connect(self.get_save_data_state)
        frontend.exportDataButton.clicked.connect(self.export_data)
        frontend.clearDataButton.clicked.connect(self.reset)
        frontend.clearDataButton.clicked.connect(self.reset_data_arrays)
        frontend.resetPiezoPosButton.clicked.connect(self.reset_piezo_position)
        frontend.trackingBeadsBox.stateChanged.connect(lambda: self.toggle_tracking(frontend.trackingBeadsBox.isChecked()))
        frontend.liveviewButton.clicked.connect(self.liveview)
        frontend.feedbackLoopBox.stateChanged.connect(lambda: self.toggle_feedback(frontend.feedbackLoopBox.isChecked()))
        
        # TODO: clean-up checkbox create continous and discrete feedback loop
        
        # lambda function and gui_###_state are used to toggle both backend
        # states and checkbox status so that they always correspond 
        # (checked <-> active, not checked <-> inactive)
        
    @pyqtSlot()    
    def stop(self):
        
        self.view_timer.stop()
        
        if self.camON:
            self.camera.stop_live_video()
            self.camera.close()

        
        # for i in range(10):
            
        #     self.pz.set_positions([9-i, 9-i, 0]) # go to the middle of the piezo range
        #     time.sleep(.1)
            
        self.pz.close()


if __name__ == '__main__':

    if not QtGui.QApplication.instance():
        app = QtGui.QApplication([])
    else:
        app = QtGui.QApplication.instance()
        
    #app.setStyle(QtGui.QStyleFactory.create('fusion'))
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    
    #if camera wasnt closed properly just keep using it without opening new one
    try:
        cam = uc480.UC480_Camera()
    except:
        print('Unable to connect to camera')
    
    try:
        pz = bpc.BenchtopPiezoWrapper(bpc.list_devices()[0])
    except:
        print('Unable to connect to piezo')
    
    print('Model {}'.format(cam.model))
    print('Serial number {}'.format(cam.serial))
    
    print('Serial number {}'.format(bpc.list_devices()))
    
    gui = Frontend()
    worker = Backend(cam, pz)
    
    gui.make_connection(worker)
    worker.make_connection(gui)
    
    xyz_thread = QtCore.QThread()
    worker.moveToThread(xyz_thread)
    worker.view_timer.moveToThread(xyz_thread)
    worker.view_timer.timeout.connect(worker.update)
    
    xyz_thread.start()
    
    time.sleep(0.200)
        
    gui.setWindowTitle('xyz drift correction')
    gui.show()
    app.exec_()