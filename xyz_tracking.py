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

import tifffile

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
VIDEO = False

PX_SIZE = 29.0 # px size of camera in nm
PX_Z = 25.0 # px size for z in nm

# N_NP = 10 # number of AuNP required

class Frontend(QtGui.QFrame):
    
    roiInfoSignal = pyqtSignal(str, int, list)
    z_roiInfoSignal = pyqtSignal(str, int, list)
    closeSignal = pyqtSignal()
    saveDataSignal = pyqtSignal(bool)
    piParamsSignal = pyqtSignal(np.ndarray)
    piezopiParamsSignal = pyqtSignal(np.ndarray)
    
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
        
        # initial ROI parameters        
        
        self.ROInumber = 0
        self.roilist = []
        self.xCurve = None 
        
        self.setup_gui()
        
    def create_roi(self, roi_type):
        
        if roi_type == 'xy':
        
            ROIpen = pg.mkPen(color='r')
    
            ROIpos = (512 - 64, 512 - 64)
            roi = viewbox_tools.ROI2(50, self.vb, ROIpos, handlePos=(1, 0),
                                     handleCenter=(0, 1),
                                     scaleSnap=True,
                                     translateSnap=True,
                                     pen=ROIpen, number=self.ROInumber)
            
            self.ROInumber += 1
            self.roilist.append(roi)
            self.xyROIButton.setChecked(False)
            
        if roi_type == 'z':
            
            ROIpen = pg.mkPen(color='y')
            
            ROIpos = (512 - 64, 512 - 64)
            self.roi_z = viewbox_tools.ROI2(140, self.vb, ROIpos,
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
                coordinates_list = []
                
                for i in range(len(self.roilist)):
                    
                    xmin, ymin = self.roilist[i].pos()
                    xmax, ymax = self.roilist[i].pos() + self.roilist[i].size()
            
                    coordinates = np.array([xmin, xmax, ymin, ymax])  
                    coordinates_list.append(coordinates)
                                                            
                self.roiInfoSignal.emit('xy', roinumber, coordinates_list)
                    
        if roi_type == 'z':
            
            xmin, ymin = self.roi_z.pos()
            xmax, ymax = self.roi_z.pos() + self.roi_z.size()
            
            coordinates = np.array([xmin, xmax, ymin, ymax]) 
            coordinates_list = [coordinates]
            
            self.z_roiInfoSignal.emit('z', 0, coordinates_list)
                        
    def emit_pi_params(self):
        
        self.piParamsSignal.emit(np.array(self.piParamsEdit.text().split(' '),
                                          dtype=np.float64))
        
    def emit_piezopi_params(self):
        
        self.piezopiParamsSignal.emit(np.array(self.piezopiParamsEdit.text().split(' '),
                                               dtype=np.int16))
    def delete_roi(self):
                
        self.vb.removeItem(self.roilist[-1])
        self.roilist[-1].hide()
        self.roilist = self.roilist[:-1]
        self.ROInumber -= 1
     
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
    
    @pyqtSlot(np.ndarray)
    def get_image(self, img):
                
        self.img.setImage(img, autoLevels=False)

        # self.xaxis.setScale(scale=PX_SIZE/1000) #scale to µm
        # self.yaxis.setScale(scale=PX_SIZE/1000) #scale to µm
        
    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    def get_data(self, tData, xData, yData, zData, avgIntData):
        
        # x data
        
        N_NP = np.shape(xData)[1]
        
        for i in range(N_NP):
        
            self.xCurve[i].setData(tData, xData[:, i])
            
        self.xmeanCurve.setData(tData, np.mean(xData, axis=1))
        
        # y data
            
        for i in range(N_NP):
        
            self.yCurve[i].setData(tData, yData[:, i])
            
        self.ymeanCurve.setData(tData, np.mean(yData, axis=1))
        
        # z data
        
        self.zCurve.setData(tData, zData)
        
        # avg intensity data
        
        self.avgIntCurve.setData(avgIntData)
        
        # set xy 2D data
        
        self.xyDataItem.setData(np.mean(xData, axis=1), np.mean(yData, axis=1))
        
        if len(xData) > 2:
            
            self.plot_ellipse(xData, yData)
            
            hist, bin_edges = np.histogram(zData, bins=60)
            self.zHist.setOpts(x=bin_edges[:-1], height=hist)
             
            xstd = np.std(np.mean(xData, axis=1))
            self.xstd_value.setText(str(np.around(xstd, 2)))
            
            ystd = np.std(np.mean(yData, axis=1))
            self.ystd_value.setText(str(np.around(ystd, 2)))
            
            zstd = np.std(zData)
            self.zstd_value.setText(str(np.around(zstd, 2)))
        
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
          
    def emit_save_data_state(self):
        
        if self.saveDataBox.isChecked():
            
            self.saveDataSignal.emit(True)
            
        else:
            
            self.saveDataSignal.emit(False)
        
    def make_connection(self, backend):
            
        backend.changedImage.connect(self.get_image)
        backend.changedData.connect(self.get_data)
        
    def setup_gui(self):
        
        # GUI layout
        
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        
        # parameters widget
        
        self.paramWidget = QGroupBox('XYZ-Tracking parameter')   
        self.paramWidget.setFixedHeight(350)
        self.paramWidget.setFixedWidth(270)
        # self.paramWidget.setFixedWidth(350)

        # stats widget
        
        self.statWidget = QGroupBox('Live statistics')   
        self.statWidget.setFixedHeight(300)
        # self.statWidget.setFixedWidth(240)
        self.statWidget.setFixedWidth(350)

        self.xstd_label = QtGui.QLabel('X std (nm)')
        self.ystd_label = QtGui.QLabel('Y std (nm)')
        self.zstd_label = QtGui.QLabel('Z std (nm)')
        
        self.xstd_value = QtGui.QLabel('0')
        self.ystd_value = QtGui.QLabel('0')
        self.zstd_value = QtGui.QLabel('0')
                
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

        # self.img.setLookupTable('plasma', )          
        # set up histogram for the liveview image

        self.hist = pg.HistogramLUTItem(image=self.img)
        # self.hist.setPredefinedGradient('greyclip')
        # lut = viewbox_tools.generatePgColormap('inferno')
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
                        
        self.xmeanCurve = self.xyzGraph.xPlot.plot(pen='b', width=40)
            
        self.xyzGraph.yPlot = self.xyzGraph.addPlot(row=1, col=0)
        self.xyzGraph.yPlot.setLabels(bottom=('Time', 's'),
                                     left=('Y position', 'nm'))
        self.xyzGraph.yPlot.showGrid(x=True, y=True)
        
        self.ymeanCurve = self.xyzGraph.yPlot.plot(pen='r', width=40)
        
        
        self.xyzGraph.zPlot = self.xyzGraph.addPlot(row=2, col=0)
        self.xyzGraph.zPlot.setLabels(bottom=('Time', 's'),
                                     left=('Z position', 'nm'))
        self.xyzGraph.zPlot.showGrid(x=True, y=True)
        self.zCurve = self.xyzGraph.zPlot.plot(pen='y')
        
        self.xyzGraph.avgIntPlot = self.xyzGraph.addPlot(row=3, col=0)
        self.xyzGraph.avgIntPlot.setLabels(bottom=('Time', 's'),
                                           left=('Av. intensity', 'Counts'))
        self.xyzGraph.avgIntPlot.showGrid(x=True, y=True)
        self.avgIntCurve = self.xyzGraph.avgIntPlot.plot(pen='g')
        
        # xy drift graph (2D point plot)
        
        self.xyPoint = pg.GraphicsWindow()
        self.xyPoint.resize(400, 400)
        self.xyPoint.setAntialiasing(False)
        # self.xyPoint.setAspectLocked(True)
        
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
        
        # z drift graph (1D histogram)
        x = np.arange(-30, 30)
        y = np.zeros(len(x))
    
        self.zHist = pg.BarGraphItem(x=x, height=y, width=0.6, brush='#3BC14A')

        self.zWin = self.xyPoint.addPlot()
        self.zWin.addItem(self.zHist)
        
        # LiveView Button

        self.liveviewButton = QtGui.QPushButton('Camera LIVEVIEW')
        self.liveviewButton.setCheckable(True)
        
        # create xy ROI button
    
        self.xyROIButton = QtGui.QPushButton('xy ROI')
        self.xyROIButton.setCheckable(True)
        self.xyROIButton.clicked.connect(lambda: self.create_roi(roi_type='xy'))
        
        # create z ROI button
    
        self.zROIButton = QtGui.QPushButton('z ROI')
        self.zROIButton.setCheckable(True)
        self.zROIButton.clicked.connect(lambda: self.create_roi(roi_type='z'))
        
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
        self.trackingBeadsBox.stateChanged.connect(self.setup_data_curves)
        self.trackingBeadsBox.stateChanged.connect(self.emit_roi_info)
        
        # position tracking checkbox
        
        # self.trackZbeamBox = QtGui.QCheckBox('Track z beam')
        # self.trackZbeamBox.stateChanged.connect(self.emit_roi_info)
        
        # turn ON/OFF feedback loop
        
        self.feedbackLoopBox = QtGui.QCheckBox('Feedback loop')

        # save data signal
        
        self.saveDataBox = QtGui.QCheckBox("Save data")
        self.saveDataBox.stateChanged.connect(self.emit_save_data_state)
        
        # pi feedback loop params
        
        self.piParamsLabel = QtGui.QLabel('PI params')
        self.piParamsEdit = QtGui.QLineEdit('0.37 0.37 0.37 0.022 0.022 0.022')
        self.piParamsEdit.textChanged.connect(self.emit_pi_params)
        
        self.piezopiParamsLabel = QtGui.QLabel('Piezo internal PI params')
        self.piezopiParamsEdit =QtGui.QLineEdit('0.0 0.0')
        self.piezopiParamsEdit.textChanged.connect(self.emit_piezopi_params)

        # button to clear the data
        
        self.clearDataButton = QtGui.QPushButton('Clear data')
        
        # button to reset piezo position
        
        self.resetPiezoPosButton = QtGui.QPushButton('Reset piezo position')
        
        # button to make custom pattern
        
        self.xyPatternButton = QtGui.QPushButton('Start pattern')
                
        # buttons and param layout
        
        grid.addWidget(self.paramWidget, 0, 1)
        grid.addWidget(imageWidget, 0, 0)
        grid.addWidget(self.statWidget, 0, 2)

        subgrid = QtGui.QGridLayout()
        self.paramWidget.setLayout(subgrid)

        subgrid.addWidget(self.liveviewButton, 0, 0)
        subgrid.addWidget(self.xyROIButton, 1, 0)
        subgrid.addWidget(self.zROIButton, 2, 0)
        subgrid.addWidget(self.selectxyROIbutton, 3, 0)
        subgrid.addWidget(self.selectzROIbutton, 4, 0)
        subgrid.addWidget(self.delete_roiButton, 5, 0)
        # subgrid.addWidget(self.exportDataButton, 6, 0)
        subgrid.addWidget(self.clearDataButton, 6, 0)
        subgrid.addWidget(self.resetPiezoPosButton, 7, 0)
        subgrid.addWidget(self.xyPatternButton, 8, 0)
        subgrid.addWidget(self.trackingBeadsBox, 1, 1)
        # subgrid.addWidget(self.trackZbeamBox, 2, 1)
        subgrid.addWidget(self.feedbackLoopBox, 2, 1)
        subgrid.addWidget(self.saveDataBox, 3, 1)
        subgrid.addWidget(self.piParamsLabel, 4, 1)
        subgrid.addWidget(self.piParamsEdit, 5, 1)
        subgrid.addWidget(self.piezopiParamsLabel, 6, 1)
        subgrid.addWidget(self.piezopiParamsEdit, 7, 1)
        
        stat_subgrid = QtGui.QGridLayout()
        self.statWidget.setLayout(stat_subgrid)
        
        stat_subgrid.addWidget(self.xstd_label, 0, 0)
        stat_subgrid.addWidget(self.ystd_label, 1, 0)
        stat_subgrid.addWidget(self.zstd_label, 2, 0)
        stat_subgrid.addWidget(self.xstd_value, 0, 1)
        stat_subgrid.addWidget(self.ystd_value, 1, 1)
        stat_subgrid.addWidget(self.zstd_value, 2, 1)
        
        grid.addWidget(self.xyzGraph, 1, 0)
        grid.addWidget(self.xyPoint, 1, 1, 1, 2)
        
        self.liveviewButton.clicked.connect(lambda: self.toggle_liveview(self.liveviewButton.isChecked()))
        
    def setup_data_curves(self):
                    
        if self.trackingBeadsBox.isChecked():
            
            if self.xCurve is not None:
        
                for i in range(len(self.roilist)): # remove previous curves
                
                    self.xyzGraph.xPlot.removeItem(self.xCurve[i]) 
                    self.xyzGraph.yPlot.removeItem(self.yCurve[i]) 
                
            self.xCurve = [0] * len(self.roilist)
            
            for i in range(len(self.roilist)):
                self.xCurve[i] = self.xyzGraph.xPlot.plot(pen='b', alpha=0.3)
                self.xCurve[i].setAlpha(0.3, auto=False)
                
            self.yCurve = [0] * len(self.roilist)
            
            for i in range(len(self.roilist)):
                self.yCurve[i] = self.xyzGraph.yPlot.plot(pen='r', alpha=0.3)
                self.yCurve[i].setAlpha(0.3, auto=False) 
                    
        else:
            
            pass
            
    def closeEvent(self, *args, **kwargs):
        
        print('close in frontend')
        
        self.closeSignal.emit()
        super().closeEvent(*args, **kwargs)
        app.quit()
        
        
class Backend(QtCore.QObject):
    
    changedImage = pyqtSignal(np.ndarray)
    changedData = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)

    xyIsDone = pyqtSignal(bool, float, float)  # signal to emit new piezo position after drift correction
    
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
        
        if VIDEO:
            self.video = []
        
        self.pz = piezo
        self.pz.connect()
        self.pz.set_zero() # important for internal piezo calibration
        
        self.setup_pi(np.array([0.38, 0.3825, 0.37, 0.015, 0.015, 0.022]))
                
        # folder
        
        today = str(date.today()).replace('-', '')
        root = r'C:/Users/Santiago/Documents/minflux/data/'
        self.folder = root + today
        
        try:
            os.mkdir(self.folder)
            
        except OSError:
            print ("Creation of the directory %s failed" % self.folder)
            
        else:
            print ("Successfully created the directory %s " % self.folder)

        
        self.view_timer = QtCore.QTimer()
        self.xyz_time = 200 # 200 ms per acquisition + fit + correction
        
        self.tracking_value = False
        self.save_data_state = False
        self.feedback_active = False
        self.camON = False

        self.npoints = 1200
        self.buffersize = 30000 # TODO: fix bug when self.j reaches buffersize

        self.counter = 0

        # saves displacement when offsetting setpoint for feedbackloop
        
        self.displacement = np.array([0.0, 0.0])
        self.pattern = False
        
        self.previous_image = None
        
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
        
        t0 = time.time()
        self.update_view()
        t1 = time.time() 
           
        # print('updateview took', (t1-t0)*1000, 'ms')

        if self.tracking_value:
            
            t0 = time.time()
            self.track('xy')
            t1 = time.time()
        
            print('track xy took', (t1-t0)*1000, 'ms')
            
            t0 = time.time()
            self.track('z')
            t1 = time.time()
            
            print('track z took', (t1-t0)*1000, 'ms')
            
            t0 = time.time()
            self.update_graph_data()
            t1 = time.time()
            
            print('update graph data took', (t1-t0)*1000, 'ms')
            
            if self.feedback_active:
                    
                t0 = time.time()    
                self.correct()
                t1 = time.time()
                
                print('correct took', (t1-t0)*1000, 'ms')
                
        if self.pattern:
            
            val = (self.counter - self.initcounter)
            num_of_frames = 50
            if (val % num_of_frames == 0):
                self.make_tracking_pattern(val//num_of_frames)

        self.counter += 1  # counter to check how many times this function is executed

    def update_view(self):
        """ Image update while in Liveview mode """

        # acquire image
        
        raw_image = self.camera.latest_frame()
        
        # self.image = np.sum(raw_image, axis=2)   # sum the R, G, B images
        self.image = raw_image[:, :, 0] # take only R channel
        
        # WARNING: fix to match camera orientation with piezo orientation
        self.image = np.rot90(self.image, k=3)
        
        if np.all(self.previous_image == self.image):
            
            print('WARNING: latest_frame equal to previous frame')
    
        self.previous_image = self.image
        
        if (VIDEO and self.save_data_state):
                
            self.video.append(self.image)
    
        # send image to the Frontend
        
        self.changedImage.emit(self.image)
        
    def update_graph_data(self):
        """ Update the data displayed in the graphs """
        
        if self.ptr < self.npoints:
            self.xData[self.ptr, :] = self.x + self.displacement[0]
            self.yData[self.ptr, :] = self.y + self.displacement[1]
            self.zData[self.ptr] = self.z
            self.avgIntData[self.ptr] = self.avgInt
            self.time[self.ptr] = self.currentTime
            
            self.changedData.emit(self.time[0:self.ptr + 1],
                                  self.xData[0:self.ptr + 1],
                                  self.yData[0:self.ptr + 1],
                                  self.zData[0:self.ptr + 1],
                                  self.avgIntData[0:self.ptr + 1])
            
        else:
            self.xData[:-1] = self.xData[1:]
            self.xData[-1, :] = self.x + self.displacement[0]
            self.yData[:-1] = self.yData[1:]
            self.yData[-1, :] = self.y + self.displacement[1]
            self.zData[:-1] = self.zData[1:]
            self.zData[-1] = self.z
            self.avgIntData[:-1] = self.avgIntData[1:]
            self.avgIntData[-1] = self.avgInt
            self.time[:-1] = self.time[1:]
            self.time[-1] = self.currentTime
            
            self.changedData.emit(self.time, self.xData, self.yData, 
                                  self.zData, self.avgIntData)

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
            
            # initialize relevant xy-tracking arrays
        
            size = len(self.roi_coordinates_list)
            
            self.currentx = np.zeros(size)
            self.currenty = np.zeros(size)
            self.x = np.zeros(size)
            self.y = np.zeros(size)
            
            if self.initial is True:
                
                self.initialx = np.zeros(size)
                self.initialy = np.zeros(size)
                    
        if val is False:
        
            self.tracking_value = False
                    
    @pyqtSlot(bool)
    def toggle_feedback(self, val, mode='continous'):
        ''' 
        Connection: [frontend] feedbackLoopBox.stateChanged
        Description: toggles ON/OFF feedback for continous active correction
        '''
        
        self.displacement = np.array([0.0, 0.0])
        
        if val is True:
            
            self.feedback_active = True

            if DEBUG:
                print(datetime.now(), '[xy_tracking] Feedback loop ON')
            
            
        if val is False:
            self.feedback_active = False
            
            if DEBUG:
                print(datetime.now(), '[xy_tracking] Feedback loop OFF')
                
    @pyqtSlot(np.ndarray)
    def setup_pi(self, params):
        
        kp_x, kp_y, kp_z, ki_x, ki_y, ki_z = params
        
        print('kp_x, kp_y, kp_z, ki_x, ki_y, ki_z')
        print(params)
        
        setpoint = 0
        
        self.pi_x = tools.PI(setpoint, multiplier=0.001, kp=kp_x, ki=ki_x)
        
        self.pi_y = tools.PI(setpoint, multiplier=0.001, kp=kp_y, ki=ki_y)
        
        self.pi_z = tools.PI(setpoint, multiplier=0.001, kp=kp_z, ki=ki_z)
      
    @pyqtSlot(np.ndarray)
    def setup_piezo_pi(self, params):
        
        kp, ki = params
        print('piezo internal kp ki')
        print(kp, ki)
        
        for i in range(1,4):
                
            chan = self.pz._piezo.GetChannel(i)
            cts = chan.GetFeedbackLoopPIconsts()
            cts.ProportionalTerm = int(kp)
            cts.IntegralTerm = int(ki)

        
    def center_of_mass(self):
        
        # set main reference frame
        
        xmin, xmax, ymin, ymax = self.zROIcoordinates
        
        # select the data of the image corresponding to the ROI

        zimage = self.image[xmin:xmax, ymin:ymax]
        
        # WARNING: extra rotation added to match the sensitive direction (hardware)
        
        zimage = np.rot90(zimage, k=3)
        
        # calculate center of mass
        
        self.m_center = np.array(ndi.measurements.center_of_mass(zimage))
        
        # calculate z estimator
        
        self.currentz = np.sqrt(self.m_center[0]**2 + self.m_center[1]**2)
          
    def gaussian_fit(self, roi_coordinates):
        
        # print('roi_coordinates', roi_coordinates)
        
        # set main reference frame

        roi_coordinates = np.array(roi_coordinates, dtype=np.int)
        
        xmin, xmax, ymin, ymax = roi_coordinates
        xmin_nm, xmax_nm, ymin_nm, ymax_nm = roi_coordinates * PX_SIZE
        
        # select the data of the image corresponding to the ROI

        array = self.image[xmin:xmax, ymin:ymax]
        
        if np.size(array) == 0:
            
            print('WARNING: array is []')
            
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
        σ = 200 # nm
        x0 = x_sub_nm[int(xsubsize/2)]
        y0 = y_sub_nm[int(ysubsize/2)]
        
        initial_guess_G = [A, x0, y0, σ, σ, bkg]
        
        if np.size(array_sub) == 0:
            
            print('WARNING: array_sub is []')
        
        poptG, pcovG = opt.curve_fit(PSF.gaussian2D, (Mx_sub, My_sub), 
                                      array_sub.ravel(), p0=initial_guess_G)
        
        perr = np.sqrt(np.diag(pcovG))
        
        print('perr', perr)
                
        # retrieve results

        poptG = np.around(poptG, 2)
    
        A, x0, y0, σ_x, σ_y, bkg = poptG
        
        x = x0 + Mx_nm[xmin_id, ymin_id]
        y = y0 + My_nm[xmin_id, ymin_id]
                
        currentx = x
        currenty = y
        
        return currentx, currenty
        
            
    def track(self, track_type):
        
        """ 
        Function to track fiducial markers (Au NPs) from the selected ROI.
        The position of the NPs is calculated through an xy gaussian fit 
        If feedback_active = True it also corrects for drifts in xy
        If save_data_state = True it saves the xy data
        
        """
        
        # Calculate average intensity in the image to check laser fluctuations
        
        self.avgInt = np.mean(self.image)
        
        print('Average intensity', self.avgInt)
        
        # xy track routine of N=size fiducial AuNP

        if track_type == 'xy':
            
            for i, roi in enumerate(self.roi_coordinates_list):
                
                # try:
                #     roi = self.roi_coordinates_list[i]
                #     self.currentx[i], self.currenty[i] = self.gaussian_fit(roi)
                    
                # except(RuntimeError, ValueError):
                    
                #     print(datetime.now(), '[xy_tracking] Gaussian fit did not work')
                #     self.toggle_feedback(False)
                
                roi = self.roi_coordinates_list[i]
                self.currentx[i], self.currenty[i] = self.gaussian_fit(roi)
           
            if self.initial is True:
                
                for i, roi in enumerate(self.roi_coordinates_list):
                       
                    self.initialx[i] = self.currentx[i]
                    self.initialy[i] = self.currenty[i]
                    
                self.initial = False
            
            for i, roi in enumerate(self.roi_coordinates_list):
                    
                self.x[i] = self.currentx[i] - self.initialx[i]  # self.x is relative to initial pos
                self.y[i] = self.currenty[i] - self.initialy[i]
                
                self.currentTime = time.time() - self.startTime
                
            # print('x, y', self.x, self.y)
            # print('currentx, currenty', self.currentx, self.currenty)
                
            if self.save_data_state:
                
                self.time_array[self.j] = self.currentTime
                self.x_array[self.j, :] = self.x + self.displacement[0]
                self.y_array[self.j, :] = self.y + self.displacement[1]
                
                self.j += 1
                            
                if self.j >= (self.buffersize - 5):    # TODO: -5, arbitrary bad fix
                    
                    self.export_data()
                    self.reset_data_arrays()
                    
            #         print(datetime.now(), '[xy_tracking] Data array, longer than buffer size, data_array reset')
            
        # z track of the reflected IR beam        
            
        if track_type == 'z':
            
            self.center_of_mass()
            
            if self.initial_focus is True:
                
                self.initialz = self.currentz
                
                self.initial_focus = False
            
            self.z = (self.currentz - self.initialz) * PX_Z
            
                
    def correct(self, mode='PI'):
        
        xmean = np.mean(self.x)
        ymean = np.mean(self.y)
        security_thr = 0.35 # in µm
        
        if mode == 'ON/OFF':

            dx = 0
            dy = 0
            dz = 0
            # threshold = 3
            # far_threshold = 12
            # correct_factor = 0.6
            
            threshold = 3
            z_threshold = 3
            far_threshold = 12
            correct_factor = 0.6
                        
            if np.abs(xmean) > threshold:
                
                dx = - (xmean)/1000 # conversion to µm
                
                if dx < far_threshold: #TODO: double check this conditions (do they work?)
                    
                    dx = correct_factor * dx #TODO: double check this conditions (do they work?)
    
    #                print('dx', dx)
                
            if np.abs(ymean) > threshold:
                            
                dy = - (ymean)/1000 # conversion to µm
                
                if dy < far_threshold:
                    
                    dy = correct_factor * dy
                
    #                print('dy', dy)
    
            if np.abs(self.z) > z_threshold:
                            
                dz = - (self.z)/1000 # conversion to µm
                
                if dz < far_threshold:
                    
                    dz = correct_factor * dz
                
    #                print('dy', dy)
    
        elif mode == 'PI':
            
            pass
        
            dx = self.pi_x.update(xmean)
            dy = self.pi_y.update(ymean)
            dz = self.pi_z.update(self.z)
        
        else:
            
            print('Please choose a valid feedback mode')
            print('Feedback modes: ON/OFF, PI')
    
        if dx > security_thr or dy > security_thr or dz > 2 * security_thr:
            
            print(datetime.now(), '[xy_tracking] Correction movement larger than 200 nm, active correction turned OFF')
            self.toggle_feedback(False)
            
        else:
            
            # compensate for the mismatch between camera/piezo system of reference
            
            # theta = np.radians(-3.7)   # 86.3 (or 3.7) is the angle between camera and piezo (measured)
            # c, s = np.cos(theta), np.sin(theta)
            # R = np.array(((c,-s), (s, c)))
            
            # dy, dx = np.dot(R, np.asarray([dx, dy]))
            
            # print('dx, dy', dx, dy)
            
            # # add correction to piezo position
            
            current_piezo_pos = self.pz.get_positions()
            
            currentx_piezo = current_piezo_pos[0] # in µm
            currenty_piezo = current_piezo_pos[1] # in µm
            currentz_piezo = current_piezo_pos[2] # in µm
            
            # print('current x, y, z', currentx_piezo, currenty_piezo, currentz_piezo)
            
            # TODO: check the ad hoc fixes
            
            targetx_piezo = currentx_piezo + dx # in µm
            # targety_piezo = currenty_piezo + dy - 0.0035 # in µm
            # targetz_piezo = currentz_piezo + dz - 0.0035 # in µm
            targety_piezo = currenty_piezo + dy  # in µm
            targetz_piezo = currentz_piezo + dz  # in µm
            
            # print('target x y z', targetx_piezo, targety_piezo, targetz_piezo)
                        
            self.actuator_xyz(targetx_piezo, targety_piezo, targetz_piezo)
                         
    def actuator_xyz(self, x, y, z=None):
        
        self.pz.set_positions([float(x), float(y), float(z)]) # in µm
        
    def start_tracking_pattern(self):
        
        self.pattern = True
        self.initcounter = self.counter
        self.save_data_state = True
        
    def make_tracking_pattern(self, step):
                
        # if (step < 1) or (step > 4):
        #     return
        # elif step == 1:
        #     dist = np.array([0.0, 10.0])
        # elif step == 2:
        #     dist = np.array([10.0, 0.0])
        # elif step == 3:
        #     dist = np.array([0.0, -10.0])
        # elif step == 4:
        #     dist = np.array([-10.0, 0.0])
            
        #     self.export_data()
        #     self.save_data_state = False
        
        # if (step < 1) or (step > 10):
        #     return
        # elif step == 1:
        #     dist = np.array([0.0, -50.0])
        # elif step == 2:
        #     dist = np.array([0.0, 10.0])
        # elif step == 3:
        #     dist = np.array([0.0, 20.0])
        # elif step == 4:
        #     dist = np.array([0.0, 30.0])
        # elif step == 5:
        #     dist = np.array([0.0, 40.0])
        # elif step == 6:
        #     dist = np.array([0.0, 50.0])
        # elif step == 7:
        #     dist = np.array([0.0, 60.0])
        # elif step == 8:
        #     dist = np.array([0.0, 70.0])
        # elif step == 9:
        #     dist = np.array([0.0, 80.0])
        # elif step == 10:
        #     dist = np.array([0.0, 90.0])
        # elif step == 11:
        #     dist = np.array([0.0, 100.0])
            
        #     self.export_data()
        #     self.save_data_state = False
        
        deltax = 10
        initialpos = -50
        
        if (step < 0) or (step > 10):
            return
        elif step == 0:
            dist = np.array([initialpos, 0.0])
        elif step == 1:
            dist = np.array([deltax, 0.0])
        elif step == 2:
            dist = np.array([deltax, 0.0])
        elif step == 3:
            dist = np.array([deltax, 0.0])
        elif step == 4:
            dist = np.array([deltax, 0.0])
        elif step == 5:
            dist = np.array([deltax, 0.0])
        elif step == 6:
            dist = np.array([deltax, 0.0])
        elif step == 7:
            dist = np.array([deltax, 0.0])
        elif step == 8:
            dist = np.array([deltax, 0.0])
        elif step == 9:
            dist = np.array([deltax, 0.0])
        elif step == 10:
            dist = np.array([deltax, 0.0])
            
            self.export_data()
            self.save_data_state = False
        
        self.initialx = self.initialx + dist[0]
        self.initialy = self.initialy + dist[1]
        self.displacement = self.displacement + dist
        
        print(datetime.now(), '[xy_tracking] Moved setpoint by', dist)
        
    def reset_piezo_position(self):
        
        for i in range(10):
            
            self.pz.set_positions([i+1, i+1, i+1]) # go to the middle of the piezo range
            time.sleep(.1)
            
    def reset(self):
        
        self.initial = True
        self.initial_focus = True
        
        try:
            self.xData = np.zeros((self.npoints, len(self.roi_coordinates_list)))
            self.yData = np.zeros((self.npoints, len(self.roi_coordinates_list)))
            
        except:
            
            self.xData = np.zeros(self.npoints)
            self.yData = np.zeros(self.npoints)
        
        self.zData = np.zeros(self.npoints)
        self.avgIntData = np.zeros(self.npoints)
        self.time = np.zeros(self.npoints)
        self.ptr = 0
        self.startTime = time.time()
        
        self.changedData.emit(self.time, self.xData, self.yData, self.zData, 
                              self.avgIntData)
        
    def reset_data_arrays(self):
        
        self.time_array = np.zeros(self.buffersize, dtype=np.float16)
        
        self.x_array = np.zeros((self.buffersize, 
                                 len(self.roi_coordinates_list)), 
                                 dtype=np.float16)
        
        self.y_array = np.zeros((self.buffersize, 
                                 len(self.roi_coordinates_list)), 
                                 dtype=np.float16)
        
        self.j = 0  # iterator on the data array

    def export_image(self):
        
        fname = self.folder + '/image'
        
        #case distinction to prevent wrong filenaming when starting minflux or psf measurement
        if fname[0] == '!':
            filename = fname[1:]
        else:
            filename = tools.getUniqueName(fname)
        
        tifffile.imwrite(fname + '.tif', self.image)   

    def export_data(self):
        
        """
        Exports the x, y and t data into a .txt file
        """
        
        fname = self.folder + '/xy_data'
        
        #case distinction to prevent wrong filenaming when starting minflux or psf measurement
        if fname[0] == '!':
            filename = fname[1:]
        else:
            filename = tools.getUniqueName(fname)
        filename = filename + '.txt'
        
        size = self.j
        N_NP = len(self.roi_coordinates_list)
        
        savedData = np.zeros((size, 2*N_NP+1))

        savedData[:, 0] = self.time_array[0:self.j]
        savedData[:, 1:N_NP+1] = self.x_array[0:self.j, :]
        savedData[:, N_NP+1:2*N_NP+1] = self.y_array[0:self.j, :]
        
        np.savetxt(filename, savedData,  header='t (s), x (nm), y(nm)') # transpose for easier loading
        
        print(datetime.now(), '[xy_tracking] xy data exported to', filename)
        print('Exported data shape', np.shape(savedData))
        
        self.export_image()
        
        if VIDEO:
            
            tifffile.imwrite(fname + 'video' + '.tif', np.array(self.video))

    @pyqtSlot(bool)
    def get_save_data_state(self, val):
        '''
        Connection: [frontend] saveDataSignal
        Description: gets value of the save_data_state variable, True -> save,
        False -> don't save
        '''
        
        print(self.pz.get_positions())

        if self.save_data_state is True:
                self.export_data()
        else:
            pass
             
        self.reset_data_arrays()
        self.save_data_state = val
        

        if DEBUG:
            print(datetime.now(), '[xy_tracking] save_data_state = {}'.format(val))

    @pyqtSlot(str, int, list)
    def get_roi_info(self, roi_type, N, coordinates_list):
        
        '''
        Connection: [frontend] roiInfoSignal
        Description: gets coordinates of the ROI in the GUI
        
        '''
        
        if roi_type == 'xy':
                            
            self.roi_coordinates_list = coordinates_list
        
            if DEBUG:
                print(datetime.now(), '[xy_tracking] got ROI coordinates list')
                
        if roi_type == 'z':
            
            self.zROIcoordinates = coordinates_list[0].astype(int)
                 
    def make_connection(self, frontend):
            
        frontend.roiInfoSignal.connect(self.get_roi_info)
        frontend.z_roiInfoSignal.connect(self.get_roi_info)
        frontend.closeSignal.connect(self.stop)
        frontend.saveDataSignal.connect(self.get_save_data_state)
        frontend.exportDataButton.clicked.connect(self.export_data)
        frontend.clearDataButton.clicked.connect(self.reset)
        frontend.clearDataButton.clicked.connect(self.reset_data_arrays)
        frontend.resetPiezoPosButton.clicked.connect(self.reset_piezo_position)
        frontend.trackingBeadsBox.stateChanged.connect(lambda: self.toggle_tracking(frontend.trackingBeadsBox.isChecked()))
        frontend.liveviewButton.clicked.connect(self.liveview)
        frontend.feedbackLoopBox.stateChanged.connect(lambda: self.toggle_feedback(frontend.feedbackLoopBox.isChecked()))
        frontend.piParamsSignal.connect(self.setup_pi)
        frontend.piezopiParamsSignal.connect(self.setup_piezo_pi)
        frontend.xyPatternButton.clicked.connect(self.start_tracking_pattern)
        
        # TODO: clean-up checkbox such that they're fully reversible
        
        # lambda function and gui_###_state are used to toggle both backend
        # states and checkbox status so that they always correspond 
        # (checked <-> active, not checked <-> inactive)
        
    @pyqtSlot()    
    def stop(self):
        
        print('close in backend')

        self.view_timer.stop()
        
        self.camera.stop_live_video()
        self.camera.close()
                                
        for i in range(10):
            
            self.pz.set_positions([9-i, 9-i, 0]) # go to the middle of the piezo range
            time.sleep(.1)
            
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
    print('Cam Serial number {}'.format(cam.serial))
    
    print('Piezo Serial number {}'.format(bpc.list_devices()))
    
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