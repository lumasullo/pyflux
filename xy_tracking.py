# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 11:59:13 2019

@author: Luciano A. Masullo
"""

import numpy as np
import time
import ctypes as ct

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import Dock, DockArea
import pyqtgraph.ptime as ptime
import tools.viewbox_tools as viewbox_tools
import tools.colormaps as cmaps
import tools.PSF as PSF
import tools.tools as tools
from scipy import optimize as opt

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
import qdarkstyle



#from lantz.drivers.legacy.andor import ccd 
from lantz.drivers.andor import ccd 


class Frontend(QtGui.QFrame):
    
    liveviewSignal = pyqtSignal(bool)
    roiInfoSignal = pyqtSignal(int, np.ndarray)
    closeSignal = pyqtSignal()
    saveDataSignal = pyqtSignal(bool)
    
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        # TO DO: finish gaussian fit calculation and plot
        # TO DO: fix multithreading
        
#        self.andor = camera
#        cam = 0
#        self.andor.current_camera = self.andor.camera_handle(cam)
#        self.andor.lib.Initialize()
#        print('idn:', self.andor.idn)
        
#        self.xyworker = xyWorker(self, self.andor)
        
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
        roicoordinates = np.zeros((roinumber, 4))
        
        for i in range(len(self.roilist)):
            
#            print(self.roilist[i].pos())
#            print(self.roilist[i].size())
            xmin, ymin = self.roilist[i].pos()
            xmax, ymax = self.roilist[i].pos() + self.roilist[i].size()

            coordinates = np.array([xmin, xmax, ymin, ymax])  
            roicoordinates[i] = coordinates
            
        self.roiInfoSignal.emit(roinumber, roicoordinates)

    def delete_roi(self):
        
        for i in range(len(self.roilist)):
            
            self.vb.removeItem(self.roilist[i])
            self.roilist[i].hide()
            
        self.roilist = []
        self.delete_roiButton.setChecked(False)
        self.ROInumber = 0
        
    def toggle_liveview(self):
        
        if self.liveviewButton.isChecked():
            
            self.liveviewSignal.emit(True)
        
        else:
            
            self.liveviewSignal.emit(False)
            self.liveviewButton.setChecked(False)
        
        
    @pyqtSlot(np.ndarray)
    def get_image(self, img):
        
        self.img.setImage(img, autoLevels=False)
        
    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def get_data(self, time, xPosition, yPosition):
        
        self.xCurve.setData(time, xPosition)
        self.yCurve.setData(time, yPosition)
        
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
        
        # create ROI button
    
        self.ROIButton = QtGui.QPushButton('add ROI')
        self.ROIButton.setCheckable(True)
        self.ROIButton.clicked.connect(self.craete_roi)
        
        # delete ROI button
        
        self.delete_roiButton = QtGui.QPushButton('delete ROIs')
        self.delete_roiButton.clicked.connect(self.delete_roi)
        
        # position tracking checkbox
        
        self.exportDataButton = QtGui.QPushButton('export current data')

        # position tracking checkbox
        
        self.trackingBeadsBox = QtGui.QCheckBox('track beads positions')
        self.trackingBeadsBox.stateChanged.connect(self.emit_roi_info)
        
        
        self.saveDataBox =  QtGui.QCheckBox("save data")
        self.saveDataBox.stateChanged.connect(self.emit_save_data_state)
        
        # buttons and param layout
        
        subgrid = QtGui.QGridLayout()
        self.paramWidget.setLayout(subgrid)

        subgrid.addWidget(self.liveviewButton, 0, 0)
        subgrid.addWidget(self.ROIButton, 1, 0)
        subgrid.addWidget(self.delete_roiButton, 2, 0)
        subgrid.addWidget(self.exportDataButton, 3, 0)
        subgrid.addWidget(self.trackingBeadsBox, 4, 0)
        subgrid.addWidget(self.saveDataBox, 5, 0)
        
        grid.addWidget(self.xyGraph, 1, 0)
        
        
    def closeEvent(self, *args, **kwargs):
        
        self.closeSignal.emit()
        
        super().closeEvent(*args, **kwargs)
        
class Backend(QtCore.QObject):
    
    changedImage = pyqtSignal(np.ndarray)
    changedData = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)

    
    def __init__(self, andor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.andor = andor
        self.initialize_camera()
        self.setup_camera()
        
        self.filename = r'C:\Data\filename'
        
        self.viewtimer = QtCore.QTimer()
        self.viewtimer.timeout.connect(self.update_view)
        
        self.tracking_value = False
        self.save_data_state = False
        self.n = 0  # number of frames that it are averaged, 0 means no average
        self.i = 0  # update counter
        self.npoints = 400
        
        self.time_array = []
        self.x_array = []
        self.y_array = []
        
        self.reset()
        
    def setup_camera(self):
        
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
        
    def initialize_camera(self):
        
        cam = 0
        self.andor.current_camera = self.andor.camera_handle(cam)
        self.andor.lib.Initialize()
        print('idn:', self.andor.idn)
    
    @pyqtSlot(bool)
    def liveview(self, value):

        if value:
            self.liveview_start()

        else:
            self.liveview_stop()

        
    def liveview_start(self):
        
        self.initial = True
        
        print('Temperature = {} °C'.format(self.andor.temperature))
        print(self.andor.temperature_status)

        # Initial image
        
        self.andor.acquisition_mode = 'Run till abort'
        print('Acquisition mode:', self.andor.acquisition_mode)
        self.andor.shutter(0, 1, 0, 0, 0)
        self.andor.start_acquisition()
        
        time.sleep(self.expTime * 2)
        self.image = self.andor.most_recent_image16(self.shape)

        self.changedImage.emit(self.image)

        self.viewtimer.start(50)
    
    def liveview_stop(self):
        
        self.viewtimer.stop()
        self.andor.abort_acquisition()
        self.andor.shutter(0, 2, 0, 0, 0)
                    
    def update_view(self):
        """ Image update while in Liveview mode
        """
        
        self.image = self.andor.most_recent_image16(self.shape)
        self.changedImage.emit(self.image)

        if self.tracking_value:
            
            for i in range(self.numberOfROIs):
                
                self.tracking(i, self.initial)
                self.update()
    
    @pyqtSlot()
    def toggle_tracking(self):
        
        self.startTime = time.time()
        
        if self.tracking_value is False:
            self.tracking_value = True
        
        else:
        
            self.tracking_value = False
            
    def tracking(self, i, initial=False):
        
        # gaussian fit for every selected particle
    
#        array = self.gui.roilist[i].getArrayRegion(self.image, self.gui.img)
        
        xmin, xmax, ymin, ymax = self.ROIcoordinates[i]
        print('self.ROIcoordinates[i]', self.ROIcoordinates[i])
        array = self.image[xmin:xmax, ymin:ymax]
        
#        ROIpos_nm = np.array(self.gui.roilist[i].pos()) * self.pxSize
    
        xmin_nm, xmax_nm, ymin_nm, ymax_nm = self.ROIcoordinates[i] * self.pxSize
        
        print('xmin', xmin_nm)
        print('ymin', ymin_nm)
        
        x = np.arange(xmin_nm, xmax_nm, self.pxSize)
        y = np.arange(ymin_nm, ymax_nm, self.pxSize)
        
        (Mx, My) = np.meshgrid(x, y)
        
        bkg = np.min(array)
        A = np.max(array) - bkg
        x0 = (xmax_nm + xmin_nm)/2
        y0 = (ymax_nm + ymin_nm)/2

        σ = 130   # in nm
        
        initial_guess_G = [A, x0, y0, σ, σ, bkg]
#
#                t0 = time.time()
#                
        poptG, pcovG = opt.curve_fit(PSF.gaussian2D, (Mx, My), array.ravel(), 
                                     p0=initial_guess_G)
        
        if initial is True:
            
            self.x0 = poptG[1] - xmin_nm
            self.y0 = poptG[2] - ymin_nm
            
            self.initial = False
            print('initial')
            
            dataG = PSF.gaussian2D((Mx, My), *poptG)
            dataG_2d = dataG.reshape(int(np.shape(array)[0]), int(np.shape(array)[0]))
            
#            plt.figure()
#            plt.imshow(array, interpolation=None, cmap=cmaps.parula)
#            
#            plt.figure()
#            plt.imshow(dataG_2d, interpolation=None, cmap=cmaps.parula)
            
        self.x = poptG[1] - xmin_nm - self.x0
        self.y = poptG[2] - ymin_nm - self.y0
        self.currentTime = ptime.time() - self.startTime
        
        print(self.x, self.y)
        
        if self.save_data_state:
            
            self.time_array.append(self.currentTime)
            self.x_array.append(self.x)
            self.y_array.append(self.y)
        
        
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
                self.time[self.ptr] = self.currentTime
                
                self.changedData.emit(self.time[1:self.ptr + 1],
                                      self.xData[1:self.ptr + 1],
                                      self.yData[1:self.ptr + 1])
    
            else:
                self.xData[:-1] = self.xData[1:]
                self.xData[-1] = self.xPosition
                self.yData[:-1] = self.yData[1:]
                self.yData[-1] = self.yPosition
                self.time[:-1] = self.time[1:]
                self.time[-1] = self.currentTime
                
                self.changedData.emit(self.time, self.xData, self.yData)
    
            self.ptr += 1
            
        else:
            
            self.i += 1  
            
    def reset(self):
        
        self.xData = np.zeros(self.npoints)
        self.yData = np.zeros(self.npoints)
        self.time = np.zeros(self.npoints)
        self.ptr = 0
        self.startTime = ptime.time()
        
    def reset_data_arrays(self):
        
        self.time_array = []
        self.x_array = []
        self.y_array = []
        
    @pyqtSlot(bool)
    def get_save_data_state(self, val):
        
        self.save_data_state = val
        print('save_data_state = True')
        
    def export_data(self):
        
        fname = tools.getUniqueName(self.filename)
        fname = fname + '_xydata'
        
        size = np.size(self.x_array)
        savedData = np.zeros((3, size))
        
        savedData[0, :] = np.array(self.time_array)
        savedData[1, :] = np.array(self.x_array)
        savedData[2, :] = np.array(self.y_array)
        
        np.savetxt(fname, savedData)
        
    @pyqtSlot(int, np.ndarray)
    def get_roi_info(self, N, coordinates_array):
        
        self.numberOfROIs = N
        self.ROIcoordinates = coordinates_array.astype(int)
        
    def make_connection(self, frontend):
            
        frontend.liveviewButton.clicked.connect(self.liveview)
        frontend.trackingBeadsBox.stateChanged.connect(self.toggle_tracking)
        frontend.roiInfoSignal.connect(self.get_roi_info)
        frontend.closeSignal.connect(self.stop)
        frontend.saveDataSignal.connect(self.get_save_data_state)
        frontend.exportDataButton.clicked.connect(self.export_data)

        
    def stop(self):
        
        self.andor.shutter(0, 2, 0, 0, 0)
        self.andor.abort_acquisition()
        self.andor.finalize()
        
            
if __name__ == '__main__':

    app = QtGui.QApplication([])
    app.setStyle(QtGui.QStyleFactory.create('fusion'))
#    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    
    andor = ccd.CCD()
    
    gui = Frontend()
    worker = Backend(andor)
    
    gui.make_connection(worker)
    worker.make_connection(gui)
    
    gui.setWindowTitle('xy drift correction')
    gui.show()
    app.exec_()
        