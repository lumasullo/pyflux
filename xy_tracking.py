# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 11:59:13 2019

@author: Luciano A. Masullo
"""

import numpy as np
import time
import ctypes as ct
from datetime import date

import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import Dock, DockArea
import pyqtgraph.ptime as ptime
from scipy import optimize as opt

import tools.viewbox_tools as viewbox_tools
import tools.colormaps as cmaps
import tools.PSF as PSF
import tools.tools as tools
import scan

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
import qdarkstyle

from lantz.drivers.andor import ccd 
import drivers.ADwin as ADwin



class Frontend(QtGui.QFrame):
    
    liveviewSignal = pyqtSignal(bool)
    roiInfoSignal = pyqtSignal(int, np.ndarray)
    closeSignal = pyqtSignal()
    saveDataSignal = pyqtSignal(bool)
    
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
        
        self.trackingBeadsBox = QtGui.QCheckBox('Track beads')
        self.trackingBeadsBox.stateChanged.connect(self.emit_roi_info)
        
        # turn ON/OFF feedback loop
        
        self.feedbackLoopBox = QtGui.QCheckBox('Closed loop')
        
        # save data signal
        
        self.saveDataBox = QtGui.QCheckBox("Save data")
        self.saveDataBox.stateChanged.connect(self.emit_save_data_state)
        
        # button to clear the data
        
        self.clearDataButton = QtGui.QPushButton('Clear data')
        
        # buttons and param layout
        
        subgrid = QtGui.QGridLayout()
        self.paramWidget.setLayout(subgrid)

        subgrid.addWidget(self.liveviewButton, 0, 0)
        subgrid.addWidget(self.ROIButton, 1, 0)
        subgrid.addWidget(self.delete_roiButton, 2, 0)
        subgrid.addWidget(self.exportDataButton, 3, 0)
        subgrid.addWidget(self.trackingBeadsBox, 4, 0)
        subgrid.addWidget(self.saveDataBox, 5, 0)
        subgrid.addWidget(self.feedbackLoopBox, 6, 0)
        subgrid.addWidget(self.clearDataButton, 7, 0)
        
        grid.addWidget(self.xyGraph, 1, 0)
        
        
    def closeEvent(self, *args, **kwargs):
        
        self.closeSignal.emit()
        
        super().closeEvent(*args, **kwargs)
        
class Backend(QtCore.QObject):
    
    changedImage = pyqtSignal(np.ndarray)
    changedData = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)

    
    def __init__(self, andor, adw, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.andor = andor
        self.adw = adw
        self.initialize_camera()
        self.setup_camera()
        
        # initialize fpar_50, fpar_51, fpar_52 ADwin position parameters
        
        pos_zero = tools.convert(0, 'XtoU')
        
        self.adw.Set_FPar(50, pos_zero)
        self.adw.Set_FPar(51, pos_zero)
        self.adw.Set_FPar(52, pos_zero)
        
        self.moveTo(10, 10, 10) # in µm
        
        time.sleep(0.200)
        
        self.piezoXposition = 10.0 # in µm
        self.piezoYposition = 10.0 # in µm
        self.piezoZposition = 10.0 # in µm
        
        # folder
        
        today = str(date.today()).replace('-', '')
        root = r'C:\\Data\\'
        folder = root + today
        
        filename = r'xydata'
        self.filename = folder + filename
        
        self.viewtimer = QtCore.QTimer()
        self.viewtimer.timeout.connect(self.update_view)
        
        self.tracking_value = False
        self.save_data_state = False
        self.feedback_active = False
        self.n = 0  # number of frames that it are averaged, 0 means no average
        self.i = 0  # update counter
        self.npoints = 400
        self.buffersize = 30000
        
        self.reset()
        self.reset_data_arrays()
        
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
            
            self.reset()
            self.reset_data_arrays()
        
        else:
        
            self.tracking_value = False
            
    @pyqtSlot()
    def toggle_feedback(self):
        
        if self.feedback_active is False:
            self.feedback_active = True
            print('Feedback loop ON')
            
        else:
            
            self.feedback_active = False
            print('Feedback loop OFF')
            
    def tracking(self, i, initial=False):
        
        # set main reference frame
        
        xmin, xmax, ymin, ymax = self.ROIcoordinates[i]
        xmin_nm, xmax_nm, ymin_nm, ymax_nm = self.ROIcoordinates[i] * self.pxSize
        
        # select the data of the image corresponding to the ROI

        array = self.image[xmin:xmax, ymin:ymax]
        
        # set new reference frame
        
        xrange_nm = xmax_nm - xmin_nm
        yrange_nm = ymax_nm - ymin_nm
             
        x_nm = np.arange(0, xrange_nm, self.pxSize)
        y_nm = np.arange(0, yrange_nm, self.pxSize)
        
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
        
        x_sub_nm = np.arange(0, xsubsize) * self.pxSize
        y_sub_nm = np.arange(0, ysubsize) * self.pxSize

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
    
        self.currentx = x0 + Mx_nm[xmin_id, ymin_id]
        self.currenty = y0 + My_nm[xmin_id, ymin_id]
               
        if self.initial is True:
            
            self.initialx = self.currentx
            self.initialy = self.currenty
            
            self.initial = False
            
        self.x = self.currentx - self.initialx
        self.y = self.currenty - self.initialy
    
        self.currentTime = ptime.time() - self.startTime
        
        if self.save_data_state:
            
            self.time_array[self.j] = self.currentTime
            self.x_array[self.j] = self.x
            self.y_array[self.j] = self.y
            
            self.j += 1
            
            if self.j >= (self.buffersize - 5):    # TO DO: -5 bad fix
                
                self.export_data()
                self.reset_data_arrays()
                print('Data array, longer than buffer size, data_array reset')
                
        if self.feedback_active:
            
            dx = 0
            dy = 0
            threshold = 7
            far_threshold = 15
            correct_factor = 0.6
            security_thr = 0.2 # in µm
            
            if np.abs(self.x) > threshold:
                
                if dx < far_threshold:
                    
                    dx = correct_factor * dx
                
                dx = - (self.x)/1000 # conversion to µm
                
                print('dx', dx)
                
            if np.abs(self.y) > threshold:
                
                if dy < far_threshold:
                    
                    dy = correct_factor * dy
                
                dy = - (self.y)/1000 # conversion to µm
        
            if dx > security_thr or dy > security_thr:
                
                print('Correction movement larger than 200 nm, active correction set too OFF')
                
            else:
                
                self.piezoXposition = self.piezoXposition + dy
                self.piezoYposition = self.piezoYposition + dx
                
                self.moveTo(self.piezoXposition, self.piezoYposition, self.piezoZposition)
                
        time.sleep(0.300)
                
                
    def set_moveTo_param(self, x_f, y_f, z_f, n_pixels_x=128, n_pixels_y=128,
                         n_pixels_z=128, pixeltime=1000):

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
        
        self.initial = True
        self.xData = np.zeros(self.npoints)
        self.yData = np.zeros(self.npoints)
        self.time = np.zeros(self.npoints)
        self.ptr = 0
        self.startTime = ptime.time()
        self.j = 0  # iterator on the data array
        
        self.changedData.emit(self.time, self.xData, self.yData)
        
    def reset_data_arrays(self):
        
        self.time_array = np.zeros(self.buffersize, dtype=np.float16)
        self.x_array = np.zeros(self.buffersize, dtype=np.float16)
        self.y_array = np.zeros(self.buffersize, dtype=np.float16)
        
    @pyqtSlot(bool)
    def get_save_data_state(self, val):
        
        self.save_data_state = val
        print('save_data_state = {}'.format(val))
        
    def export_data(self):

        fname = self.filename
        print('fname', fname)
        filename = tools.getUniqueName(fname)
        print('filename', filename)
        
        size = self.j
        savedData = np.zeros((3, size))
        
        savedData[0, :] = self.time_array[0:self.j]
        savedData[1, :] = self.x_array[0:self.j]
        savedData[2, :] = self.y_array[0:self.j]
        
        np.savetxt(filename, savedData.T) # transpose for easier loading
        
        print('data exported')
        
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
        frontend.clearDataButton.clicked.connect(self.reset)
        frontend.clearDataButton.clicked.connect(self.reset_data_arrays)
        frontend.feedbackLoopBox.stateChanged.connect(self.toggle_feedback)

        
    def stop(self):
        
        # Go back to 0 position

        x_0 = 0
        y_0 = 0
        z_0 = 0

        self.moveTo(x_0, y_0, z_0)
        
        self.andor.shutter(0, 2, 0, 0, 0)
        
        try:
            self.andor.abort_acquisition()
            
        except:  # TO DO: write this code properly
            
            pass
            
        self.andor.finalize()
        

if __name__ == '__main__':

    app = QtGui.QApplication([])
    app.setStyle(QtGui.QStyleFactory.create('fusion'))
#    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    
    andor = ccd.CCD()
    
    DEVICENUMBER = 0x1
    adw = ADwin.ADwin(DEVICENUMBER, 1)
    scan.setupDevice(adw)
    
    gui = Frontend()
    worker = Backend(andor, adw)
    
    gui.make_connection(worker)
    worker.make_connection(gui)
    
    gui.setWindowTitle('xy drift correction')
    gui.show()
    app.exec_()
        