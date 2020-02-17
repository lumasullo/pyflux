# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:05:18 2019

@author: Lars Richter

GUI for MINFLUX analysis and simulations

conda command for converting QtDesigner file to .py:
pyuic5 -x AnalysisDesign.ui -o AnalysisDesign.py

Next steps:
    delete hard-coded 25 in trace_seg
    make optimization finding offset in fit, start fitting slightly already at downslope of exponential
    Save and load tcspc-windows
    Make update_trace() function similar to update_image()
    Neglect zeros in on-times when discarding them
    Delete regions --> e.g. double-click on region
    Swap order of EBP -- match peaks in rel-time and fitted psf
    
    ... 
    
from pyqtgraph import examples
examples.run()
"""

import os

os.chdir(r'C:\Users\Lars\Documents\Studium\Argentina\research\minflux\code\pyflux')

import sys
import time
import ctypes
import configparser
from tkinter import Tk, filedialog
from datetime import date, datetime
import numpy as np
import imageio as iio
from scipy import optimize as opt
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from pyqtgraph.Point import Point

import tools.viewbox_tools as viewbox_tools
import tools.colormaps as cmaps
import tools.tools as tools
import gui.AnalysisDesign
import qdarkstyle
from tools.lineprofile import linePlotWidget

π = np.pi

# see https://stackoverflow.com/questions/1551605
# /how-to-set-applications-taskbar-icon-in-windows-7/1552105#1552105
# to understand why you need the preceeding two lines
myappid = 'mycompany.myproduct.subproduct.version' # arbitrary string
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

class Frontend(QtGui.QMainWindow):
    
    paramSignal = pyqtSignal(dict)
    fitPSFSignal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    loadTCSPCSignal = pyqtSignal()
    estimatorSignal = pyqtSignal(np.ndarray)
    sendPSFfitSignal = pyqtSignal(np.ndarray)

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        self.ui = gui.AnalysisDesign.Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.initialDir = r'Desktop'
        self.k = int(self.ui.spinBox_donuts.text())
        self.regionNum = 0
        self.region = []
        self.ontimes = []
        self.tracePlot = None
        self.tcspcMode = self.ui.radioButton_NP.text()
        
        self.ui.buttonGroup_tcspcmode.buttonClicked['QAbstractButton *'].connect(self.check_tcspcmode)
       
#        self.ui.LoadTCSPCButton.clicked.connect(self.update_line_profile)
#        self.lplotWidget = linePlotWidget()
#        self.lplotWidget.gauss = False
#        self.lplotWidget.get_connection(self)

    def update_line_profile(self):
        
        self.lplotWidget.show()
                
        data = np.arange(60)
        self.lplotWidget.linePlot.clear()
        x = 6.25 * np.arange(np.size(data))*1000
        self.lplotWidget.linePlot.plot(x, data)
        
        print(self.lplotWidget.gauss)
        
        if self.lplotWidget.gauss:
            self.lplotWidget.linePlot.plot(x, 2*data)
        
    def emit_param(self):
        params = dict()

        params['psfFilename'] = self.ui.psfFileEditBox.text()
        params['tcspcFilename'] = self.ui.tcspcEditBox.text()
        params['numDonuts'] = int(self.ui.spinBox_donuts.text())
        params['SBR'] = float(self.ui.lineEdit_sbr.text())
        params['NP_binning_window'] = float(self.ui.lineEdit_winlen.text())
        params['lifetime_win_i'] = float(self.ui.lineEdit_lifetimewin_i.text())
        params['lifetime_win_f'] = float(self.ui.lineEdit_lifetimewin_f.text())
        params['tcspcMode'] = self.tcspcMode
        params['binwidth'] = float(self.ui.binWidthlineEdit.text())
        params['threshold'] = float(self.ui.threshold_lineEdit.text())
        params['minontime'] = float(self.ui.minONlimit_lineEdit.text())
        params['minphotons'] = float(self.ui.minNperON_lineEdit.text())
                
        self.paramSignal.emit(params)

    @pyqtSlot(dict)
    def get_backend_param(self, params):
        
        self.PX = params['pxSize']
        self.ui.threshold_lineEdit.setText(str(params['threshold']))
        
        self.tcspc_binning = params['binwidth']
        self.ui.binWidthlineEdit.setText(str(self.tcspc_binning))
    
    def select_exppsf(self):
        try:
            root = Tk()
            root.withdraw()
            root.filenamePSF = filedialog.askopenfilename(initialdir=self.initialDir,
                                                      title = 'Select PSF file',
                                                      filetypes = [('tiff files','.tiff')])
            if root.filenamePSF != '':
                self.ui.psfFileEditBox.setText(root.filenamePSF)
                
        except OSError:
            pass
        
        if root.filenamePSF == '':
            return
        
        im = iio.mimread(root.filenamePSF)
        self.img_array = np.array(im)
       
        self.emit_param()
        
        self.ui.psfScrollbar.setEnabled(True)
        self.ui.psfScrollbar.setMaximum(self.img_array.shape[0]-1)
        self.ui.psfScrollbar.setSliderPosition(0)
        self.show_psf(self.img_array, 0)
        
        
    def show_psf(self, image_array, image_number):
        
        self.current_images = image_array
            
        imageWidget = pg.GraphicsLayoutWidget()
        
        # set up axis items, scaling is performed in get_image()
        self.xaxis = pg.AxisItem(orientation='bottom', maxTickLength=5)
        self.xaxis.showLabel(show=True)
        self.xaxis.setLabel('x', units='px')
        
        self.yaxis = pg.AxisItem(orientation='left', maxTickLength=5)
        self.yaxis.showLabel(show=True)
        self.yaxis.setLabel('y', units='px')

        # image widget set-up and layout
        self.vb = imageWidget.addPlot(row=0, col=0, axisItems={'bottom': self.xaxis, 
                                                 'left': self.yaxis})

        img = pg.ImageItem(self.current_images[image_number,:,:]) #TODO:edit later on 
        self.vb.clear()
        self.vb.addItem(img)
        self.vb.setAspectLocked(True)

        hist = pg.HistogramLUTItem(image=img)   #set up histogram for the liveview image
        lut = viewbox_tools.generatePgColormap(cmaps.inferno)
        hist.gradient.setColorMap(lut)
        hist.vb.setLimits(yMin=0, yMax=10000) #TODO: check maximum value
        for tick in hist.gradient.ticks:
            tick.hide()
        #imageWidget.addItem(hist, row=0, col=1)

        self.empty_layout(self.ui.psfLayout)        
        self.ui.psfLayout.addWidget(imageWidget)
        
        #test crosshair
        vbox = self.vb.vb
        self.ch = viewbox_tools.Crosshair(vbox)
        self.ch.show()
        
    
    def fit_exppsf(self):
        #room for reading analysis parameter
        self.k = int(self.ui.spinBox_donuts.text())
        self.emit_param()
        print(datetime.now(), '[analysis] PSF fitting started')
        self.x0 = np.zeros(self.k)
        self.y0 = self.x0
        self.fitPSFSignal.emit(self.img_array, self.x0, self.y0)

    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def plot_psffit(self, fittedpsf, x0, y0):
        self.x0 = x0
        self.y0 = y0
        print(datetime.now(), '[analysis] Fitting done and PSF received')

        self.ui.psfScrollbar.setMaximum(fittedpsf.shape[0]-1)
        self.ui.psfScrollbar.setSliderPosition(0)
        self.show_psf(fittedpsf, 0)
        
    @pyqtSlot(int)
    def update_image(self, image_number):
        self.show_psf(self.current_images, image_number)
        
    @pyqtSlot(np.ndarray, np.ndarray)
    def plot_tcspc(self, abs_time, rel_time):
        #TODO: set loaded times as global variables        
        if not self.tracePlot == None:
            for reg in self.region:
                self.tracePlot.removeItem(reg)
        
        dataWidget = pg.GraphicsLayoutWidget()
        dataWidget.clear()
        
        histPlot = dataWidget.addPlot(row=1, col=0, title="Relative Time Histogram")
        histPlot.setLabels(bottom=('ns'),
                                left=('counts'))
        
        self.tracePlot = dataWidget.addPlot(row=2, col=0, title="Time Trace")
        self.tracePlot.setLabels(bottom=('ms'),
                                left=('counts'))
        
        nbins = int((np.max(abs_time)/self.tcspc_binning))
        counts, bin_edges = np.histogram(rel_time, bins=nbins)
        x = np.ediff1d(bin_edges) + bin_edges[:-1]
        histPlot.plot(x, counts)

        counts, bin_edges = np.histogram(abs_time, bins=nbins)
        x = np.ediff1d(bin_edges) + bin_edges[:-1]
        self.tracePlot.plot(x, counts)
            
        self.empty_layout(self.ui.tcspcLayout)
        self.ui.tcspcLayout.addWidget(dataWidget)
        
        self.regionNum = 0 
        self.region_selection()
        self.region[0].setMovable(False)
        self.ui.checkBox_TraceSelection.setChecked(False)
        self.ui.radioButton_NP.setChecked(True)

        print(datetime.now(), '[analysis] TCSPC data received and plotted')
    
    @pyqtSlot(bool)
    @pyqtSlot(QtGui.QAbstractButton)    
    def check_tcspcmode(self, id_or_state):
        
        if isinstance(id_or_state, QtGui.QAbstractButton):
            self.tcspcMode = id_or_state.text()
            if not self.tcspcMode == 'Origami (manual)':
                self.ui.pushButton_addWindow.setEnabled(False)
            if self.tcspcMode == 'Nanoparticle':
                for i in np.arange(1, self.regionNum):
                    self.tracePlot.removeItem(self.region[i])
                self.regionNum = 1
            if self.tcspcMode == 'Origami (auto)':
                self.tracePlot.removeItem(self.region[0])
                self.regionNum = 0
        else:
            select_ontimes = id_or_state
            for i in np.arange(self.regionNum):
                self.region[i].setMovable(select_ontimes)
            if select_ontimes:
                if self.ui.radioButton_NP.isChecked():
                    self.ui.comLabel.setText('Modify macrotime window in plot \n and set window length for binning.')
                if self.ui.radioButton_origami_manual.isChecked():
                    self.ui.comLabel.setText('Feel free to add more windows for on-times.')
                    self.ui.pushButton_addWindow.setEnabled(True)
            else:
                self.ui.pushButton_addWindow.setEnabled(False)
                
    def region_selection(self):
        self.region.append(pg.LinearRegionItem())
        self.region[self.regionNum].setZValue(10)
        self.tracePlot.addItem(self.region[self.regionNum], ignoreBounds=True)
        self.region[self.regionNum].setRegion([0, 0.5])
        
        self.regionNum += 1
                
    def read_ontimes(self):
        for i in np.arange(self.regionNum):
            self.ontimes.append(self.region[i].getRegion()) 
            
        #TODO: write logfile containing selected trace regions and binning information
            
    def select_tcspc(self):
        try:
            root = Tk()
            root.withdraw()
            root.filenameTCSPC = filedialog.askopenfilename(initialdir=self.initialDir,
                                                      title = 'Select TCSPC file',
                                                      filetypes = [('txt files','.txt')])
            if root.filenameTCSPC != '':
                self.ui.tcspcEditBox.setText(root.filenameTCSPC)
                
        except OSError:
            pass 

    @pyqtSlot()
    def load_tcspc(self):

        self.emit_param()
        print(datetime.now(), '[analysis] Loading TCSPC data')
        
        self.loadTCSPCSignal.emit()

    @pyqtSlot()
    def position_estimation(self):
        
        print(datetime.now(), '[analysis] Position estimation started')
        self.ontimes = []
        self.read_ontimes()
        
        if self.ui.radioButton_psffit.isChecked():
            self.sendPSFfitSignal.emit(self.current_images)
            
        self.emit_param()
        self.estimatorSignal.emit(np.array(self.ontimes))
        
    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    def plot_position(self, indrec, pos, Ltot, N):
#https://stackoverflow.com/questions/41060163/pyqtgraph-scatterplotitem-setbrush

        #do some statistics        
        sigmax = np.round(np.std(pos[:, 0]), decimals = 1)
        sigmay = np.round(np.std(pos[:, 1]), decimals = 1)

        Nmean = np.mean(N)
        meanx = '<x> = ' + str(np.round(np.mean(pos[:, 0]), decimals = 1)) + ' nm'
        meany = '<y> = ' + str(np.round(np.mean(pos[:, 1]), decimals = 1)) + ' nm'
        sx = '<σx> = ' + str(sigmax) + ' nm'
        sy = '<σy> = ' + str(sigmay) + ' nm'  
        Nm = '<N> = ' + str(int(np.round(Nmean, decimals = 0)))
        
        #start actual plotting
        estimatorWidget = pg.GraphicsLayoutWidget()
        estimatorWidget.clear()
        plot = estimatorWidget.addPlot(title="Position estimation")
        plot.setLabels(bottom=('x [nm]'), left=('y [nm]'))
        
        for i in np.arange(pos.shape[0]):
            pos_i = pg.ScatterPlotItem([pos[i, 0]], [pos[i, 1]], size=10, symbol='x', pen=pg.mkPen(None))
            plot.addItem(pos_i)
       
        for i in np.arange(self.x0.shape[0]):
            donut_i = pg.ScatterPlotItem([self.x0[i]], [self.y0[i]], size=20, pen=pg.mkPen(None))
            donut_i.setBrush(QtGui.QBrush(QtGui.QColor(QtCore.qrand() % 256, QtCore.qrand() % 256, QtCore.qrand() % 256)))
            plot.addItem(donut_i)
            
        self.empty_layout(self.ui.estimateLayout)
        self.ui.estimateLayout.addWidget(estimatorWidget)
        
        text = meanx + '\n' + meany + '\n' + sx + '\n' + sy + '\n' + Nm 
        self.ui.comLabel.setText(text)
        
        print(datetime.now(), '[analysis] Estimation done and results plotted')
        print(datetime.now(), '[analysis] Stats: \n', text)
        
        #TODO: add options to save plots and result summary in txt file, also save trace length and binning in logfile, plus name of psffile plus name of tcspc file
        #set up everything allowing to save plot
        self.ui.SaveResultsButton.setEnabled(True)
        self.ui.CreatepdfcheckBox.setEnabled(True)
        
    def empty_layout(self, layout):
        for i in reversed(range(layout.count())): 
            layout.itemAt(i).widget().setParent(None)
            
    def make_connection(self, backend):
        
        backend.paramSignal.connect(self.get_backend_param)
        backend.fitPSFSignal.connect(self.plot_psffit)
        backend.plotTCSPCSignal.connect(self.plot_tcspc)
        backend.positionSignal.connect(self.plot_position)
        
    def testmethod(self):
        print('test worked very fine')
        
    def closeEvent(self, event, *args, **kwargs):

        reply = QtGui.QMessageBox.question(self, 'Quit', 'Are you sure you want to quit?',
                                           QtGui.QMessageBox.No |
                                           QtGui.QMessageBox.Yes)
        if reply == QtGui.QMessageBox.Yes:
            event.accept()
            print(datetime.now(), '[analysis] Analysis module closed')        
            super().closeEvent(*args, **kwargs)
            app.quit()
        else:
            event.ignore()
            

class Backend(QtCore.QObject):

    paramSignal = pyqtSignal(dict)
    plotTCSPCSignal = pyqtSignal(np.ndarray, np.ndarray)
    fitPSFSignal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    positionSignal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)

    
    """
    Signals
    
    - paramSignal:
         To: [frontend]
         Description: 
             
    - imageSignal:
         To: [frontend]
         Description: 
        
    """
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        self.PX = 1.0 #px size of function grid for psf interpolation/ fit in nm
        self.ABS_TIME_CONVERSION = 1.0*10**(-3) #conversion factor for absolute time
        self.size = None
        self.pxexp = None
        self.bins = 1000

    def emit_param(self):
        
        params = dict()
        
        params['pxSize'] = self.PX
        params['absTimeConv'] = self.ABS_TIME_CONVERSION
        params['fitpsfSize'] = self.size
        params['pxCamera'] = self.pxexp
        params['binwidth'] = self.tcspc_binning
        params['threshold'] = self.threshold
        
        self.paramSignal.emit(params)
        
    @pyqtSlot(dict)
    def get_frontend_param(self, params):
        
        # updates parameters according to what is input in the GUI
        
        self.psffilename = params['psfFilename']
        self.tcspcfilename = params['tcspcFilename']
        self.k = params['numDonuts']
        self.sbr = params['SBR']
        self.NP_binning = params['NP_binning_window']
        self.lifetime_win_i = params['lifetime_win_i']
        self.lifetime_win_f = params['lifetime_win_f']
        self.tcspcMode = params['tcspcMode']
        self.threshold = params['threshold']
        self.tcspc_binning = params['binwidth']
        self.deltaTmin = params['minontime']
        self.Nmin = params['minphotons']
        
    def space_to_index(self, space):

        # size and px have to be in nm
        index = np.zeros(2)
        offset = [self.size/2, self.size/2]    
        index[1] = (space[0] + offset[0])/self.PX
        index[0] = (offset[1] - space[1])/self.PX
    
        return np.array(index, dtype=np.int)   

    def index_to_space(self, index):
    
        space = np.zeros(2)
        space[1] = (self.size - index[0]) * self.PX
        space[0] = index[1] * self.PX
        return np.array(space)
    
    #TODO: add number of donuts as parameter, adjust input, return description
    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def fit_psf(self, psfexp, x0, y0):
        """   
        Open exp PSF images and drift data, fit exp data with poly_func
    
        Input
        ----------
        datadir : directory where MINFLUX data are saved
        folder : folder with psf excitation and drift data files    
        filename: name or prefix of file containing psf/ drift data 
        k : number of excitation donut positions
        
        Returns
        -------
        
        PSF : (k, sizeg, sizeg) array, function from fit evaluated over grid
        x0, y0 : arrays, coordinates of EBP centers
        index : array, coordinates of EBP centers in indexes
        aopt : fit parameters, coefficients of polynomial function
        
        TODO: please change name of configfile
        """
        
        self.psfexp = psfexp.astype(float)
    
        #create strings pointing to psf, drift and config files
        root = os.path.splitext(self.psffilename)[0]
        drift_file = root + '_xydata.txt'
        config_file = os.path.split(root)[0] + '/filename.txt'
        
        # open any file with metadata from PSF images 
        config = configparser.ConfigParser()
        config.sections()
        config.read(config_file, encoding='ISO-8859-1')
        
        pxex = config['Scanning parameters']['pixel size (µm)']
        self.pxexp = float(pxex) * 1000.0 # convert to nm
    
        # open txt file with xy drift data
        coord = np.loadtxt(drift_file, unpack=True)
        
        # total number of frames
        frames = np.min(self.psfexp.shape)
        
        # number of px in frame
        npx = np.size(self.psfexp, axis = 1)
        
        # final size of fitted PSF arrays (1 nm pixels)             
        sizepsf = int(self.pxexp*npx)
    
        # number of frames per PSF (asumes same number of frames per PSF)
        fxpsf = frames//self.k
    
        # initial frame of each PSF
        fi = fxpsf*np.arange((self.k+1))  
        
        # interpolation to have 1 nm px and realignment with drift data
        psf = np.zeros((frames, sizepsf, sizepsf))        
        for i in np.arange(frames):
            psfz = ndi.interpolation.zoom(self.psfexp[i,:,:], self.pxexp)    
            deltax = coord[1, i] - coord[1, 0]
            deltay = coord[2, i] - coord[2, 0]
            psf[i, :, :] = ndi.interpolation.shift(psfz, [deltax, deltay])
    
        # sum all interpolated and re-centered images for each PSF
        psfT = np.zeros((frames//fxpsf, sizepsf, sizepsf))
        for i in np.arange(self.k):
            psfT[i, :, :] = np.sum(psf[fi[i]:fi[i+1], :, :], axis = 0)
            
        # crop borders to avoid artifacts 
        w, h = psfT.shape[1:3]
        border = (w//5, h//5, w//5, h//5) # left, up, right, bottom
        psfTc = psfT[:, border[1]:h-border[1], border[0]:w-border[0]]
        psfTc = psfT[:, border[1]:h-border[1], border[0]:w-border[0]]
              
        # spatial grid
        self.size = np.size(psfTc, axis = 1)
      
        x = np.arange(0, self.size, self.PX)
        y = self.size - np.arange(0, self.size, self.PX)
        x, y = np.meshgrid(x, y)
        
        # fit PSFs  with poly_func and find central coordinates (x0, y0)
        self.PSF = np.zeros((self.k, self.size, self.size))
        self.x0 = np.zeros(self.k)
        self.y0 = np.zeros(self.k)
        self.index = np.zeros((self.k,2))
        self.aopt = np.zeros((self.k,27))
                            
        for i in np.arange(self.k): 
            # initial values for fit parameters x0,y0 and c00
            ind1 = np.unravel_index(np.argmin(psfTc[i, :, :], 
                axis=None), psfTc[i, :, :].shape)
            x0i = x[ind1]
            y0i = y[ind1]
            c00i = np.min(psfTc[i, :, :])        
            p0 = [x0i, y0i, c00i, 1 ,1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.aopt[i,:], cov = opt.curve_fit(self.poly_func, (x,y), psfTc[i, :, :].ravel(), p0)   
            q = self.poly_func((x,y), *self.aopt[i,:])   
            self.PSF[i, :, :] = np.reshape(q, (self.size, self.size))
            # find min value for each fitted function (EBP centers)
            ind = np.unravel_index(np.argmin(self.PSF[i, :, :], 
                axis=None), self.PSF[i, :, :].shape)
            self.x0[i] = x[ind]
            self.y0[i] = y[ind]
            self.index[i,:] = ind
            print(datetime.now(), '[analysis]', str(i+1), '/', str(self.k), ' donuts fitted')
            
        psfTc[0,:,:] = self.PSF[0, :, :]
        self.PSF[0, :, :] = self.PSF[1, :, :]
        self.PSF[1, :, :] = psfTc[0,:,:]
        psfTc[0,:,:] = self.PSF[2, :, :]
        self.PSF[2, :, :] = self.PSF[3, :, :]
        self.PSF[3, :, :] = psfTc[0,:,:]
        
        self.emit_param()
        self.fitPSFSignal.emit(self.PSF, self.x0, self.y0)


    @pyqtSlot()
    def open_tcspc(self):
        """   
        Open experimental TCSPC data
    
        Input
        ----------
    
        datadir : directory where MINFLUX data are saved
        folder : folder with TCSPC data files    
        filename: name of file containing TSCPC data  
        k : number of excitation donut positions
        
        Returns
        -------
        
        
        abs_time : array, absolute time tags of collected photons
        rel_time : array, relative time tags of collected photon
        τ : times of EBP pulses
        
        """        
        # open txt file with TCSPC data 
        coord = np.loadtxt(self.tcspcfilename, unpack=True)
        
        #convert absTime to s, relTime already given in ns
        self.abs_time = coord[1, :] * self.ABS_TIME_CONVERSION
        self.rel_time = coord[0, :] 
        
        # find EBP pulses times
        nbins = int((np.max(self.abs_time)/self.tcspc_binning))
        [y, bin_edges] = np.histogram(self.rel_time, bins = nbins)
        x = np.ediff1d(bin_edges) + bin_edges[:-1]
        
        T = len(y)//(self.k) * np.arange((self.k+1))
        
        ind = np.zeros(self.k)
        
        for i in np.arange(self.k):        
            ind[i] = np.argmax(y[T[i]:T[i+1]]) + T[i]
                
        ind = ind.astype(int)
        self.τ = x[ind]

        self.emit_param()
        self.plotTCSPCSignal.emit(self.abs_time, self.rel_time)
    
    @pyqtSlot(np.ndarray)    
    def catch_psf(self, psffit_array):    
        print(datetime.now(), '[analysis] PSF fit received in backend.')
        self.PSF = np.zeros((self.k, psffit_array.shape[1], psffit_array.shape[2]))
        self.PSF = psffit_array
        self.read_parameterfile()
        
        self.fitPSFSignal.emit(self.PSF, self.x0, self.y0)
     
    def read_parameterfile(self):        
        fitparameter = os.path.splitext(self.psffilename)[0] + '-parameter.txt'
        
        config = configparser.ConfigParser()
        config.sections()
        config.read(fitparameter, encoding='ISO-8859-1')
        
        self.pxexp = float(config['PSF-fit parameter']['Experimental px size [nm]'])
        self.size = int(config['PSF-fit parameter']['x-/ y-size of saved PSF fit [px]'])
        self.PX = float(config['PSF-fit parameter']['Final px size of PSF fit [nm]'])
        self.k = int(config['PSF-fit parameter']['Number of fitted excitation donuts'])
        #update donut number in gui
        x0 = config['PSF-fit parameter']['x-coordinates of donut centers [px]']
        y0 = config['PSF-fit parameter']['y-coordinates of donut centers [px]']
        
        x0 = x0[1:-2].split('.')
        y0 = y0[1:-2].split('.')
        self.x0 = np.zeros(len(x0))
        self.y0 = np.zeros(len(y0))

        for i in range(len(x0)):
            self.x0[i] = int(x0[i])
            self.y0[i] = int(y0[i])
            
        self.emit_param()
        
    @pyqtSlot(np.ndarray)
    def find_position(self, ontimes):
        
        mode = self.tcspcMode

        print(datetime.now(), '[analysis] Position estimation performed in <', mode,  '> mode')
        
        ontimes = ontimes.flatten()
        
        #TODO: change to fixed number of windows and then show window length
        if mode == 'Nanoparticle':
            tiabs = ontimes[0]
            tfabs = ontimes[1]
            num_windows = int((tfabs - tiabs) / self.NP_binning)
            timeON = np.zeros(2*num_windows)
            for i in np.arange(0, 2*num_windows, 2):
                timeON[i] = tiabs
                tiabs = tiabs + 0.1
                timeON[i+1] = tiabs
            timeON[-1] = tfabs
            
        elif mode == 'Origami (manual)':
            timeON = ontimes
        
        elif mode == 'Origami (auto)':
            [T, bwt] = self.trace_seg(self.abs_time)
            #TODO: soft-code scaling factors
            bwt = 10*bwt
            T = 5*T
            nbinsopt = int((np.max(self.abs_time)/bwt))
            seqbin, times = np.histogram(self.abs_time, bins=nbinsopt)
            
            fig1, ax1 = plt.subplots()
            ax1.plot(times[0:-1],seqbin) 
            ax1.axhline(y=(T), color = 'r') 
            ax1.legend(['bin width = ' + str(bwt), 'T = ' + str(T)], loc='upper right')
            
            mask = seqbin>T
            indexes = np.argwhere(np.diff(mask)).squeeze()
            timeON = times[indexes]            
            print('Threshold:', T, 'Binwidth:', bwt)
        
        self.timeON = timeON
        number = len(timeON)
        t = {}
        absTimeON = {}
        self.relTimeON = {}
        
        for i in np.arange(0, number, 2):
            j = i//2
            t[i] = (timeON[i] < self.abs_time) & (self.abs_time < timeON[i+1])
            absTimeONp = self.abs_time * t[i]
            #changed fron relTime... to not exclude rel. times exactly equal to 0.000...
            absTimeON[j] = self.abs_time[np.nonzero(absTimeONp)]
            self.relTimeON[j] = self.rel_time[np.nonzero(absTimeONp)] 
            
        Nsegments = len(absTimeON)                
        Tot = Nsegments
        n = np.zeros((Tot, 4))
        self.pos = np.zeros((Tot, 2))
        indrec = np.zeros((Tot, 2))    
        self.N = np.zeros(Tot)
        deltaT = np.zeros(Tot)
                
        for i in np.arange(Tot):
            n[i, :] = self.n_minflux(self.relTimeON[i])
            self.N[i] = np.sum(n[i, :])
            
            #calculate time length of selected trace segment
            deltaT[i] = absTimeON[i][-1] - absTimeON[i][0]

            #check if above min. time length and photon threshold
            if deltaT[i] > self.deltaTmin:
                if self.N[i] > self.Nmin:
                    [indrec[i,:], self.pos[i, :], Ltot] = self.pos_minflux(n[i, :])
                else:
                    print(datetime.now(), '[analysis] Time window', str(i), ' neglected as it contains too few photons')
            else:
                print(datetime.now(), '[analysis] Time window', str(i), ' neglected as too short')
        
        self.positionSignal.emit(indrec, self.pos, Ltot, self.N)

    def n_minflux(self, reltimes_i):
        
        """
        Photon collection in a MINFLUX experiment
        (n0, n1, n2, n3)
        
        Inputs
        ----------
        τ : array, times of EBP pulses (1, k)
        rel_time : photon arrival times relative to sync (N)
        a : init of temporal window (in ns) aka from when after the donut exposure do we consider collected photons to origin from previous donut
        b : the temporal window length of EBP exposure (in ns) aka. Time window in which we suspect photon emission to only origin from previous donut exposure.
        k : number of donut excitations
        a, b can be adapted for different lifetimes
    
        Returns
        -------
        n : (1, k) array with photon numbers corresponding to each exposure
        
        """

        # total number of detected photons
        #n_tot = np.shape(relTime)[0]
    
        # number of photons in each exposition
        n = np.zeros(self.k)    
        for i in np.arange(self.k):
            ti = self.τ[i] + self.lifetime_win_i
            tf = self.τ[i] + self.lifetime_win_i + self.lifetime_win_f
            r = reltimes_i[(reltimes_i>ti) & (reltimes_i<tf)]
            n[i] = np.size(r)
            
        return n 
        
    def pos_minflux(self, n):
        """    
        MINFLUX position estimator (using ML)
        
        Inputs
        ----------
        n : array with photon numbers corresponding to each exposure
        psf : array with EBP (K x size x size)
        sbr : estimated (exp) signal to background ratio
    
        Returns
        -------
        indrec : position estimator in index coordinates (MLE)
        pos_estimator : position estimator (MLE)
        like_tot : Likelihood function
            
        """
        # FOV size
        norm_psf = np.sum(self.PSF, axis = 0)
        self.size = np.size(self.PSF, axis = 1)
        
        # probabilitiy vector 
        p = np.zeros((self.k, self.size, self.size))

        for i in np.arange(self.k):        
            p[i,:,:] = (self.sbr/(self.sbr + 1)) * self.PSF[i,:,:]/norm_psf + (1/(self.sbr + 1)) * (1/self.k)
    
    #    else:        
    #    for i in np.arange(K):
    #        p[i,:,:] = PSF[i,:,:]/normPSF
                    
        # likelihood function
        like = np.zeros((self.k, self.size, self.size))
        for i in np.arange(self.k):
            like[i, :, :] = n[i] * np.log(p[i, : , :])        
        like_tot = np.sum(like, axis = 0)
        
        # maximum likelihood estimator for the position    
        indrec = np.unravel_index(np.argmax(like_tot, axis=None), like_tot.shape)
        pos_estimator = self.index_to_space(indrec)
        
        return indrec, pos_estimator, like_tot
    
    def save_psffit(self):
         
        root = os.path.splitext(self.psffilename)[0]
        filename = root + '_fit'
        uname = tools.getUniqueName_New(filename, '.txt')
        parameter_filename = uname + '.txt'
        fit_filename = uname + '.tiff'
        
        data = np.array(self.PSF, dtype=np.float32)
        iio.mimwrite(fit_filename, data)
        
        config = configparser.ConfigParser()
    
        config['PSF-fit parameter'] = {
    
            'Date and time': datetime.now(),
            'Name of PSF source file': self.psffilename,
            'Experimental px size [nm]': str(self.pxexp),
            'x-/ y-size of saved psf fit [px]': str(self.size),
            'Final px size of PSF fit [nm]': str(self.PX),
            'Number of fitted excitation donuts': str(self.k),
            'x-coordinates of donut centers [nm]': str(self.x0),
            'y-coordinates of donut centers [nm]': str(self.y0)}
    
        with open(parameter_filename, 'w') as configfile:
            config.write(configfile)

        print(datetime.now(), '[analysis] Fitted PSF and parameter file saved')
        
    @pyqtSlot(bool)    
    def save_results(self, saveplot):
        
        root = os.path.splitext(self.psffilename)[0]
        filename = root + '_localization'
        uname = tools.getUniqueName_New(filename, '.txt')
        log_filename = uname + '.txt'
        plot_filename = uname + '.pdf'

        #### PLOT SECTION#####
        if saveplot:
            marker = ['yo', 'ro', 'ko', 'bo']
                    
            fig, ax = plt.subplots(1, 1)  
            ax.set_title('Np Au, <N> = ' + str(int(np.round(np.mean(self.N), decimals = 0))) , fontsize=16) 
            ax.set_aspect('equal', 'box')
            ax.set_xlabel('x [nm]')
            ax.set_ylabel('y [nm]')
            
            for i in np.arange(self.k):
                ax.plot(self.x0[i], self.y0[i], marker[i], markersize = 10)
                
            for i in np.arange(self.pos.shape[0]):
                ax.plot(self.pos[i,0], self.pos[i,1], marker='*', markersize = 5)
            
            plt.savefig(plot_filename, format='pdf', dpi=1000)
        
        ###  PARAMETER FILE####
        
        sigmax = np.round(np.std(self.pos[:, 0]), 1)
        sigmay = np.round(np.std(self.pos[:, 1]), 1)
        meanx = np.round(np.mean(self.pos[:, 0]), 1)
        meany = np.round(np.mean(self.pos[:, 1]), 1)
        Nmean = np.round(np.mean(self.N), 0)
        
        #configparser can save parameter description only in lower-case letters
        config = configparser.ConfigParser()
        config['Position Estimation Summary'] = {
    
            'Date and time': datetime.now(),
            'Name of psf source file': self.psffilename,
            'Name of tcspc source file': self.tcspcfilename,
            'Number of excitation donuts': str(self.k),
            'x-coordinates of donut centers [nm]': str(self.x0),
            'y-coordinates of donut centers [nm]': str(self.y0),
            'signal-noise ratio': str(self.sbr),
            'Timing of detected EBP pulses': self.τ, 
            'Lifetime init [ns]': str(self.lifetime_win_i),
            'Lifetime crop [ns]': str(self.lifetime_win_f),
            'Analysis mode': self.tcspcMode,
            'on-times': self.timeON,
            '<n>': str(Nmean),
            '<pos x> [nm]': str(meanx),
            '<pos y> [nm]': str(meany),
            '<std x> [nm]': str(sigmax),
            '<std y> [nm]': str(sigmay)}
    
        with open(log_filename, 'w') as configfile:
            config.write(configfile)

        print(datetime.now(), '[analysis] Saved Results')
        
    def find_lifetime(self):
        
        print('hello')
        self.fitarray = []
        for i in np.arange(len(self.relTimeON)):
            self.fitarray = np.concatenate((self.fitarray, self.relTimeON[i]))
        
        #TODO: delete later on
        print(self.fitarray)
        
        # inter-photon times histogram 
        #TODO: soft-code bin width
        #change self.bins to self.tcspc_binning to clean up
        [y, bin_edges] = np.histogram(self.fitarray, self.bins)
        x = np.ediff1d(bin_edges) + bin_edges[:-1]
        
        # double exponential fit of the ipt histogram
        def single_exp(t, a, b, offset):
            return (a * np.exp(b * t) + offset)
        
        lifetguess = -1.0
        offsetguess = 25
        aguess = 500
        
        p0 = [aguess, lifetguess, offsetguess]
        
        p, cov = opt.curve_fit(single_exp, x, y, p0)
        
        print(p, cov)
        
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.plot(x, single_exp(x, *p))
   
            
    def trace_seg(self, absTime):
        #TODO: NEEDS TO BE FULLY AUTOMATED
        """    
        Trace segmentation in case of blinking
        Find optimal bin width and threshold
        (Adapted from F.D. Stefani PhD Thesis)
            
        Inputs
        ----------
        absTime : time tags of detected photons (macro time)
    
        Returns
        -------
        bw : optimal bin width
        T : threshold
        
        Parameters
        ----------
        bwmin, bwmax : boundaries for bw
        Tamx : max threshold
           
        """
        # compute inter-photon times (in absTime units)
        ipt = np.ediff1d(absTime)
        size = len(ipt)
        binsipt = int(np.sqrt(size))
        # inter-photon times histogram 
        [y, bin_edges] = np.histogram(ipt, bins=binsipt)
        x = np.ediff1d(bin_edges) + bin_edges[:-1]
        
        ind = np.min(np.where(y<0.01)) 
        x = x[0:ind]
        y = y[0:ind]
        # double exponential fit of the ipt histogram
        def double_exp(t, a1, a2, b1, b2, offset):
            return a1 * np.exp(b1 * t) + a2 * np.exp(b2 * t) + offset
        
        p,cov = opt.curve_fit(double_exp, x, y)
        
        # ON and OFF parameters from fit
        Ion = -np.min(p[2:4])
        Ioff = -np.max(p[2:4])
        Aon = np.max(p[0:2])
        Aoff = np.min(p[0:2])
        
        fig, ax = plt.subplots()
        ax.plot(x, y)
        lifelabel = 'τ_on = ' + str(1/Ion*10**(6)) + ' +/- ' + str(1/cov[3, 3]*10**(6)) + ' ns'
        ax.plot(x, double_exp(x, *p), label = lifelabel)
        ax.set_xlabel('t [ms]')
        ax.set_ylabel('counts')
        print(lifelabel)
         
        # minimum bin width
#        bwmin = 2*(np.max(absTime)/len(absTime))  
#        ordermin = int(-np.log10(bwmin))
        
        # bw boundaries
        bwmin = 0.001
        bwmax = 0.005
        
        # iterations over bw and over threshold T    
        for i in np.arange(bwmin, bwmax, 0.0001):
      
            bwt = i     
            # average ON and OFF times
            Ton = Aon/(Ion*(1-np.exp(-Ion*bwt)))
            Toff = Aoff/(Ioff*(1-np.exp(-Ioff*bwt)))
            # bins on and off
            bon = Ton/bwt
            boff = Toff/bwt
            
    #        # maximum threshold for a given bw
    #        Tmax = Ion * bwt
            # Pon and Poff functions
            def P(n, I, b):
                
                a = (((I*b)**n)*np.exp(-I*b))
                b = np.math.factorial(n)
         
                r = (a/b)
                
                return r
            # maximum threshold for a given bw
            Tmax = 30000 #Tmax
            for j in np.arange(0, int(Tmax), 1):
            
                T = j 
                Pon = np.zeros(T+1)
                Poff = np.zeros(T+1)
                
                for n in np.arange(0, T+1, 1):
                    Pon[n] = P(n, Ion, bwt)
                    Poff[n] = P(n, Ioff, bwt)
                
                PonT = np.sum(Pon)
                PoffT = np.sum(Poff)
                # evaluate wrong bins for a given bw and T
                Wb = bon*PonT + boff*(1-PoffT)
    #            print(Wb)
    #        
                if Wb <0.0000001:
                    break
            if Wb <0.000001:
                break 
    
        return T, bwt
    
    def poly_func(self, grid, x0, y0, c00, c01, c02, c03, c04, c10, c11, c12, c13, c14, 
                  c20, c21, c22, c23, c24, c30, c31, c32, c33, c34,
                  c40, c41, c42, c43, c44):
        """    
        Polynomial function to fit PSFs.
        Uses built-in function polyval2d.
        
        Inputs
        ----------
        grid : x,y array
        cij : coefficients of the polynomial function
    
        Returns
        -------
        q : polynomial evaluated in grid.
        
        """
    
        x, y = grid
        c = np.array([[c00, c01, c02, c03, c04], [c10, c11, c12, c13, c14], 
                      [c20, c21, c22, c23, c24], [c30, c31, c32, c33, c34],
                      [c40, c41, c42, c43, c44]])
        q = np.polynomial.polynomial.polyval2d((x - x0), (y - y0), c)
    
        return q.ravel()
    
   
    def exp_func(self, x, a, b, c,d):
        """    
        Exponential function.
        
        Inputs
        ----------
        x : function argument
        a, b, c, d : coefficients of the exponential function
    
        Returns
        -------
        Function value corresponding to specified parameters and argument.
        
        """
        return a*np.exp(-(x-d)/b)+c
    
    def make_connection(self, frontend):
        
        frontend.paramSignal.connect(self.get_frontend_param)
        frontend.fitPSFSignal.connect(self.fit_psf)
        frontend.loadTCSPCSignal.connect(self.open_tcspc)
        frontend.estimatorSignal.connect(self.find_position)
        frontend.sendPSFfitSignal.connect(self.catch_psf)

        frontend.ui.pushButton_saveFit.clicked.connect(self.save_psffit)
        frontend.ui.SaveResultsButton.clicked.connect(lambda: self.save_results(frontend.ui.CreatepdfcheckBox.isChecked()))
        frontend.ui.TestpushButton.clicked.connect(self.find_lifetime)

if __name__ == '__main__':
    
    if not QtGui.QApplication.instance():
        app = QtGui.QApplication([])
    else:
        app = QtGui.QApplication.instance()

    #app.setStyle(QtGui.QStyleFactory.create('fusion'))
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    
    icon_path = r'pics/icon.jpg'
    app.setWindowIcon(QtGui.QIcon(icon_path))
    
    print(datetime.now(), '[analysis] Analysis module running')
    
    worker = Backend()    
    gui = Frontend()
    
    worker.make_connection(gui)
    gui.make_connection(worker)     

    gui.emit_param()
    worker.emit_param()
    
    gui.setWindowIcon(QtGui.QIcon(icon_path))
    gui.show() #Maximized()
    #gui.showFullScreen()
        
    #app.exec_()