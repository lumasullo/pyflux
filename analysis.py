# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:05:18 2019

@author: Lars Richter

GUI for MINFLUX analysis and simulations

conda command for converting QtDesigner file to .py:
pyuic5 -x AnalysisDesign.ui -o AnalysisDesign.py

Next steps:
    Write and load x0 and y0 correctly --> use configparser in tools
    Save plots and result summary file
    Check whether code works for non-zero tcscpc time-window
    Save and load tcspc-windows 
    ... 

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

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot

import tools.viewbox_tools as viewbox_tools
import tools.colormaps as cmaps
import gui.AnalysisDesign
import qdarkstyle

π = np.pi

# see https://stackoverflow.com/questions/1551605
# /how-to-set-applications-taskbar-icon-in-windows-7/1552105#1552105
# to understand why you need the preceeding two lines
myappid = 'mycompany.myproduct.subproduct.version' # arbitrary string
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

class Frontend(QtGui.QMainWindow):
    
    paramSignal = pyqtSignal(dict)
    closeSignal = pyqtSignal()
    fitPSFSignal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    loadTCSPCSignal = pyqtSignal()
    estimatorSignal = pyqtSignal()
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
        
    def emit_param(self):
        params = dict()

        params['psfFilename'] = self.ui.psfFileEditBox.text()
        params['tcspcFilename'] = self.ui.tcspcEditBox.text()
        params['numDonuts'] = int(self.ui.spinBox_donuts.text())
        params['SBR'] = float(self.ui.lineEdit_sbr.text())
        params['NP_binning_window'] = float(self.ui.lineEdit_winlen.text())
        params['lifetime_win_i'] = float(self.ui.lineEdit_lifetimewin_i.text())
        params['lifetime_win_f'] = float(self.ui.lineEdit_lifetimewin_f.text())
        
        if (len(self.ontimes) != 0):
            params['trace_window_i'] = self.ontimes[0][0]
            params['trace_window_f'] = self.ontimes[0][1]
        
        self.paramSignal.emit(params)

    @pyqtSlot(dict)
    def get_backend_param(self, params):
        
        self.PX = params['pxSize']
        
    
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
        self.psfViewBox = imageWidget.addViewBox(row=0, col=0)
        self.psfViewBox.setAspectLocked(True)
        self.psfViewBox.setMouseMode(pg.ViewBox.RectMode)

        img = pg.ImageItem(self.current_images[image_number,:,:]) #TODO:edit later on 
        self.psfViewBox.clear()
        self.psfViewBox.addItem(img)
        
        hist = pg.HistogramLUTItem(image=img)   # set up histogram for the liveview image
        lut = viewbox_tools.generatePgColormap(cmaps.inferno)
        hist.gradient.setColorMap(lut)
        hist.vb.setLimits(yMin=0, yMax=10000) #TODO: check maximum value
        for tick in hist.gradient.ticks:
            tick.hide()
        imageWidget.addItem(hist, row=0, col=1)

        self.empty_layout(self.ui.psfLayout)        
        self.ui.psfLayout.addWidget(imageWidget)
    
    def fit_exppsf(self):
        #room for reading analysis parameter
        self.k = int(self.ui.spinBox_donuts.text())
        self.emit_param()
        print(datetime.now(), '[analysis] PSF fitting started')
        self.x0 = np.zeros(self.k)
        self.y0 = self.x0
        self.fitPSFSignal.emit(self.img_array, self.x0, self.y0)

    #TODO: create one function for adding psf images independent of being fit or raw
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
        
        dataWidget = pg.GraphicsLayoutWidget()
        dataWidget.clear()
        
        histPlot = dataWidget.addPlot(row=1, col=0, title="relative time histogram")
        histPlot.setLabels(bottom=('ns'),
                                left=('counts'))
        
        self.tracePlot = dataWidget.addPlot(row=2, col=0, title="Time trace")
        self.tracePlot.setLabels(bottom=('ms'),
                                left=('counts'))
        
        counts, bin_edges = np.histogram(rel_time, bins=100) # TODO: choose proper binning, from fernandos fct
        x = np.ediff1d(bin_edges) + bin_edges[:-1]
        histPlot.plot(x, counts)

        counts, bin_edges = np.histogram(abs_time, bins=100) # timetrace with 50 bins
        x = np.ediff1d(bin_edges) + bin_edges[:-1]
        self.tracePlot.plot(x, counts)
        
        self.empty_layout(self.ui.tcspcLayout)
        self.ui.tcspcLayout.addWidget(dataWidget)
        
        self.region_selection()
        
        print(datetime.now(), '[analysis] TCSPC received and plotted')
        
    def check_tcspcmode(self, state):
        if state:
            if self.ui.radioButton_NP.isChecked():
                self.ui.comLabel.setText('Select macrotime window in plot and set window length for binning.')
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
        self.loadTCSPCSignal.emit()

    @pyqtSlot()
    def position_estimation(self):
        #TODO: read in parameter
        self.ontimes = []
        self.read_ontimes()
        
        if self.ui.radioButton_psffit.isChecked():
            #TODO load parameter file
            #self.x0
            #self.y0
            self.sendPSFfitSignal.emit(self.current_images)
            
        self.emit_param()
        self.estimatorSignal.emit()
        
    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def plot_position(self, indrec, pos, Ltot):
#https://stackoverflow.com/questions/41060163/pyqtgraph-scatterplotitem-setbrush

        #do some statistics
        window = float(self.ui.lineEdit_winlen.text())
        tiabs = self.ontimes[0][0]
        ti = int(tiabs/window) # in window units
        tfabs = self.ontimes[0][1]
        tf = int(tfabs/window)
        
        sigmax = np.round(np.std(pos[ti:tf, 0]), decimals = 1)
        sigmay = np.round(np.std(pos[ti:tf, 1]), decimals = 1)
        #Nmean[j] = np.int(N[(np.nonzero(N))].mean())
        meanx = np.mean(pos[ti:tf, 0])
        meany = np.mean(pos[ti:tf, 1])
        sx = 'σx = '+ str(sigmax) + ' nm'
        sy = 'σy = '+ str(sigmay) + ' nm'  

        #Nm = '<N> =' + str(Nmean[j])
        
        #start actual plotting
        estimatorWidget = pg.GraphicsLayoutWidget()
        estimatorWidget.clear()
        plot = estimatorWidget.addPlot(title="Position estimation")
        plot.setLabels(bottom=('x [nm]'), left=('y [nm]'))
        
        for i in np.arange(pos.shape[0]):
            if pos[i,0] == 0:
                continue
            pos_i = pg.ScatterPlotItem([pos[i, 0]], [pos[i, 1]], size=10, symbol='x', pen=pg.mkPen(None))
            plot.addItem(pos_i)
       
        #TODO: undo comment
        # for i in np.arange(self.x0.shape[0]):
        #     donut_i = pg.ScatterPlotItem([self.x0[i]], [self.y0[i]], size=20, pen=pg.mkPen(None))
        #     donut_i.setBrush(QtGui.QBrush(QtGui.QColor(QtCore.qrand() % 256, QtCore.qrand() % 256, QtCore.qrand() % 256)))
        #     plot.addItem(donut_i)
            
        self.empty_layout(self.ui.estimateLayout)
        self.ui.estimateLayout.addWidget(estimatorWidget)
        
        text = '[analysis] Position estimation done \n' + str(meanx) + '\n' + str(meany) + '\n' + sx + '\n' + sy 
        self.ui.comLabel.setText(text)
        #TODO: add options to save plots and result summary in txt file, also save trace length and binning in logfile, plus name of psffile plus name of tcspc file

    def empty_layout(self, layout):
        for i in reversed(range(layout.count())): 
            layout.itemAt(i).widget().setParent(None)
            
    def make_connection(self, backend):
        
        backend.paramSignal.connect(self.get_backend_param)
        backend.fitPSFSignal.connect(self.plot_psffit)
        backend.plotTCSPCSignal.connect(self.plot_tcspc)
        backend.positionSignal.connect(self.plot_position)
        
    def closeEvent(self, *args, **kwargs):
        
        super().closeEvent(*args, **kwargs)
        self.closeSignal.emit()
        print(datetime.now(), '[analysis] Analysis module closed')
        app.quit()
        

class Backend(QtCore.QObject):

    paramSignal = pyqtSignal(dict)
    plotTCSPCSignal = pyqtSignal(np.ndarray, np.ndarray)
    fitPSFSignal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    estimatorSignal = pyqtSignal()
    positionSignal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)

    
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

    def emit_param(self):
        
        params = dict()
        
        params['pxSize'] = self.PX
        params['absTimeConv'] = self.ABS_TIME_CONVERSION
        params['fitpsfSize'] = self.size
        params['pxCamera'] = self.pxexp
        
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
        try:
            self.trace_window_i = params['trace_window_i']
            self.trace_window_f = params['trace_window_f']
        except:
            pass

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
        x0 = np.zeros(self.k)
        y0 = np.zeros(self.k)
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
            x0[i] = x[ind]
            y0[i] = y[ind]
            self.index[i,:] = ind
            print(datetime.now(), '[analysis]', str(i+1), '/', str(self.k), ' donuts fitted.')
            
        psfTc[0,:,:] = self.PSF[0, :, :]
        self.PSF[0, :, :] = self.PSF[1, :, :]
        self.PSF[1, :, :] = psfTc[0,:,:]
        psfTc[0,:,:] = self.PSF[2, :, :]
        self.PSF[2, :, :] = self.PSF[3, :, :]
        self.PSF[3, :, :] = psfTc[0,:,:]
        
        self.fitPSFSignal.emit(self.PSF, x0, y0)


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
        [y, bin_edges] = np.histogram(self.rel_time, bins=100)
        x = np.ediff1d(bin_edges) + bin_edges[:-1]
        
        T = len(y)//(self.k) * np.arange((self.k+1))
        
        ind = np.zeros(self.k)
        
        for i in np.arange(self.k):        
            ind[i] = np.argmax(y[T[i]:T[i+1]]) + T[i]
                
        
        ind = ind.astype(int)
        self.τ = x[ind]

        self.plotTCSPCSignal.emit(self.abs_time, self.rel_time)
    
    @pyqtSlot(np.ndarray)    
    def catch_psf(self, psffit_array):    
        print(datetime.now(), '[analysis] PSF fit received in backend.')
        self.PSF = np.zeros((self.k, psffit_array.shape[1], psffit_array.shape[2]))
        self.PSF = psffit_array

        print(self.PSF.shape)
        
    @pyqtSlot()
    def find_position(self):
        
        windows = [self.NP_binning]
        nwin = len(windows)
        
        for j in np.arange(nwin):
            window = windows[j] # 1s 
            nwindows = int(np.max(self.abs_time)//window)
            Len = int(len(self.rel_time)//np.max(self.abs_time//window))*np.arange(nwindows+1)
            reltimes = np.reshape(self.rel_time[0:Len[nwindows]], (nwindows, Len[1]))
        
            Tot=len(Len) - 1
            n = np.zeros((Tot, 4))
            pos = np.zeros((Tot, 2))
            indrec = np.zeros((Tot, 2))
        
            N = np.zeros(Tot)
            
            tiabs = self.trace_window_i
            ti = int(tiabs/window) # in window units
            tfabs = self.trace_window_f
            tf = int(tfabs/window)
            
        
            for i in np.arange(ti, tf):
                n[i, :] = self.n_minflux(reltimes[i])
                [indrec[i,:], pos[i, :], Ltot] = self.pos_minflux(n[i, :])
                N[i] = np.sum(n[i, :])
 
            self.positionSignal.emit(indrec, pos, Ltot)
    
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
         
        #TODO: save fit parameter in seperate file
        root = os.path.splitext(self.psffilename)[0]
        
        #TODO: undo comment
        # fit_filename = root + '_fit.tiff'
        # data = np.array(self.PSF, dtype=np.float32)
        # iio.mimwrite(fit_filename, data)
        
        parameter_filename = root + '_fit-parameter.txt'
        #file size and enconding, numbers of donuts fitted, x0, y0
                
        description = np.array(['Name of fitted PSF file:',
                                self.psffilename,
                                'Experimental px size in nm:',
                                'x-/ y-size of saved PSF fit in px:',
                                'Final px size of PSF fit in nm:',
                                'Number of fitted excitation donuts:',
                                'x0, y0', '', '', '', ''])
        
        self.x0 = np.zeros(self.k)
        values = np.array([None, None, self.pxexp, self.size, self.PX, self.k, None])
        values = np.append(values, self.x0)
        
        parameter = np.zeros(description.size, dtype=[('col1', 'U100'), ('col2', float)])
        parameter['col1'] = description
        parameter['col2'] = values
        
        np.savetxt(parameter_filename, parameter.T, fmt='%s %f')

        print(datetime.now(), '[analysis] Fitted PSF and parameter file saved')
        
    
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

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    app.setStyle(QtGui.QStyleFactory.create('fusion'))
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
        
    sys.exit(app.exec_())