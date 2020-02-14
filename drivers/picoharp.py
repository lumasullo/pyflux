# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:46:50 2018

@author: USUARIO
"""

import ctypes
from ctypes import byref, POINTER
from lantz import LibraryDriver
from lantz import Driver, Feat, Action
import time
from datetime import datetime
import numpy as np

LIB_VERSION = "3.0"
MAXDEVNUM = 8
MODE_T2 = 2
MODE_T3 = 3
TTREADMAX = 131072
FLAG_OVERFLOW = 0x0040
FLAG_FIFOFULL = 0x0003
DEV_NUM = 0     # device number, works for only 1 PH device


class PicoHarp300(LibraryDriver):
    
    LIBRARY_NAME = "phlib64.dll"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        #From phdefin.h
        
        # Variables to store information read from DLLs
        self.buffer = (ctypes.c_uint * TTREADMAX)()
        self.libVersion = ctypes.create_string_buffer(b"", 8)
        self.hwSerial = ctypes.create_string_buffer(b"", 8)
        self.hwPartno = ctypes.create_string_buffer(b"", 8)
        self.hwVersion = ctypes.create_string_buffer(b"", 8)
        self.hwModel = ctypes.create_string_buffer(b"", 16)
        self.errorString = ctypes.create_string_buffer(b"", 40)
        self.res = ctypes.c_double()
        self.countRate0 = ctypes.c_int()
        self.countRate1 = ctypes.c_int()
        self.flags = ctypes.c_int()
        self.nactual = ctypes.c_int()
        self.ctcDone = ctypes.c_int()
        self.warnings = ctypes.c_int()
        self.warningstext = ctypes.create_string_buffer(b"", 16384)
        
        # Measurement parameters, these are hardcoded since this is just a demo
        self.mode = MODE_T3 # set T2 or T3 here, observe suitable Syncdivider and Range!
        self.binning = 0 # you can change this, meaningful only in T3 mode
        self.offsetValue = 0 # you can change this, meaningful only in T3 mode
        self.tacq = 1000 # Measurement time in millisec, you can change this
        self.syncDiv = 1 # you can change this, observe mode! READ MANUAL!
        self.CFDZeroCross0 = 10 # you can change this (in mV)
        self.CFDLevel0 = 50 # you can change this (in mV)
        self.CFDZeroCross1 = 10 # you can change this (in mV)
        self.CFDLevel1 = 150 # you can change this (in mV)
        
        self.maxRes = 4 # max res of PicoHarp 300 in ps
        
#        print('Library version {}'.format(self.getLibraryVersion()))
#        self.open()
#        self.initialize()
#

    def getLibraryVersion(self):
        
        self.lib.PH_GetLibraryVersion(self.libVersion)
        
        return self.libVersion.value.decode("utf-8")
    
    @Action()
    def open(self):
        
        retcode = self.lib.PH_OpenDevice(ctypes.c_int(DEV_NUM), self.hwSerial)
        if retcode == 0:
            print(datetime.now(), "[picoharp 300]  device-number: %1d     S/N %s initialized" % (DEV_NUM, 
                                        self.hwSerial.value.decode("utf-8")))

        else:
            if retcode == -1: # ERROR_DEVICE_OPEN_FAIL
                print(datetime.now(), "[picoharp 300]  %1d     no device" % DEV_NUM)
            else:
                self.lib.PH_GetErrorString(self.errorString, 
                                           ctypes.c_int(retcode))
                
                print(datetime.now(), "[picoharp 300]  %1d     %s" % (DEV_NUM, 
                                        self.errorString.value.decode("utf8")))
        
    def getHardwareInfo(self):
        
        self.lib.PH_GetHardwareInfo(DEV_NUM, self.hwModel, self.hwPartno, 
                                    self.hwVersion)
        
        return [self.hwModel.value.decode("utf-8"),
               self.hwPartno.value.decode("utf-8"),
               self.hwVersion.value.decode("utf-8")]
               
    def setup_ph300(self):
        #setup ph300
        self.lib.PH_Calibrate(ctypes.c_int(DEV_NUM))
        
        self.lib.PH_SetSyncDiv(ctypes.c_int(DEV_NUM), 
                               ctypes.c_int(self.syncDivider))

        self.lib.PH_SetInputCFD(ctypes.c_int(DEV_NUM), ctypes.c_int(0),
                                ctypes.c_int(self.CFDLevel0), 
                                ctypes.c_int(self.CFDZeroCross0))

        self.lib.PH_SetInputCFD(ctypes.c_int(DEV_NUM), ctypes.c_int(1),
                                ctypes.c_int(self.CFDLevel1), 
                                ctypes.c_int(self.CFDZeroCross1))
        
        time.sleep(0.2)
        
    def setup_phr800(self):
        #seperate setup procedure for PHR800
        #start enabling routing
        self.lib.PH_EnableRouting(ctypes.c_int(DEV_NUM), ctypes.c_int(1))
            
        model = ctypes.create_string_buffer(b"", 8)
        version = ctypes.create_string_buffer(b"", 8)
        
        self.lib.PH_GetRouterVersion(ctypes.c_int(DEV_NUM), byref(model), byref(version))
        modelstr = model.value.decode("utf-8")
        
        if modelstr == 'PHR 800':
            # prepare all PHR800 channels
            for i in range(0, 4):
                #setup phr channels and their respective levels to -200 mV
                #no other channel level was working in order to setup the phr800
                #according to the official software a level of 1500 mV should be adequate ???
                self.lib.PH_SetPHR800Input(ctypes.c_int(DEV_NUM), ctypes.c_int(i), 
                                        ctypes.c_int(-200), 
                                        ctypes.c_int(0))
                self.lib.PH_SetPHR800CFD(ctypes.c_int(DEV_NUM), ctypes.c_int(i),
                                        ctypes.c_int(0), 
                                        ctypes.c_int(0))
        else:
            print(datetime.now(), '[picoharp 300] No PHR800 router connected!')
        
        time.sleep(0.2)
        
    @Feat 
    def binning(self):
        
        return self.binningValue
        
    @binning.setter 
    def binning(self, value):
        

        self.lib.PH_SetBinning(ctypes.c_int(DEV_NUM), 
                               ctypes.c_int(value))
        
        self.binningValue = value
        
    @Feat    
    def offset(self):
        
        return self.offsetValue
        
    @offset.setter
    def offset(self,value):
        
        self.lib.PH_SetOffset(ctypes.c_int(DEV_NUM), ctypes.c_int(value))
        self.offsetValue = value
        
    @Feat
    def resolution(self):
        
        self.lib.PH_GetResolution(ctypes.c_int(DEV_NUM), 
                                  byref(self.res))
        
        return self.res.value
    
    @resolution.setter
    def resolution(self, value):
        
        # calculation: resolution = maxRes * 2**binning

        self.binning = int(np.log(value/self.maxRes)/np.log(2))
        
        
        
    def countrate(self, channel):
        
        if channel == 0:
            
            self.lib.PH_GetCountRate(ctypes.c_int(DEV_NUM), ctypes.c_int(0), 
                                     byref(self.countRate0))
            value = self.countRate0.value
            
        if channel == 1:
            
            self.lib.PH_GetCountRate(ctypes.c_int(DEV_NUM), ctypes.c_int(1), 
                                     byref(self.countRate1))
            
            value = self.countRate1.value
        
        return value
     
    @Feat
    def syncDivider(self):
        
        return self.syncDiv
      
    @syncDivider.setter
    def syncDivider(self, value):
        
        self.lib.PH_SetSyncDiv(ctypes.c_int(DEV_NUM), ctypes.c_int(value))
        self.syncDiv = value
               
    def startTTTR(self, outputfilename):
        
        outputfile = open(outputfilename, "wb+") 
        progress = 0
       
        self.lib.PH_StartMeas(ctypes.c_int(DEV_NUM), ctypes.c_int(self.tacq))
        print(datetime.now(), '[picoharp 300] TCSPC measurement started')
        
        # save real time for correlating with confocal images for FLIM
        f = open(outputfilename + '_ref_time_tcspc', "w+")
        f.write(str(datetime.now()) + '\n')
        f.write(str(time.time()) + '\n')
        
        meas = True
        self.measure_state = 'measuring'
        
        while meas is True:
            self.lib.PH_GetFlags(ctypes.c_int(DEV_NUM), byref(self.flags))
            
            if self.flags.value & FLAG_FIFOFULL > 0:
                print(datetime.now(), "[picoharp 300] FiFo Overrun!")
                self.stopTTTR()
            
            
            self.lib.PH_ReadFiFo(ctypes.c_int(DEV_NUM), byref(self.buffer), 
                                 TTREADMAX, byref(self.nactual))
                
        
            if self.nactual.value > 0:
                print(datetime.now(), '[picoharp 300] current photon count:', self.nactual.value)
                # We could just iterate through our buffer with a for loop, however,
                # this is slow and might cause a FIFO overrun. So instead, we shrinken
                # the buffer to its appropriate length with array slicing, which gives
                # us a python list. This list then needs to be converted back into
                # a ctype array which can be written at once to the output file
                outputfile.write((ctypes.c_uint*self.nactual.value)(*self.buffer[0:self.nactual.value]))
                progress += self.nactual.value
#                sys.stdout.write("\rProgress:%9u" % progress)
#                sys.stdout.flush()
                
            else:
                self.lib.PH_CTCStatus(ctypes.c_int(DEV_NUM), byref(self.ctcDone))
                
                if self.ctcDone.value > 0: 
                    print(datetime.now(), "[picoharp 300] Done")
                    self.numRecords = progress
                    self.stopTTTR()
                    
                    # save real time for correlating with confocal images for FLIM
                    f.write(str(datetime.now()) + '\n')
                    f.write(str(time.time()) + '\n')
                    f.close()
                    
                    print(datetime.now(), '[picoharp 300] {} events recorded'.format(self.numRecords))
                    meas = False
                    self.measure_state = 'done'
                    
    def stopTTTR(self):
        
        self.lib.PH_StopMeas(ctypes.c_int(DEV_NUM))
       
    def initialize(self):
        
        self.lib.PH_Initialize(ctypes.c_int(DEV_NUM), ctypes.c_int(self.mode))
        
    def finalize(self):
        
        self.lib.PH_CloseDevice(ctypes.c_int(DEV_NUM))
        
