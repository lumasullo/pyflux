# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 11:42:43 2018

@author: USUARIO
"""

import numpy as np
import picoharp
import Read_PTU
import os
import matplotlib.pyplot as plt

ph = picoharp.PicoHarp300()

ph.getLibraryVersion()

ph.open()
ph.initialize()
ph.setup()

hwinfo = self.getHardwareInfo()
print('Device: {}, Part No: {}, Hardware Version: {}'.format(*hwinfo))

ph.syncDivider = 4 # this parameter must be set such that the count rate at channel 0 (sync) is equal or lower than 10MHz
ph.resolution = 16 # desired resolution in ps
ph.offset = 0

print('Acquisition mode is T{}'.format(ph.mode))
print('Resolution set to {} ps'.format(ph.resolution))
print('Countrate at channel 0 is {} c/s'.format(ph.countrate(0)))
print('Countrate at channel 1 is {} c/s'.format(ph.countrate(1)))
print('Acquisition time is set to {} ms'.format(ph.tacq))

outputfilename = 'tttr_data.out'
ph.startTTTR(outputfilename)

########### This part of the code reads the data ###########

directory = os.getcwd()
os.chdir(directory)

filename = 'tttr_data.out'
inputfile = open(filename, "rb")

numRecords = ph.numRecords # number of records
globRes = 2.5e-8  # in ns, corresponds to sync @40 MHz
timeRes = ph.resolution * 1e-12 # time resolution in s

relTime, absTime = Read_PTU.readPT3(inputfile, numRecords)

inputfile.close()

relTime = relTime * timeRes # in real time units (s)
relTime = relTime * 1e9  # in (ns)

plt.hist(relTime, bins=300)
plt.xlabel('time (ns)')
plt.ylabel('ocurrences')

absTime = absTime * globRes * 1e9  # true time in (ns), 4 comes from syncDivider, 10 Mhz = 40 MHz / syncDivider 
absTime = absTime / 1e6 # in ms

plt.figure()
timetrace, time = np.histogram(absTime, bins=50) # timetrace with 10 ms bins

plt.plot(time[0:-1], timetrace)
plt.xlabel('time (ms)')
plt.ylabel('counts')