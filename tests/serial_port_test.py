# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:01:41 2020

@author: USUARIO
"""

import serial.tools.list_ports
import sys

#%%
ports = list(serial.tools.list_ports.comports())
for p in ports:
    print(p.hwid)
    print(p.device)
#    if 'USB' in str(p):
#        print('yes')
#        try:
#            minilaser = MiniLasEvo(str(p)[0:4])
#            print('succes')
#            #print(minilaser.status())
#        except:
#            print('failed')
        


#%%
            
import win32com.client

i = 1
j = 1
wmi = win32com.client.GetObject ("winmgmts:")
for usb in wmi.InstancesOf ("Win32_SerialPort"):
    strid = usb.DeviceID
    print(strid)
    if ('ML069719' in strid):
        savei = i
        
    if ('VID_0403&PID_6001' in strid):
        savej = j
    i+= 1
    j+= 1
    
if savei<savej:
    port = 'COM3'
else:
    port = 'COM7'

print(port)

#%%
import win32com.client

wmi = win32com.client.GetObject ("winmgmts:")
for usb in wmi.InstancesOf ("Win32_SerialPort"):
    print(usb.DeviceID)