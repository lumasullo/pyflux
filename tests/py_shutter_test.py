# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 08:18:18 2019

@author: Lars Richter

Python script to control six shutters via ADwin board
"""

import numpy as np
import os

import drivers.ADwin as ADwin
import time

#%% load and initialize ADwin board

def setupDevice(adw):

    BTL = "ADwin11.btl"
    PROCESS_7 = "sixshutters.TB7"
    
    btl = adw.ADwindir + BTL
    adw.Boot(btl)

    currdir = os.getcwd()
    process_folder = os.path.join(currdir, "processes")

    process_7 = os.path.join(process_folder, PROCESS_7)
    adw.Load_Process(process_7)


#%%    
def toggle_shutter(num, val):
    
    num=num-1
    adw.Set_Par(53, num)
    
    if val is True:
        adw.Set_Par(52, 1)
        adw.Start_Process(7)
        print('Shutter', str(num+1), 'opened')
        
    if val is False:
        adw.Set_Par(52, 0)
        adw.Start_Process(7)
        print('Shutter', str(num+1), 'closed')
        
#%%
def test(n):
    for x in range(20):
        time.sleep(0.5)
        if (x % 2 == 0):
            toggle_shutter(n, True)
        else:
            toggle_shutter(n, False)
        print(x)
  
#%%      
def test_all():
    for x in range(20):
        time.sleep(0.5)
        if (x % 2 == 0):
            party(True)
        else:
            party(False)
        print(x)

#%%
def party(on):
    if on is True:
        for num in range(6):
            adw.Set_Par(53, num)
            adw.Set_Par(52, 1)
            adw.Start_Process(7) 
    if on is False:
        for num in range(6):
            adw.Set_Par(53, num)
            adw.Set_Par(52, 0)
            adw.Start_Process(7) 
        
#%%
DEVICENUMBER = 0x1
adw = ADwin.ADwin(DEVICENUMBER, 1)
setupDevice(adw)

#%%

import tools.tools as tools
tools.toggle_shutter(adw, 3, False)