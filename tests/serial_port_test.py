# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:01:41 2020

@author: USUARIO
"""

import serial.tools.list_ports
from drivers.minilasevo import MiniLasEvo

#%%
ports = list(serial.tools.list_ports_windows.comports())
for p in ports:
    print(p)
    if 'USB' in str(p):
        print('yes')
        try:
            minilaser = MiniLasEvo(str(p)[0:4])
            print(minilaser.status())
        except:
            print('failed')
            pass
        


