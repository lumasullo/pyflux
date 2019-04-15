'<ADbasic Header, Headerversion 001.001>
' Process_Number                 = 5
' Initial_Processdelay           = 2000
' Eventsource                    = Timer
' Control_long_Delays_for_Stop   = No
' Priority                       = Low
' Priority_Low_Level             = 1
' Version                        = 1
' ADbasic_Version                = 6.2.0
' Optimize                       = Yes
' Optimize_Level                 = 2
' Stacksize                      = 1000
' Info_Last_Save                 = PC-MINFLUX  PC-MINFLUX\USUARIO
'<Header End>
'process 5: shutter by luciano a. masullo

'par_55: 0 -> shutter, 1 -> flipper

'par_50: 1 -> open shutter, 0 -> close shutter
'par_57: 1 -> proceed to shutter action (open/close), 0 -> wait until flag value is 1

'par_51: 1 -> open flipper, 0 -> close flipper
'par_58: 1 -> proceed to flipper action (open/close), 0 -> wait until flag value is 1

'README:
'Very simpe process to digitally control the Thorlabs shutter

#INCLUDE .\data-acquisition.inc

dim flag as long at dm_local

INIT:
 
  Rem Configure DIO00.DIO15 as inputs and DIO16.DIO31 as outputs
  Conf_DIO(1100b)

EVENT:
  
  if (par_55 = 0) then
  
    DO
      flag = par_57
    UNTIL (flag = 1)
  
    Digout(16, par_50)
  
    par_57 = 0
  
  endif

  if (par_55 = 1) then
  
    DO
      flag = par_58
    UNTIL (flag = 1)
  
    Digout(17, par_51)
  
    par_58 = 0

  endif
  
FINISH:
  
