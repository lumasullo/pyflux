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
'process 5: flipper by luciano a. masullo

'par_55: 0 -> shutter, 1 -> flipper
'par_50: 1 -> open shutter, 0 -> close shutter
'par_51: 1 -> open flipper, 0 -> close flipper


'README:
'Very simple process to digitally control the Thorlabs shutter

'The shutter part is not used because there is now a seperate script
'The flipper part is different because it needs rising edges to switch state

#INCLUDE .\data-acquisition.inc

dim flag as long at dm_local
dim time0, time1, wait_time as float at dm_local

INIT:
 
  Rem Configure DIO00.DIO15 as inputs and DIO16.DIO31 as outputs
  Conf_DIO(1100b)
  wait_time = 300000

EVENT:
  
  if (par_55 = 0) then
  
    Digout(16, par_50)
  
    End
  
  endif

  if (par_55 = 1) then

    'Digout(17, par_51)
    Digout(17, 1)
    
    time0 = Read_Timer() 'initial time of the wait event
  
    DO 
      time1 = Read_Timer()
        
    UNTIL (Abs(time1 - time0) > wait_time)
    
    Digout(17, 0)
    

    End

  endif
  
FINISH:
  
