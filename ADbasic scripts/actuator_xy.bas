'<ADbasic Header, Headerversion 001.001>
' Process_Number                 = 4
' Initial_Processdelay           = 3000
' Eventsource                    = Timer
' Control_long_Delays_for_Stop   = No
' Priority                       = Low
' Priority_Low_Level             = 1
' Version                        = 1
' ADbasic_Version                = 6.2.0
' Optimize                       = Yes
' Optimize_Level                 = 1
' Stacksize                      = 1000
' Info_Last_Save                 = PC-MINFLUX  PC-MINFLUX\USUARIO
'<Header End>
'process moveto: moveto_xy by luciano a. masullo

'Parameters from 40 to 49 are used

'function to actuate in an ON/OFF xy feedback loop.
'WARNING: the movement is not supposed to be smooth, therefore this script is to be used for small corrections ( < 200 nm)

'par_61: number of x pixels
'par_62: number of y pixels

'fpar63: setpoint x
'fpar64: setpoint y

'fpar_66: pixeltime
'par_67:  start/stop actuator flag

'fpar_50: keeps track of x position of the piezo
'fpar_51: keeps track of y position of the piezo

#INCLUDE .\data-acquisition.inc

dim currentx, currenty as float at dm_local
dim setpointx, setpointy as float at dm_local
dim flag as long at dm_local
dim time0, time1 as float at dm_local


INIT:
  
  time0 = 0
  time1 = 0

EVENT:
  
  'This loop holds the actuator until the flag is passed to start the actual function
  'DO
  '  flag = par_40
  'UNTIL (flag = 1)
 
  setpointx = fpar_40
  setpointy = fpar_41
  
  if (currentx > POSMAX) then currentx = POSMAX 'check that set x position is not higher than POSMAX
  if (currentx < POSMIN) then currentx = POSMIN 'check that set x position is not lower than POSMIN
  
  if (currenty > POSMAX) then currenty = POSMAX 'check that set x position is not higher than POSMAX
  if (currenty < POSMIN) then currenty = POSMIN 'check that set x position is not lower than POSMIN

  currentx = setpointx
  currenty = setpointy
  
  DAC(1, currentx)
  DAC(2, currenty)
  
  fpar_70 = currentx
  fpar_71 = currenty

  par_40 = 0
    
  time0 = Read_Timer() 
  DO 
    time1 = Read_Timer()
    
  UNTIL (Abs(time1 - time0) > fpar_46)
  
   
FINISH:
  

