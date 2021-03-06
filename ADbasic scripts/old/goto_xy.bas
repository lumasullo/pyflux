'<ADbasic Header, Headerversion 001.001>
' Process_Number                 = 2
' Initial_Processdelay           = 3000
' Eventsource                    = Timer
' Control_long_Delays_for_Stop   = No
' Priority                       = High
' Version                        = 1
' ADbasic_Version                = 6.2.0
' Optimize                       = Yes
' Optimize_Level                 = 1
' Stacksize                      = 1000
' Info_Last_Save                 = USUARIO-PC  USUARIO-PC\USUARIO
'<Header End>
'process goto: goto_xy by luciano a. masullo

'par_11: number of x pixels
'par_12: number of y pixels

'fpar_11: x initial position
'fpar_12: y initial position

'fpar13: x final position
'fpar14: y final position

'fpar_15: pixeltime


#INCLUDE .\data-acquisition.inc

dim currentx, currenty as float at dm_local
dim setpointx, setpointy as float at dm_local
dim dx, dy as float at dm_local
dim Nx,Ny as long at dm_local
dim time0, time1 as float at dm_local

INIT:
  
  time0 = 0
  time1 = 0

  currentx = ADC(1)
  currenty = ADC(2)
  
  setpointx = fpar_13 
  setpointy = fpar_14
  
  Nx = par_11
  Ny = par_12
 
  dx = (setpointx-currentx)/Nx
  dy = (setpointy-currenty)/Ny
  

EVENT:

  time0 = Read_Timer() 
  DO 
    time1 = Read_Timer()
    
  UNTIL (Abs(time1 - time0) > fpar_15)
    
  currentx = currentx + dx
  currenty = currenty + dy

  DAC(1, currentx)
  DAC(2, currenty)
  
  fpar_11 = currentx
  fpar_12 = currenty
  
  if ((currentx = setpointx) & (currenty = setpointy)) then End

FINISH:
  

