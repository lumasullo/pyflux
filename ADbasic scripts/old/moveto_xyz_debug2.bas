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

'fpar_21: x initial position
'fpar_22: y initial position

'fpar23: x final position
'fpar24: y final position

'fpar_25: pixeltime


#INCLUDE .\data-acquisition.inc

dim currentx, currenty as float at dm_local
dim setpointx, setpointy as float at dm_local
dim dx, dy as float at dm_local
dim Nx,Ny,p as long at dm_local
dim time0, time1 as float at dm_local



INIT:
  
  time0 = 0
  time1 = 0

  currentx = ADC(1)
  currenty = ADC(2)
  
  fpar_8 = currentx
  fpar_9 = currenty
  
  setpointx = fpar_23 
  setpointy = fpar_24
  
  Nx = par_21
  Ny = par_22
 
  dx = (setpointx-currentx)/Nx
  dy = (setpointy-currenty)/Ny
  
  fpar_34 = dx
  fpar_35 = dy

EVENT:

  time0 = Read_Timer() 
  DO 
    time1 = Read_Timer()
    
  UNTIL (Abs(time1 - time0) > fpar_26)
    
  currentx = currentx + dx
  currenty = currenty + dy
  
  if (currentx > POSMAX) then currentx = POSMAX 'check that set x position is not higher than POSMAX
  if (currentx < POSMIN) then currentx = POSMIN 'check that set x position is not lower than POSMIN
  
  if (currenty > POSMAX) then currenty = POSMAX 'check that set x position is not higher than POSMAX
  if (currenty < POSMIN) then currenty = POSMIN 'check that set x position is not lower than POSMIN
  
  fpar_1 = currentx
  fpar_2 = currenty

  DAC(1, currentx)
  DAC(2, currenty)
  
  p = 40
  
  if ((Abs(currentx - setpointx) < p) & (Abs(currenty - setpointy) < p)) then End
  'if ((currentx = setpointx) & (currenty = setpointy)) then End
  

FINISH:
  

