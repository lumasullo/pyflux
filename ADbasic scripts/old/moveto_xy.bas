'<ADbasic Header, Headerversion 001.001>
' Process_Number                 = 4
' Initial_Processdelay           = 3000
' Eventsource                    = Timer
' Control_long_Delays_for_Stop   = No
' Priority                       = High
' Version                        = 1
' ADbasic_Version                = 6.2.0
' Optimize                       = Yes
' Optimize_Level                 = 1
' Stacksize                      = 1000
' Info_Last_Save                 = PC-MINFLUX  PC-MINFLUX\USUARIO
'<Header End>
'process moveto: moveto_xy by luciano a. masullo

'function to do a single (smooth) movement to desired position

'par_41: number of x pixels
'par_42: number of y pixels

'fpar43: setpoint x
'fpar44: setpoint y

'fpar_46: pixeltime

'fpar_50: keeps track of x position of the piezo
'fpar_51: keeps track of y position of the piezo

#INCLUDE .\data-acquisition.inc

dim currentx, currenty as float at dm_local
dim setpointx, setpointy as float at dm_local
dim dx, dy as float at dm_local
dim Nx,Ny,p as long at dm_local
dim time0, time1 as float at dm_local


INIT:
  
  time0 = 0
  time1 = 0

  currentx = fpar_50
  currenty = fpar_51
  
  setpointx = fpar_43 
  setpointy = fpar_44
  
  if (setpointx > POSMAX) then setpointx = POSMAX 'check that set x position is not higher than POSMAX
  if (setpointx < POSMIN) then setpointx = POSMIN 'check that set x position is not lower than POSMIN
  
  if (setpointy > POSMAX) then setpointy = POSMAX 'check that set x position is not higher than POSMAX
  if (setpointy < POSMIN) then setpointy = POSMIN 'check that set x position is not lower than POSMIN
  
  Nx = par_41
  Ny = par_42
 
  dx = (setpointx-currentx)/Nx
  dy = (setpointy-currenty)/Ny

EVENT:

  time0 = Read_Timer() 
  DO 
    time1 = Read_Timer()
    
  UNTIL (Abs(time1 - time0) > fpar_46)
    
  currentx = currentx + dx
  currenty = currenty + dy
  
  if (currentx > POSMAX) then currentx = POSMAX 'check that set x position is not higher than POSMAX
  if (currentx < POSMIN) then currentx = POSMIN 'check that set x position is not lower than POSMIN
  
  if (currenty > POSMAX) then currenty = POSMAX 'check that set x position is not higher than POSMAX
  if (currenty < POSMIN) then currenty = POSMIN 'check that set x position is not lower than POSMIN

  DAC(1, currentx)
  DAC(2, currenty)
  
  fpar_50 = currentx
  fpar_51 = currenty
  
  p = 1 'error margin in ADwin units at which the moveTo is completed with currentpos = setpointpos
  
  if (((Abs(currentx - setpointx) < p) & (Abs(currenty - setpointy) < p))) then
    
    time0 = Read_Timer() 
    DO 
      time1 = Read_Timer()
    
    UNTIL (Abs(time1 - time0) > fpar_46)
  
    currentx = setpointx
    currenty = setpointy
  
    if (currentx > POSMAX) then currentx = POSMAX 'check that set x position is not higher than POSMAX
    if (currentx < POSMIN) then currentx = POSMIN 'check that set x position is not lower than POSMIN
  
    if (currenty > POSMAX) then currenty = POSMAX 'check that set x position is not higher than POSMAX
    if (currenty < POSMIN) then currenty = POSMIN 'check that set x position is not lower than POSMIN

    DAC(1, currentx)
    DAC(2, currenty)
    
    fpar_50 = currentx
    fpar_51 = currenty
    
    End
   
  endif  
  

FINISH:
  

