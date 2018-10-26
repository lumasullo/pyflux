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
'process moveto: moveto_xyz by luciano a. masullo

'par_11: number of x pixels
'par_12: number of y pixels

'fpar_21: x initial position
'fpar_22: y initial position

'fpar23: x final position
'fpar24: y final position

'fpar_25: pixeltime


#INCLUDE .\data-acquisition.inc

dim currentx, currenty, currentz as float at dm_local
dim setpointx, setpointy, setpointz as float at dm_local
dim dx, dy, dz as float at dm_local
dim Nx,Ny,Nz,p as long at dm_local
dim time0, time1 as float at dm_local


INIT:
  
  time0 = 0
  time1 = 0

  currentx = ADC(1)
  currenty = ADC(2)
  currentz = ADC(6)
  
  setpointx = fpar_23 
  setpointy = fpar_24
  setpointz = fpar_25
  
  if (setpointx > POSMAX) then setpointx = POSMAX 'check that set x position is not higher than POSMAX
  if (setpointx < POSMIN) then setpointx = POSMIN 'check that set x position is not lower than POSMIN
  
  if (setpointy > POSMAX) then setpointy = POSMAX 'check that set x position is not higher than POSMAX
  if (setpointy < POSMIN) then setpointy = POSMIN 'check that set x position is not lower than POSMIN
  
  if (setpointz > POSMAX) then setpointz = POSMAX 'check that set x position is not higher than POSMAX
  if (setpointz < POSMIN) then setpointz = POSMIN 'check that set x position is not lower than POSMIN
  
  Nx = par_21
  Ny = par_22
  Nz = par_23
 
  dx = (setpointx-currentx)/Nx
  dy = (setpointy-currenty)/Ny
  dz = (setpointz-currentz)/Nz
  

EVENT:

  time0 = Read_Timer() 
  DO 
    time1 = Read_Timer()
    
  UNTIL (Abs(time1 - time0) > fpar_26)
    
  currentx = currentx + dx
  currenty = currenty + dy
  currentz = currentz + dz
  
  if (currentx > POSMAX) then currentx = POSMAX 'check that set x position is not higher than POSMAX
  if (currentx < POSMIN) then currentx = POSMIN 'check that set x position is not lower than POSMIN
  
  if (currenty > POSMAX) then currenty = POSMAX 'check that set x position is not higher than POSMAX
  if (currenty < POSMIN) then currenty = POSMIN 'check that set x position is not lower than POSMIN
  
  if (currentz > POSMAX) then currentz = POSMAX 'check that set x position is not higher than POSMAX
  if (currentz < POSMIN) then currentz = POSMIN 'check that set x position is not lower than POSMIN

  DAC(1, currentx)
  DAC(2, currenty)
  DAC(6, currentz)
  
  p = 40 'error margin in ADwin units at which the moveTo is completed with currentpos = setpointpos
  
  if (((Abs(currentx - setpointx) < p) & (Abs(currenty - setpointy) < p)) & (Abs(currentz - setpointz) < p)) then
  
    currentx = setpointx
    currenty = setpointy
    currentz = setpointz
  
    if (currentx > POSMAX) then currentx = POSMAX 'check that set x position is not higher than POSMAX
    if (currentx < POSMIN) then currentx = POSMIN 'check that set x position is not lower than POSMIN
  
    if (currenty > POSMAX) then currenty = POSMAX 'check that set x position is not higher than POSMAX
    if (currenty < POSMIN) then currenty = POSMIN 'check that set x position is not lower than POSMIN
  
    if (currentz > POSMAX) then currentz = POSMAX 'check that set x position is not higher than POSMAX
    if (currentz < POSMIN) then currentz = POSMIN 'check that set x position is not lower than POSMIN

    DAC(1, currentx)
    DAC(2, currenty)
    DAC(6, currentz)
    
    End
   
  endif  
  

FINISH:
  

