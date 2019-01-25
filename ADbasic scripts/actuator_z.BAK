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
' Info_Last_Save                 = PC-MINFLUX  PC-MINFLUX\USUARIO
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

dim currentz as float at dm_local
dim setpointz as float at dm_local
dim dz as float at dm_local
dim Nz,p as long at dm_local
dim time0, time1 as float at dm_local


INIT:
  
  time0 = 0
  time1 = 0

  currentz = fpar_52
  
  setpointz = fpar_25
  
  if (setpointz > POSMAX) then setpointz = POSMAX 'check that set x position is not higher than POSMAX
  if (setpointz < POSMIN) then setpointz = POSMIN 'check that set x position is not lower than POSMIN
  
  Nz = par_23

  dz = (setpointz-currentz)/Nz
  

EVENT:

  time0 = Read_Timer() 
  DO 
    time1 = Read_Timer()
    
  UNTIL (Abs(time1 - time0) > fpar_26)
    
  currentz = currentz + dz
  
  if (currentz > POSMAX) then currentz = POSMAX 'check that set x position is not higher than POSMAX
  if (currentz < POSMIN) then currentz = POSMIN 'check that set x position is not lower than POSMIN

  DAC(6, currentz)

  fpar_52 = currentz
  
  p = 1 'error margin in ADwin units at which the moveTo is completed with currentpos = setpointpos
  
  if (Abs(currentz - setpointz) < p) then
    
    time0 = Read_Timer() 
    DO 
      time1 = Read_Timer()
    
    UNTIL (Abs(time1 - time0) > fpar_26)
 
    currentz = setpointz
  
    if (currentz > POSMAX) then currentz = POSMAX 'check that set x position is not higher than POSMAX
    if (currentz < POSMIN) then currentz = POSMIN 'check that set x position is not lower than POSMIN

    DAC(6, currentz)
    
    fpar_52 = currentz
    
    End
   
  endif  
  

FINISH:
  

