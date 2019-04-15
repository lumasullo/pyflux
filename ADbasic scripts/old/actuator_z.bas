'<ADbasic Header, Headerversion 001.001>
' Process_Number                 = 3
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
'process actuator_z: actuator_z by luciano a. masullo

'actuator for focus lock feedback loop

'par_33 = number of pixels

'fpar_35 = setpoint z
'fpar_36: pixel time

'fpar_52 = currentz


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
  setpointz = fpar_35
  
  if (setpointz > POSMAX) then setpointz = POSMAX 'check that set x position is not higher than POSMAX
  if (setpointz < POSMIN) then setpointz = POSMIN 'check that set x position is not lower than POSMIN
  
  Nz = par_33

  dz = (setpointz-currentz)/Nz
  

EVENT:

  time0 = Read_Timer() 
  DO 
    time1 = Read_Timer()
    
  UNTIL (Abs(time1 - time0) > fpar_36)
    
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
    
    UNTIL (Abs(time1 - time0) > fpar_36)
 
    currentz = setpointz
  
    if (currentz > POSMAX) then currentz = POSMAX 'check that set x position is not higher than POSMAX
    if (currentz < POSMIN) then currentz = POSMIN 'check that set x position is not lower than POSMIN

    DAC(6, currentz)
    
    fpar_52 = currentz
    
    End
   
  endif  
  

FINISH:
  

