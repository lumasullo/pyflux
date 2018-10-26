'<ADbasic Header, Headerversion 001.001>
' Process_Number                 = 3
' Initial_Processdelay           = 2000
' Eventsource                    = Timer
' Control_long_Delays_for_Stop   = No
' Priority                       = High
' Version                        = 1
' ADbasic_Version                = 6.2.0
' Optimize                       = Yes
' Optimize_Level                 = 2
' Stacksize                      = 1000
' Info_Last_Save                 = USUARIO-PC  USUARIO-PC\USUARIO
'<Header End>
'process 3: linescan_x by luciano a. masullo
'derived from xyz-profile_c.bas

'par_1: number of x pixels

'par_30: digital input = 0 (APD), analog input = 1 (photodiode)

'fpar_1: x start position
'fpar_2: y position
'fpar_3: x increment per x step

'fpar_5: pixeltime (setpoint)
'fpar_6: pixeltime (measured)

'data_1: counter 1 (APD 1)

#INCLUDE .\data-acquisition.inc

dim co1, co1old, signal, ai_offset as long at dm_local
dim currentx, yposition as float at dm_local
dim time0, time1 as float at dm_local
dim pixcnt_x as long at dm_local
dim data_1[1024] as long as fifo

INIT:
  
  fifo_clear(1)
  
  ai_offset = 32768
  
  time0 = 0
  time1 = 0
  
  co1 = 0
  co1old = 0
  signal = 0
  
  pixcnt_x = 0
  
  currentx = fpar_1
  yposition = fpar_2
  
  DAC(1, currentx)
  DAC(2, yposition)
  
  start_counters()

EVENT:
  time0 = Read_Timer() 'initial time of the pixel event
  
  'delay to wait for a given pixeltime)
  DO 
    time1 = Read_Timer()
    
  UNTIL (Abs(time1 - time0) > fpar_5)

  if (par_30 = 0) then
    
    co1 = cnt_read(1) 'read counter
    data_1 = co1 - co1old 'calculate counts in the current pixel
    co1old = co1 'update co1old
    
  endif

  if (par_30 = 1) then
    
    signal = ADC(3) - ai_offset
    data_1 = signal

  endif
  
  if (pixcnt_x < par_1) then
  
    currentx = currentx + fpar_3 'set position to move in x (forward)
  
    if (currentx > POSMAX) then currentx = POSMAX 'check that set x position is not higher than POSMAX
    if (currentx < POSMIN) then currentx = POSMIN 'check that set x position is not lower than POSMIN
  
    DAC(1, currentx) 'move one step in x
  
    INC(pixcnt_x) 'update x pixel counter
  
    fpar_6 = Read_Timer() - time0 'measure real pixeltime
    
  endif
  
  if (pixcnt_x >= par_1) then 
    
    currentx = currentx - fpar_3 'set position to move in x (backward)
    
    if (currentx > POSMAX) then currentx = POSMAX 'check that set x position is not higher than POSMAX
    if (currentx < POSMIN) then currentx = POSMIN 'check that set x position is not lower than POSMIN
  
    DAC(1, currentx) 'move one step in x
  
    INC(pixcnt_x) 'update x pixel counter
      
    if (pixcnt_x >= 2 * par_1) then End
    
  endif
   
    

FINISH:

  pixcnt_x = 0 'reset x pixel counter
  
  stop_counters()
