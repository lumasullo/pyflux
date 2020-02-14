'<ADbasic Header, Headerversion 001.001>
' Process_Number                 = 1
' Initial_Processdelay           = 2000
' Eventsource                    = Timer
' Control_long_Delays_for_Stop   = No
' Priority                       = High
' Version                        = 1
' ADbasic_Version                = 6.2.0
' Optimize                       = Yes
' Optimize_Level                 = 2
' Stacksize                      = 1000
' Info_Last_Save                 = PC-MINFLUX  PC-MINFLUX\USUARIO
' Foldings                       = 100,107,115
'<Header End>
'process 1: linescan_x by luciano a. masullo

' last update:  
' 30.01.2020  JK  time variables as LONG --> changes of ADwin support guy

'Parameters from 1 to 19 are used

'function to do a line scan (+ one-step movement in slow axis)
'x --> fast axis
'y --> slow axis

'par_1: number of total elements in data arrays
'par_2: 0 flag meaning process started (0) or finished (1)
'par_3: digital input = 0 (APD), analog input = 1 (photodiode)

'fpar_2: y_offset
'fpar_6: scan pixeltime (measured)
'fpar_9: pixel time (setpoint

'fpar_10: 1 = x, 2 = y, 6 = z
'fpar_11: 1 = x, 2 = y, 6 = z

'fpar_70: keeps track of x position of the piezo
'fpar_71: keeps track of y position of the piezo
'fpar_72: keeps track of z position of the piezo

'data_1: counter 1 (APD 1)
'data_2: time array
'data_3: position array (fast axis)
'data_4: position array (slow axis)

#INCLUDE .\data-acquisition.inc

dim co1, co1old, signal, ai_offset as long at dm_local
dim x, y, z, currentx, currenty, y_offset as float at dm_local
dim time0, time1, px_time as long ' JK1
dim time_delta as long ' JK1
dim i as long at dm_local
dim data_1[1024] as long as fifo
dim DATA_2[8192] as long 
dim DATA_3[8192] as long 
dim DATA_4[8192] as long 

INIT:

  par_2 = 0 'flag meaning process started
  
  fifo_clear(1)
  
  ai_offset = 32768
  
  time0 = 0
  time1 = 0
  
  co1 = 0
  co1old = 0
  signal = 0
  
  i = 1
  
  y_offset = fpar_2
  
  '  if (yposition > POSMAX) then yposition = POSMAX 'check that set x position is not higher than POSMAX
  '  if (yposition < POSMIN) then yposition = POSMIN 'check that set x position is not lower than POSMIN
  '  
  '  DAC(2, yposition) 
  
  start_counters()

EVENT:
  
  'select position and move
  
  currentx = data_3[i] 
  currenty = data_4[i] + y_offset
 
  if (currentx > POSMAX) then currentx = POSMAX 'check that set x position is not higher than POSMAX
  if (currentx < POSMIN) then currentx = POSMIN 'check that set x position is not lower than POSMIN
  
  if (currenty > POSMAX) then currenty = POSMAX 'check that set x position is not higher than POSMAX
  if (currenty < POSMIN) then currenty = POSMIN 'check that set x position is not lower than POSMIN
  
  DAC(fpar_10, currentx) 'move one step in x
  DAC(fpar_11, currenty) 'y position (typically constant after intial dy during aux time)'
  
  if (fpar_10 = 1) then fpar_70 = currentx
  if (fpar_10 = 2) then fpar_71 = currentx
  
  if (fpar_11 = 2) then fpar_71 = currenty
  if (fpar_11 = 6) then fpar_72 = currenty
  
  'wait for the given pixel time
  
  px_time = data_2[i+1] - data_2[i]
  
  fpar_9 = px_time
  
  time0 = Read_Timer() 'initial time of the pixel event
  
  DO 
    time1 = Read_Timer()
    
  UNTIL (Abs(time1 - time0) > px_time)

  'read the counter or analog input
  
  if (par_3 = 0) then
    
    co1 = cnt_read(1) 'read counter
    data_1 = co1 - co1old 'calculate counts in the current pixel
    co1old = co1 'update co1old
    
  endif

  if (par_3 = 1) then
    
    signal = ADC(3) - ai_offset
    data_1 = signal

  endif
  
  INC(i) 'update element counter
  
  fpar_6 = Read_Timer() - time0 'measure real pixeltime
    
  if (i >= par_1) then 
    
    End
 
  endif
   

FINISH:
  
  stop_counters()
  par_2 = 1 'flag meaning the process finished
