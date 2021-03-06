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
dim v, currentv, t_aux, x_aux, dt_aux as float at dm_local
dim time0, time1 as float at dm_local
dim pix_cnt as long at dm_local
dim data_1[1024] as long as fifo

INIT:
  
  fifo_clear(1)
  
  ai_offset = 32768
  
  time0 = 0
  time1 = 0
  
  co1 = 0
  co1old = 0
  signal = 0
  
  pix_cnt = 0
  
  currentx = fpar_1
  yposition = fpar_2
  
  currentv = 0
  v = fpar_3 / fpar_5
  t_aux = v / fpar_7
  x_aux = .5 * fpar_7 * t_aux^2 
  
  dt_aux = t_aux/par_2
  
  if (currentx > POSMAX) then currentx = POSMAX 'check that set x position is not higher than POSMAX
  if (currentx < POSMIN) then currentx = POSMIN 'check that set x position is not lower than POSMIN
  
  if (yposition > POSMAX) then yposition = POSMAX 'check that set x position is not higher than POSMAX
  if (yposition < POSMIN) then yposition = POSMIN 'check that set x position is not lower than POSMIN
  
  DAC(1, currentx)
  DAC(2, yposition)
  
  start_counters()

EVENT:
  
  'PART 1
  
  if (pix_cnt < par_2) then
  
    time0 = Read_Timer() 'initial time of the pixel event
  
    'delay to wait for a given pixeltime)
    DO 
      time1 = Read_Timer()
    
    UNTIL (Abs(time1 - time0) > dt_aux)

    if (par_30 = 0) then
    
      co1 = cnt_read(1) 'read counter
      data_1 = co1 - co1old 'calculate counts in the current pixel
      co1old = co1 'update co1old
    
    endif

    if (par_30 = 1) then
    
      signal = ADC(3) - ai_offset
      data_1 = signal

    endif
    
    currentv = currentv + fpar_7 * dt_aux
    currentx = currentx + currentv * dt_aux
    
    if (currentx > POSMAX) then currentx = POSMAX 'check that set x position is not higher than POSMAX
    if (currentx < POSMIN) then currentx = POSMIN 'check that set x position is not lower than POSMIN
  
    DAC(1, currentx)
    
    INC(pix_cnt)
    
  endif
  
  'PART 2
  
  if ((par_2 <= pix_cnt) and (pix_cnt < par_1 + par_2)) then
  
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
    
    currentx = currentx + fpar_3
    
    if (currentx > POSMAX) then currentx = POSMAX 'check that set x position is not higher than POSMAX
    if (currentx < POSMIN) then currentx = POSMIN 'check that set x position is not lower than POSMIN
  
    DAC(1, currentx)
    
    INC(pix_cnt)
    
  endif
  
  'PART 3
  
  if ((par_1 + par_2 <= pix_cnt) and (pix_cnt < par_1 + 3 * par_2)) then
  
    time0 = Read_Timer() 'initial time of the pixel event
  
    'delay to wait for a given pixeltime)
    DO 
      time1 = Read_Timer()
    
    UNTIL (Abs(time1 - time0) > dt_aux)

    if (par_30 = 0) then
    
      co1 = cnt_read(1) 'read counter
      data_1 = co1 - co1old 'calculate counts in the current pixel
      co1old = co1 'update co1old
    
    endif

    if (par_30 = 1) then
    
      signal = ADC(3) - ai_offset
      data_1 = signal

    endif
    
    currentv = currentv - fpar_7 * dt_aux
    currentx = currentx + currentv * dt_aux
    
    if (currentx > POSMAX) then currentx = POSMAX 'check that set x position is not higher than POSMAX
    if (currentx < POSMIN) then currentx = POSMIN 'check that set x position is not lower than POSMIN
  
    DAC(1, currentx)
    
    INC(pix_cnt)
    
  endif
  
  'PART 4
  
  if ((par_1 + 3 * par_2 <= pix_cnt) and (pix_cnt < 2 * par_1 + 3 * par_2)) then
  
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
    
    currentx = currentx + fpar_3
    
    if (currentx > POSMAX) then currentx = POSMAX 'check that set x position is not higher than POSMAX
    if (currentx < POSMIN) then currentx = POSMIN 'check that set x position is not lower than POSMIN
  
    DAC(1, currentx)
    
    INC(pix_cnt)
    
  endif
  
  'PART 5
  
  if ((2 * par_1 + 2 * par_2 <= pix_cnt) and (pix_cnt < 2 * par_1 + 4 * par_2)) then
  
    time0 = Read_Timer() 'initial time of the pixel event
  
    'delay to wait for a given pixeltime)
    DO 
      time1 = Read_Timer()
    
    UNTIL (Abs(time1 - time0) > dt_aux)

    if (par_30 = 0) then
    
      co1 = cnt_read(1) 'read counter
      data_1 = co1 - co1old 'calculate counts in the current pixel
      co1old = co1 'update co1old
    
    endif

    if (par_30 = 1) then
    
      signal = ADC(3) - ai_offset
      data_1 = signal

    endif
    
    currentv = currentv + fpar_7 * dt_aux
    currentx = currentx + currentv * dt_aux
    
    if (currentx > POSMAX) then currentx = POSMAX 'check that set x position is not higher than POSMAX
    if (currentx < POSMIN) then currentx = POSMIN 'check that set x position is not lower than POSMIN
  
    DAC(1, currentx)
    
    INC(pix_cnt)
    
  endif
  
  'END
      
  if (pix_cnt >= 2 * par_1 + 4 * par_2) then End
   
    

FINISH:

  pix_cnt = 0 'reset x pixel counter
  
  stop_counters()
