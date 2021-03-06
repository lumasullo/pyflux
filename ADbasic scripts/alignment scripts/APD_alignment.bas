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
'process 3: APD_alignment by luciano a. masullo


'fpar_5: pixeltime (setpoint)
'fpar_6: pixeltime (measured)

'fpar_7: show pixel count

'data_1: counter 1 (APD 1)

'README:
'processdelay sets the pixeltime if this process is run on ADbasic
'fpar_7 will then display the number of counts every 1 ms

#INCLUDE .\data-acquisition.inc

dim co1, co1old as long at dm_local
dim currentx, yposition as float at dm_local
dim time0, time1 as float at dm_local
dim pixcnt_x as long at dm_local
dim data_1[1000] as long as fifo

INIT:
  
  PROCESSDELAY = 100000000
  
  fifo_clear(1)
  
  time0 = 0
  time1 = 0
  
  co1 = 0
  co1old = 0
  
  start_counters()

EVENT:
  time0 = Read_Timer() 'initial time of the pixel event
  
  'delay to wait for a given pixeltime
  DO 
    time1 = Read_Timer()
    
  UNTIL (Abs(time1 - time0) > fpar_5)

  co1 = cnt_read(1) 'read counter
  data_1 = co1 - co1old 'calculate counts in the current pixel
  
  'read APD
  fpar_7 = (co1 - co1old)/3.333333 'FIX provisorio, funciona solo para pixeltime = 1 ms y PROCESSDELAY = 1000000!!!

  
  co1old = co1 'update co1old
  
  fpar_6 = Read_Timer() - time0 'measure real pixeltime
  

FINISH:
  
  stop_counters()
