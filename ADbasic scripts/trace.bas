'<ADbasic Header, Headerversion 001.001>
' Process_Number                 = 6
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
'<Header End>
'process 6: trace measurement by luciano a. masullo


'fpar_65: bintime (setpoint)
'fpar_66: bintime (measured)
'fpar_67: total time

'fpar_68: show bin count

'data_6: counter 1 (APD 1)

'README:
'processdelay sets the pixeltime if this process is run on ADbasic



#INCLUDE .\data-acquisition.inc

dim co1, co1old as long at dm_local
dim currentx, yposition as float at dm_local
dim time0, time1 as float at dm_local
dim pixcnt_x as long at dm_local
dim i as integer
dim tot_bin as integer
dim data_6[1024] as long as fifo

INIT:
  
  'PROCESSDELAY = 300000
 
  fifo_clear(6)
  
  i = 1
  
  time0 = 0
  time1 = 0
  
  co1 = 0
  co1old = 0
   
  tot_bin = par_60
  
  start_counters()

EVENT:
 
  time0 = Read_Timer() 'initial time of the pixel event
  
  'delay to wait for a given pixeltime
  DO 
    time1 = Read_Timer()
    
  UNTIL (Abs(time1 - time0) > fpar_65)

  co1 = cnt_read(1) 'read counter
  data_6 = co1 - co1old 'calculate counts in the current pixel
  
  'display counter on ADbasic
  fpar_68 = (co1 - co1old) 

  co1old = co1 'update co1old
  
  fpar_66 = Read_Timer() - time0 'measure real pixeltime
  
  INC(i)
  
  if (i > tot_bin) then End
  
FINISH:
  
  stop_counters()
