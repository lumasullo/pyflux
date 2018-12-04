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
' Info_Last_Save                 = PC-MINFLUX  PC-MINFLUX\USUARIO
'<Header End>
'process 3: APD_alignment by luciano a. masullo


'par_8: 1 -> open shutter, 0 -> close shutter

'README:
'Very simpe process to digitally control the Thorlabs shutter

#INCLUDE .\data-acquisition.inc


INIT:
 
  Rem Configure DIO00.DIO15 as inputs and DIO16.DIO31 as outputs
  Conf_DIO(1100b)

EVENT:
  
  Digout(16, par_8)
  
FINISH:
  
