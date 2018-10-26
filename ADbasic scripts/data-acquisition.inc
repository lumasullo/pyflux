'<ADbasic Header, Headerversion 001.001>
'<Header End>
'global include file

#INCLUDE ADwinGoldII.inc

'scanner-related parameters

#DEFINE POSMIN 32768 'enforce safe limits 
#DEFINE POSMAX 65635 '

#DEFINE MAXINT 1E3 'maximum value of I (fpar_72)
#DEFINE INTLIMIT 1E3 'maximum value of the deviation integral

#DEFINE THRESHOLD1 1500 'dev < threshold1 -> start integrator
#DEFINE THRESHOLD2 1500 'x/ydev <= threshold2 -> actuating value x/y (par_71/73) considered OK
#DEFINE THRESHOLD3 1500 'zdev <= threshold2 -> actuating value z (par_75) considered OK

'#DEFINE THRESHOLD1 100 'dev < threshold1 -> start integrator
'#DEFINE THRESHOLD2 35 'x/ydev <= threshold2 -> actuating value x/y (par_71/73) considered OK
'#DEFINE THRESHOLD3 35 'zdev <= threshold2 -> actuating value z (par_75) considered OK
'#DEFINE THRESHOLD1 100 'dev < threshold1 -> start integrator
'#DEFINE THRESHOLD2 6 'x/ydev <= threshold2 -> actuating value x/y (par_71/73) considered OK
'#DEFINE THRESHOLD3 10 'zdev <= threshold2 -> actuating value z (par_75) considered OK
#DEFINE GOODCOUNT1 300 'how many points with x/ydev < threshold2 to stop integrator and settle 
#DEFINE GOODCOUNT2 400 'how many points with zdev < threshold2 to stop integrator and settle 
#DEFINE NUMAVG 10 'number of ADC averages (important: processdelay * NUMAVG = 1 ms)

#DEFINE PCOEFF 1.3 'PID feedback coefficients
#DEFINE ICOEFF 0.1
#DEFINE DCOEFF 5

'counter-related macros

Sub start_counters()
  cnt_enable(0000b)  'disable all counters
  cnt_se_diff(0000b)  'set all counters to single-ended input
  cnt_mode(1, 00001000b)  'set counter 1 to clock/dir mode, dir inverted, CLR
  cnt_mode(2, 00001000b)  'set counter 2 to clock/dir mode, dir inverted, CLR
  cnt_clear(0011b)  'clear counters 1 and 2
  cnt_enable(0011b) 'enable counters 1 and 2
EndSub

Sub stop_counters()
  cnt_enable(0000b)    'disable all counters
EndSub
