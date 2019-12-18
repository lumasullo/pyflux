# -*- coding: utf-8 -*-
"""
    driver for RGB Lasersystems MiniLaser Evo
    adopted from Lantz package
    written by Lars Richter and Luciano A. Masullo; date: 17.12.2019

"""

import serial

MSG_SUFFIX = '\r\n'

class MiniLasEvo:
    """Driver for any RGB Lasersystems MiniLas Evo laser.
    """
        
    def __init__(self, port):
        
        self.PORT = port
        
        BAUDRATE = 57600
        BYTESIZE = 8
        PARITY = serial.PARITY_NONE
        STOPBITS = 1
        self.TIMEOUT = 2 #in s, necessarry for readline() call
    
        #flow control flags
        RTSCTS = False
        DSRDTR = False
        XONXOFF = False
        
        self.serial = serial.Serial(port=self.PORT, baudrate=BAUDRATE,
                              bytesize=BYTESIZE, parity=PARITY,
                              stopbits=STOPBITS, timeout=self.TIMEOUT,
                              xonxoff=XONXOFF, rtscts=RTSCTS, dsrdtr=DSRDTR)
        
    def query(self, command):
        """Send query to the laser and return the answer, after handling
        possible errors.

        :param command: command to be sent to the instrument
        :type command: string
        """
        
        command = command + MSG_SUFFIX
        write_return = self.serial.write(command.encode('ascii'))
        
        code = int(write_return)
        if code != 0:
            if code == '1':
                print('[MiniLasEvo] Command invalid')
            elif code == '2':
                print('[MiniLasEvo] Wrong number of parameters')
            elif code == '3':
                print('[MiniLasEvo] Parameter value is out of range')
            elif code == '4':
                print('[MiniLasEvo] Unlocking code is wrong')
            elif code == '5':
                print('[MiniLasEvo] Device is locked for this command')
            elif code == '6':
                print('[MiniLasEvo] This function is not supported')
            elif code == '7':
                print('[MiniLasEvo] Timeout while reading command (60 s)')
            elif code == '8':
                print('[MiniLasEvo] This value is currently not available')
                
        message = self.serial.readline().decode()
        message = message.replace('0 ', '')
        message = message.replace(MSG_SUFFIX, '')
        
        return message

    def idn(self):
        """Identification of the device
        """
        manufacturer = self.query('DM?')
        device = self.query('DT?')
        serialnum = self.query('DS?')
        ans = manufacturer + ', ' + device + ', serial number: ' + serialnum
        return ans

    def status(self):
        """Current device status
        """
        ans = self.query('S?')
        if ans == '0x10':
            ans = 'Temperature of laser head is ok'
        elif ans == '0x01':
            ans = 'Laser system is active, radiation can be emitted'
        elif ans == '0x02':
            ans = '(reserved)'
        elif ans == '0x04':
            ans = 'The interlock is open'
        elif ans == '0x08':
            ans = self.query('E?')
            if ans == '0x01':
                ans = 'Temperature of laser head is too high'
            elif ans == '0x02':
                ans = 'Temperature of laser head is too low'
            elif ans == '0x04':
                ans = 'Temperature-sensor connection is broken'
            elif ans == '0x08':
                ans = 'Temperature sensor cable is shortened'
            elif ans == '0x40':
                ans = 'Current for laser head is too high'
            elif ans == '0x80':
                ans = 'Internal error (laser system cannot be activated)'
        return ans

    def operating_hours(self):
        """Total operating hours [hhhh:mm]
        """
        return self.query('R?')

    def software_version(self):
        """Software version
        """
        return self.query('DO?')

    def emission_wavelength(self):
        """Emission wavelength in nm
        """
        return self.query('DW?')

    def available_features(self):
        """Available features (reserved for future use)
        """
        return self.query('DF?')

    def control_mode(self):
        """Active current (power) control
        """
        ans = self.query('DC?')
        if ans == 'ACC':
            ans = 'Active current control'
        else:
            ans = 'Active power control'
        return ans

    # TEMPERATURE

    def temperature(self):
        """Current temperature in ºC
        """
        return self.query('T?')

    def temperature_min(self):
        """Lowest operating temperature in ºC
        """
        return self.query('LTN?')

    def temperature_max(self):
        """Highest operating temperature in ºC
        """
        return self.query('LTP?')

    # ENABLED REQUEST

    @property
    def enabled(self):
        """Method for turning on the laser
        """
        return bool(int(self.query('O?')))

    @enabled.setter
    def enabled(self, value):
        """Method for turning on the laser
        value: True, False
        """
        
        value = str(int(value))
        self.query('O=' + value)

#    # LASER POWER

    def initialize(self):
        super().initialize()
        self.feats.power.limits = (0, self.maximum_power.magnitude)

    def maximum_power(self):
        """Gets the maximum emission power of the laser
        """
        return float(self.query('LP?'))

    @property
    def power(self):
        """Gets and sets the emission power
        """
        return float(self.query('P?'))

    @power.setter
    def power(self, value):
        self.query('P={:.1f}'.format(value))

if __name__ == '__main__':

    port = 'COM3'
    minilaser = MiniLasEvo(port)
    
    #add initialize call
