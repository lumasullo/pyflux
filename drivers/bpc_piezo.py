# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 10:56:25 2021

@author: Lucia Lopez
"""

"""
Adapted from rwb27 / benchtop_piezo.py

This is a Python 3 wrapper for the Thorlabs BPC203 Benchtop Piezo controller.
It relies on the Thorlabs Kinesis API (so you should copy in, or add to your
Python path, the Kinesis DLLs).  The easiest way to copy the right DLLs is
to use the "DLL copying utility" which is probably located in 
c:/Program Files/Thorlabs/Kinesis
I also use the excellent ``pythonnet`` package to get access to the .NET API.
This is by far the least painful way to get Kinesis to work nicely as it 
avoids the low-level faffing about.
"""
import clr # provided by pythonnet, .NET interface layer
import sys
import time

# this is seriously nasty.  Points for a better way of fixing this!
sys.path.append(r"C:\Program Files\Thorlabs\Kinesis")

# NB the 
clr.AddReference("Thorlabs.MotionControl.Benchtop.PiezoCLI")
clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")
clr.AddReference("System")

from Thorlabs.MotionControl.Benchtop.PiezoCLI import BenchtopPiezo
from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
from System import Decimal


def list_devices():
    """Return a list of Kinesis serial numbers"""
    DeviceManagerCLI.BuildDeviceList()
    return DeviceManagerCLI.GetDeviceList()



class BenchtopPiezoWrapper():
    def __init__(self, serial_number):
        self._ser = str(serial_number)
        DeviceManagerCLI.BuildDeviceList()
        self._piezo = BenchtopPiezo.CreateBenchtopPiezo(self._ser)
        self.channels = []
        self.connected = False

    def connect(self):
        """Initialise communications, populate channel list, etc."""
        assert not self.connected
        self._piezo.Connect(self._ser)
        self.connected = True
        assert len(self.channels) == 0, "Error connecting: we've already initialised channels!"
        for i in range(self._piezo.ChannelCount):
            chan = self._piezo.GetChannel(i+1) # Kinesis channels are one-indexed
            chan.WaitForSettingsInitialized(5000)
            chan.StartPolling(250) # getting the voltage only works if you poll!
            time.sleep(0.5) # ThorLabs have this in their example...
            chan.EnableDevice()
            # I don't know if the lines below are necessary or not - but removing them
            # may or may not work...
            time.sleep(0.5)
            config = chan.GetPiezoConfiguration(chan.DeviceID)
            info = chan.GetDeviceInfo()
            max_v = Decimal.ToDouble(chan.GetMaxOutputVoltage())
            self.channels.append(chan)

    def close(self):
        """Shut down communications"""
        if not self.connected:
            print("Not closing piezo device {self._ser}, it's not open!")
            return
        for chan in self.channels:
            chan.StopPolling()
        self.channels = []
        self._piezo.Disconnect(True)

    def __del__(self):
        try:
            if self.connected:
                self.close()
        except:
            print("Error closing communications on deletion of device {self._ser}")
    
    def set_zero(self):
        """Sets the voltage output to zero and defines the ensuing actuator position az zero. """
        for chan in self.channels:
            chan.SetZero()
        
    def set_output_voltages(self, voltages):
        """Set the output voltage"""
        assert len(voltages) == len(self.channels), "You must specify exactly one voltage per channel"
        for chan, v in zip (self.channels, voltages):
            chan.SetOutputVoltage(Decimal(v))
    
    def get_output_voltages(self):
        """Retrieve the output voltages as a list of floating-point numbers"""
        return [Decimal.ToDouble(chan.GetOutputVoltage()) for chan in self.channels]
    
    output_voltages = property(get_output_voltages, set_output_voltages)
    
    def get_pos_control_mode(self):
        """Gets the Position Control Mode. 1 = open loop, 2 = closed loop"""
        return [(chan.GetPositionControlMode()) for chan in self.channels]
    
    
    def set_pos_control_mode(self, mode):
        """Sets the position control mode for all the channels"""
        for chan in self.channels:
            chan.SetPositionControlMode(mode)
    
    def get_positions(self):
        """Retrieve the position as a list of floating-point numbers [μm]"""
        return [Decimal.ToDouble(chan.GetPosition()) for chan in self.channels]

    
    def set_positions(self, positions):
        """Set the position [μm]"""
        assert len(positions) == len(self.channels), "You must specify exactly one position per channel"
        for chan, p in zip (self.channels, positions):
            chan.SetPosition(Decimal(p))

    

        
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    