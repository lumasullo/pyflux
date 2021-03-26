# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 09:56:43 2021

@author: Lucia Lopez
"""

import bpc_piezo as bpc

print('Serial number {}'.format(bpc.list_devices()))

pz = bpc.BenchtopPiezoWrapper(bpc.list_devices()[0])

pz.connect()

pz.set_zero()

# pz.set_positions([1, 2, 3])

# pz.get_positions()

# pz.close()

pz.get_positions()
pz.set_positions([1.234, 2.985, 3.256])
pz.get_positions()
pz.set_positions([1.234, 2.985, 3.256])
pz.get_positions()

# chan = pz._piezo.GetChannel(1)

# cts = chan.GetFeedbackLoopPIconsts()

# cts.ProportionalTerm = 100
# cts.IntegralTerm = 0

