#!/usr/bin/python
#================================ track01top ===============================

#================================ track01top ===============================
#
# @author   Patricio A. Vela,       pvela@gatech.edu
# @date     2023/07/28              [created]
#
#================================ track01top ===============================

import trackpointer.toplines as tp
import numpy as np




Ip = np.zeros((80,50))
Ip[30:51,20:31] = 1

trackpt = tp.fromTop()
trackpt.measure(Ip)

print(trackpt.tpt)

trackpt = tp.fromBottom()
trackpt.measure(Ip)

print(trackpt.tpt)

#
#================================ track01top ===============================
