#!/usr/bin/python
#=============================== glove3track ==============================
"""
@brief          Bag file implementation of red glove segmentation + tracking.

Extending glove02track and including track state filter.  Let's see what the 
result is.  Merging the estimator dfilt01simple code into this too.

Execution:
----------
Requires user input.

There are two phases of operation.  The first does a little model updating
until the `q` key is pressed.  The second freezes the glove target model and
performs color-based detection.  Hitting `q` again quits the routine.

Assumes availability of an Intel Realsense D435 (or compatible) stream.

"""
#=============================== glove3track ==============================
#
# @author         Patricio A. Vela,       pvela@gatech.edu
# @date           2023/05/26              [created]
#
#=============================== glove3track ==============================

import numpy as np
import matplotlib.pyplot as plot
import cv2

import camera.utils.display as display
from camera.d435.runner2 import CfgD435
from camera.d435.runner2 import Replay

import detector.fgmodel.Gaussian as SGM 
import trackpointer.toplines as tp
import estimator.dtfilters as df
import detector.activity.simple as ad


#==[0]  Setup the camera and the red glove target model.
#       Use hard coded glove configuration.
#
#----[0.1] Camera stuff.
#cfgStream = CfgD435.builtForReplay('../../camera/testing/d435/bagsource.bag')
cfgStream = CfgD435.builtForReplay('/home/mary/Documents/20230817_113532.bag')
cfgStream.camera.align = True
theStream = Replay(cfgStream)

#----[0.2] Target model (do not update).
fgModP  = SGM.SGMdebug(mu = np.array([150.0,2.0,30.0]), 
                      sigma = np.array([700.0,200.0,200.0]) )
fgModel = SGM.Gaussian( SGM.CfgSGT.builtForRedGlove(), None, fgModP )

tpCenter = tp.fromBottom()

#----[0.3] Trackpoint filter
dt = 1/10
A = np.array( [[1, 0, dt, 0], [0, 1, 0, dt],[0, 0, 1, 0],[0, 0, 0, 1]])
C = np.array( [[1, 0, 0, 0], [0, 1, 0, 0]] )
L = df.calcGainByDARE(A, C, 0.5*np.identity(4), 0.15*np.identity(2))

x0 = np.array([[50],[50],[0],[0]])
pEst = df.Linear(A, C, L, x0)

#----[0.4] Activity detector
movcfg = ad.CfgMoving()
movcfg.tau = 15
movdet = ad.isMovingInImage(movcfg)

Cvel = np.array( [[0, 0, 1, 0], [0, 0, 0, 1]] )

#---[0.5] Store motion data.
tdat = np.array([0])
xdat = x0
zdat = ad.MoveState.STOPPED.value

print(xdat)


#==[1] Run loop and see what get.
#

# @note Will crash here if the bag file is compressed.
theStream.start()

while(True):
    rgb, dep, success = theStream.get_frames()
    if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()

    fgModel.detect(rgb)
    fgS = fgModel.getState()

    tpCenter.process(fgS.fgIm)

    if (tpCenter.haveMeas):
      pEst.process( tpCenter.tpt )
      movdet.detect( np.matmul(Cvel, pEst.x_hat) )

    tpt = np.matmul(C, pEst.x_hat)
    display.trackpoint_cv(rgb, tpt, ratio=0.5, window_name="Tracking")
    print(movdet.z)

    xdat = np.hstack((xdat,pEst.x_hat))
    tdat = np.hstack((tdat, tdat[-1]+1))

    opKey = cv2.waitKey(1)
    if opKey == ord('q'):
        break

fsig = display.plotfig(None, None)
fsig.num, fsig.axis = plot.subplots()
ax2 = fsig.axis.twinx()

fsig.axis.plot(tdat, np.transpose(np.matmul(C,xdat)))
ax2.plot(tdat, np.transpose(np.matmul(Cvel,xdat)))

vnorm = np.linalg.norm(np.matmul(Cvel,xdat), axis=0)

fnorm = display.plotfig(None, None)
fnorm.num, fnorm.axis = plot.subplots()
fnorm.axis.plot(tdat, vnorm)

plot.show()

#
#=============================== glove3track ==============================
