#!/usr/bin/python
#=============================== glove04track ==============================
"""
@brief          Bag file implementation of red glove segmentation + tracking.

Extending glove03track, which had issues tracking the velocity.  This one
includes an estimate of the velocities from finite differences.  That
seems to do better subject to below.

How to set parameters depends on the resolution.  Seems like processing
in a timely manner is best done at lower resolution.  Using max resolution
starts to introduce jitter, especially because I don't quite know how to
tune the filter gains.  I need to read up on how discrete time estimators
work because it doesn't quite align with expectation.


Execution:
----------
Requires user input.
Require realsense saved ROS bag.

"""
#=============================== glove04track ==============================
#
# @author         Patricio A. Vela,       pvela@gatech.edu
# @date           2023/05/26              [created]
#
#=============================== glove04track ==============================

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
#cfgStream = CfgD435.builtForReplay('/home/mary/Documents/20230817_113532.bag')
cfgStream = CfgD435.builtForReplay('/home/mary/Documents/20230817_152320.bag')
cfgStream.camera.align = True
theStream = Replay(cfgStream)

#----[0.2] Target model (do not update).
fgModP  = SGM.SGMdebug(mu = np.array([150.0,2.0,30.0]), 
                      sigma = np.array([700.0,200.0,200.0]) )
fgModel = SGM.Gaussian( SGM.CfgSGT.builtForRedGlove(), None, fgModP )

tpCenter = tp.fromBottom()

#----[0.3] Trackpoint filter
dt = 1
A = np.array( [[1, 0, dt, 0], [0, 1, 0, dt],[0, 0, 1, 0],[0, 0, 0, 1]])
print(A)
#C = np.array( [[1, 0, 0, 0], [0, 1, 0, 0]] )
C = np.identity(4)
L = df.calcGainByDARE(A, C, 1*np.diag([8, 8, 4, 4]), 1*np.diag([2, 2, 1, 1]))

x0 = np.array([[50],[50],[0],[0]])
pEst = df.Linear(A, C, L, x0)

#----[0.4] Activity detector
movcfg = ad.CfgMoving()
#movcfg.tau = 50    # 1920x1080
movcfg.tau = 15     # 840x480
movdet = ad.isMovingInImage(movcfg)

Cpos = np.array( [[1, 0, 0, 0], [0, 1, 0, 0]] )
Cvel = np.array( [[0, 0, 1, 0], [0, 0, 0, 1]] )

#---[0.5] Store motion data.
tdat = np.array([0])
xdat = x0
zdat = [ad.MoveState.STOPPED.value]
mdat = x0

xmes = x0
tprv = None
print(xdat)


#==[1] Run loop and see what get.
#

# @note Will crash here if the bag file is compressed.
theStream.start()
doPlot = True
addWin = True

while(True):
    rgb, dep, success = theStream.get_frames()
    if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()

    fgModel.detect(rgb)
    fgS = fgModel.getState()

    tpCenter.process(fgS.fgIm)

    if tprv is None:
      tprv = tpCenter.tpt

    if (tpCenter.haveMeas):
      xmes = np.vstack( (tpCenter.tpt, (tpCenter.tpt - tprv) ))
      pEst.process( xmes )
      tprv = tpCenter.tpt
    else:
      pEst.process(None)
      movdet.detect(None)

    movdet.detect( np.matmul(Cvel, pEst.x_hat) )

    tpt = np.matmul(C, pEst.x_hat)

    tdat = np.hstack((tdat, tdat[-1]+1))
    xdat = np.hstack((xdat,pEst.x_hat))
    mdat = np.hstack((mdat,xmes))
    zdat.append(movdet.z.value)

    if doPlot:
      display.trackpoint_cv(rgb, tpt, ratio=0.5, window_name="Tracking")
      opKey = cv2.waitKey(1)
      if opKey == ord('q'):
          break
    else:
      if addWin:
        display.trackpoint_cv(rgb, tpt, ratio=0.5, window_name="Tracking")
        addWin = False

      opKey = cv2.waitKey(1)
      if opKey == ord('q'):
          break

num_stop = np.sum((np.diff(zdat,1) < 0))
print(num_stop)

      
fsig = display.plotfig(None, None)
fsig.num, fsig.axis = plot.subplots()
ax2 = fsig.axis.twinx()

fsig.axis.plot(tdat, np.transpose(np.matmul(Cpos,xdat)), 'bx-')
ax2.plot(tdat, np.transpose(np.matmul(Cvel,xdat)), 'g+-.')

fmes = display.plotfig(None, None)
fmes.num, fmes.axis = plot.subplots()
ax2 = fmes.axis.twinx()

fmes.axis.plot(tdat, np.transpose(np.matmul(Cpos,mdat)), 'bx-')
ax2.plot(tdat, np.transpose(np.matmul(Cvel,mdat)), 'g+-.')

vnorm = np.linalg.norm(np.matmul(Cvel,xdat), axis=0)

fnorm = display.plotfig(None, None)
fnorm.num, fnorm.axis = plot.subplots()
fnorm.axis.plot(tdat, vnorm)

fact = display.plotfig(None, None)
fact.num, fact.axis = plot.subplots()
fact.axis.plot(tdat, zdat)

plot.show(block=False)

while True:
    plot.pause(0.01)
    opKey = cv2.waitKey(1)
    if opKey == ord('q'):
        break

plot.close('all')
#
#=============================== glove04track ==============================
