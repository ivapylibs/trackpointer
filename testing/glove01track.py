#!/usr/bin/python
#=============================== glove01track ==============================
"""
@brief          Real-time implementation of red glove segmentation + tracking.

Copied base code from ``tgm03depth435.py`` from the detector testing code.
Expands on it by including a trackpointer.  The code is deconstructed in the
sense that it is not packaged up as a perceiver.

Execution:
----------
Requires user input.

There are two phases of operation.  The first does a little model updating
until the `q` key is pressed.  The second freezes the glove target model and
performs color-based detection.  Hitting `q` again quits the routine.

Assumes availability of an Intel Realsense D435 (or compatible) stream.

"""
#============================== tgm03depth435 ==============================
#
# @author         Patricio A. Vela,       pvela@gatech.edu
# @date           2023/05/26              [created]
#
#============================== tgm03depth435 ==============================

import cv2
import camera.utils.display as display
import camera.d435.runner2 as d435

import numpy as np
import detector.fgmodel.Gaussian as SGM 
import trackpointer.centroid as tp


#==[0]  Setup the camera and the red glove target model.
#       Use hard coded glove configuration.
#
d435_configs = d435.CfgD435()
d435_configs.merge_from_file('depth435.yaml')
theStream = d435.D435_Runner(d435_configs)
theStream.start()

fgModP  = SGM.SGMdebug(mu = np.array([150.0,2.0,30.0]), 
                      sigma = np.array([1100.0,250.0,250.0]) )
fgModel = SGM.Gaussian( SGM.CfgSGT.builtForRedGlove(), None, fgModP )

tpCenter = tp.centroid()

#==[1] Run the red glove detector with model updating until `q` pressed.
#
fgModel.refineFromRGBDStream(theStream, True)

#==[2] Run red glove detector with frozen model until `q` pressed.
#
while(True):
    rgb, dep, success = theStream.get_frames()
    if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()

    fgModel.detect(rgb)
    fgS = fgModel.getState()

    tpCenter.process(fgS.fgIm)

    if tpCenter.haveMeas:
      display.trackpoint_cv(rgb, tpCenter.tpt, ratio=0.5, window_name="Tracking")
    else:
      display.rgb_cv(rgb, ratio=0.5, window_name="Tracking")
    opKey = cv2.waitKey(1)
    if opKey == ord('q'):
        break

print('Mean is:' , fgModel.mu)
print('Var  is:' , fgModel.sigma)
#
#============================== tgm03depth435 ==============================
