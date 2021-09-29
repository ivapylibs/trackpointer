#================================ centroidMulti ===============================
#
# @brief    Used to track the centroid of multiple (binarized) objects.
#
# Performs tracking of multiple objects by computing their centroids.
#
# There are two ways to perform centroid tracking.  One is to presume
# that the object to track has already been binarized and it is simply a
# matter of computing the centroid.  The other is that the binarization
# needs to first be computed, then the centroid calculated.  If
# binarization of the input is desired, use an ``improcessor`` to perform the
# binarization as a pre-processing step.  If the binarization is too
# complicated for an improcess procedure, then write a tracker wrapper around
# the binarization + tracking combination. It will usually then become a
# ``perceiver`` object.
#
#================================ centroidMulti ===============================

#
# @file     centroidMulti.m
#
# @author   Patricio A. Vela,       pvela@gatech.edu
#           Yunzhi Lin,             yunzhi.lin@gatech.edu
# @date     2020/11/10  [created]
#           2021/07/21  [modified]
#
#!NOTE:
#!  set indent to 2 spaces.
#!  do not indent function code.
#!  set tab to 4 spaces with conversion to spaces.
#
#
#================================ centroid ===============================

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import cv2

from skimage.measure import regionprops
from trackpointer.centroid import centroid
from trackpointer.centroid import Params as cParams

class State(object):

  def __init__(self, tpt=None, haveMeas=None):
    self.tpt = tpt
    self.haveMeas = haveMeas

@dataclass
class Params(cParams):
  improcessor: any = None 


class centroidMulti(centroid):

  # ============================== centroid =============================
  #
  # @brief      Centroid track-pointer constructor.
  #
  # @param[in]  iPt     The initial track point coordinates.
  #             params   The parameter structure.
  #
  def __init__(self, iPt=None, params=Params()):

    super(centroidMulti,self).__init__(iPt,params)


  #=============================== set ===============================
  #
  # @brief  Set parameters for the tracker.
  #
  # @param[in]  fname       Name of parameter.
  #             fval        Value of parameter.
  #
  def set(self, fname, fval):

    # @todo    fill out.

    pass


  #=============================== get ===============================
  #
  # @brief  Get parameter of the tracker.
  #
  # @param[in]  fname       Name of parameter.
  #
  # @param[out] fval        Value of parameter.
  #
  def get(self, fname):

    # @todo    fill out.

    pass

  #============================== measure ==============================
  #
  # @brief  Measure the track point from the given image.
  #
  # @param[in]  I   The input image.
  #
  def measure(self, I):

    if hasattr(self.tparams, 'improcessor') and self.tparams.improcessor:
      Ip = self.tparams.improcessor.apply(I)
    else:
      Ip = I

    binReg = centroidMulti.regionProposal(Ip)
    self.tpt =  np.array(binReg).T # from N x 2 to 2 x N

    if len(self.tpt) == 0:
      self.haveMeas = 0
    else:
      self.haveMeas = self.tpt.shape[1] > 0

    # @todo
    # Not sure if the translation is correct
    # if (nargout == 1):
    #   mstate = this.getstate();
    # end
    mstate = self.getstate()

    return mstate

  #============================== process ==============================
  #
  # @brief  Process the input image according to centroid tracking.
  #
  # @param[in]  I   The input image.
  #
  def process(self, I):

    self.measure(I)

  #============================ displayState ===========================
  #
  # @brief  Displays the current track pointer measurement.
  #
  # Assumes that the current figure to plot to is activate.  If the plot has
  # existing elements that should remain, then hold should be enabled prior to
  # invoking this function.
  #
  def displayState(self, dstate = None):

    if dstate:
      if dstate.haveMeas:
        plt.plot(dstate.tpt[0,:], dstate.tpt[1,:], self.tparams.plotStyle)
    else:
      if self.haveMeas:
        plt.plot(self.tpt[0,:], self.tpt[1,:], self.tparams.plotStyle)


  #========================= displayDebugState =========================
  #
  # @brief  Displays internally stored intermediate process output.
  #
  # Currently, there is no intermediate output, though that might change
  # in the future.
  #
  def displayDebugState(self, dbstate=None):
    pass

  #========================= regionProposal =========================
  #
  # @brief  Find out the centroid for multiple objects
  #
  # @param[in]  I   The input mask image.
  #
  @ staticmethod
  def regionProposal(I):
    mask = np.zeros_like(I)
    cnts = cv2.findContours(I, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    # For OpenCV 4+
    cnts = cnts[0]
    
    for idx, cnt in enumerate(cnts):
      cv2.drawContours(mask, cnt, -1, (idx+1), 1)

    # Note that regionprops assumes different areas are with different labels
    # See https://stackoverflow.com/a/61591279/5269146
    binReg = [[i.centroid[1], i.centroid[0]] for i in regionprops(mask)]

    return binReg
#
#================================ centroidMulti ===============================
