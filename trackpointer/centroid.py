#================================ centroid ===============================
#
# @brief    Used to track the centroid of a target object.
#
# Centroid track pointer interface object.  Basically performs tracking
# of an object by computing its centroid.
#
# There are two ways to perform centroid tracking.  One is to presume
# that the object to track has already been binarized and it is simply a
# matter of computing the centroid.  The other is that the binarization
# needs to first be computed, then the centroid calculated.  If
# binarization of the input is desired, use the improcessor to perform
# the binarization as a pre-processing step.  If the binarization is too
# complicated for an improcess procedure, then write a tracker wrapper
# around the binarization + tracking combination. It will usually then
# become a ``perceiver`` object.
#
#================================ centroid ===============================

#
# @file     centroid.m
#
# @author   Patricio A. Vela,       pvela@gatech.edu
#           Yunzhi Lin,         yunzhi.lin@gatech.edu
# @date     2020/11/04  [created]
#           2021/07/12 [modified]
#
#!NOTE:
#!  set indent to 2 spaces.
#!  do not indent function code.
#!  set tab to 4 spaces with conversion to spaces.
#
#
#================================ centroid ===============================

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class State:
  tpt: np.ndarray = np.array([])
  haveMeas: bool = False

class Params(object):

  def __init__(self):
    pass

class centroid(object):

  # ============================== centroid =============================
  #
  # @brief      Centroid track-pointer constructor.
  #
  # @param[in]  iPt      The initial track point coordinates.
  #             params   The parameter structure.
  #
  def __init__(self, iPt=None, params=None):

    if iPt:
      self.tpt = iPt

    if params is None:
      params = self.setIfMissing(params,'plotStyle','rx')

    self.tparams = params
    self.haveMeas = False

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
  # @param[out] fval        Value of parameter.
  #
  def get(self, fname):

    # @todo    fill out.

    pass

  #============================= emptyState ============================
  #
  # @brief  Return an empty state structure.
  #
  #
  def emptyState(self):

    estate= State(tpt=[], haveMeas=False)

    return estate

  #============================== setState =============================
  #
  # @brief  Set the state vector.
  #
  # @param[in]  g   The desired state.
  #
  def setState(self, g):

    self.tpt = g
    self.haveMeas = True



  #============================== getState =============================
  #
  # @brief  Return the track-pointer state.
  #
  # @param[out] tstate  The track point state structure.
  #
  def getState(self):

    tstate = State(tpt=self.tpt, haveMeas=self.haveMeas)

    return tstate

  #=============================== offset ==============================
  #
  # @brief  Apply a vector offset to the track point.
  #
  # @param[in]  dp  The change in position to apply.
  #
  def offset(self, dp):

    if self.tpt:
      self.tpt = self.tpt + dp


  #============================= transform =============================
  #
  # @brief  Apply a Lie group transformation (linear/affine) to the track point.
  #
  # @param[in]  g   The Lie group class instance to apply.
  #
  def transform(self, g):

    if self.tpt:
      self.tpt = g * self.tpt


  #============================== predict ==============================
  #
  # @brief  Predict.  This is a no-op.
  #
  def predict(self):
    pass

  #============================== measure ==============================
  #
  # @brief  Measure the track point from the given image.
  #
  # @param[in]  I         The input image.
  #
  # @param[out] mstate    The measured state.
  #
  def measure(self, I):


    if hasattr(self.tparams, 'improcessor') and self.tparams.improcessor:
      Ip = self.tparams.improcessor.apply(I)
    else:
      Ip = I


    ibin, jbin = np.nonzero(Ip)

    self.tpt = np.array([np.mean(jbin), np.mean(ibin)]).reshape(-1,1)

    self.haveMeas = self.tpt.shape[1] > 0

    #   center = transpose(size(image)*[0 1;1 0]+1)/2;
    #  trackpoint = trackpoint - center;

    # @todo
    # Not sure if the translation is correct
    # if (nargout == 1):
    #   mstate = this.getState();
    # end
    mstate = self.getState()

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

  #========================= setIfMissing =========================
  #
  #  @brief Set missing parameters in the registration parameters structure.
  #
  #  @param[in]  params      The parameter structure.
  #              pname       Name of parameter.
  #              pval        Value of parameter.
  #
  #  @param[out] params      The parameter structure.
  #
  def setIfMissing(self, params, pname, pval):

    # @todo
    # Need double check on this translation
    if params is None or not isinstance(params):
      params = Params()
    setattr(params, pname, pval)
    return params

#
#================================ centroid ===============================
