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
from detector.Configuration import AlgConfig

@dataclass
class State:
  tpt: np.ndarray = np.array([])
  haveMeas: bool = False

class CfgCentroid(AlgConfig):
  """The parameters for the centroid tracker

  Args:
    plotStyle (str): The plot style from the matplotlib for the centroid. Defaults to "rx". \
      Detailed choices see: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
  """
  plotStyle:str = "rx"

#
#---------------------------------------------------------------------------
#====================== Configuration Node : Centroid ======================
#---------------------------------------------------------------------------
#

class CfgCentroind(AlgConfig):
  '''!
  @brief  Configuration setting specifier for centroidMulti.
  '''
  #============================= __init__ ============================
  #
  '''!
  @brief        Constructor of configuration instance.

  @param[in]    cfg_files   List of config files to load to merge settings.
  '''
  def __init__(self, init_dict=None, key_list=None, new_allowed=True):

    if (init_dict == None):
      init_dict = CfgCentroid.get_default_settings()

    super().__init__(init_dict, key_list, new_allowed)

    # self.merge_from_lists(XX)

  #========================= get_default_settings ========================
  #
  # @brief    Recover the default settings in a dictionary.
  #
  @staticmethod
  def get_default_settings():
    '''!
    @brief  Defines most basic, default settings for RealSense D435.

    @param[out] default_dict  Dictionary populated with minimal set of
                              default settings.
    '''
    default_dict = dict(plotStyle = 'rx')
    return default_dict


#
#---------------------------------------------------------------------------
#================================= Centroid ================================
#---------------------------------------------------------------------------
#

class centroid(object):

  # ============================== centroid =============================
  #
  # @brief      Centroid track-pointer constructor.
  #
  # @param[in]  iPt      The initial track point coordinates.
  #             params   The parameter structure.
  #
  def __init__(self, iPt=None, params=CfgCentroid()):

    self.tparams = params
    self.haveMeas = False

    if iPt:
      self.tpt = iPt
      self.haveMeas = True
    else:
      self.tpt = None

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

    estate= State(tpt=np.array([]), haveMeas=False)

    return estate

  #============================== setState =============================
  #
  # @brief  Set the state vector.
  #
  # @param[in]  dPt   The desired state.
  #
  def setState(self, dPt):

    self.tpt = dPt.tpt
    self.haveMeas = dPt.haveMeas



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

    # y,x in OpenCV
    ibin, jbin = np.nonzero(Ip)

    if ibin.size == 0:
      self.tpt = None
      self.haveMeas = False
    else:
      # x,y in OpenCV
      self.tpt = np.array([np.mean(jbin), np.mean(ibin)]).reshape(-1,1)
      self.haveMeas = True

    mstate = self.getState()

    return mstate

  #============================== correct ==============================
  #
  # @brief  Correct.  This is a no-op.
  #
  def correct(self):
    pass

  #=============================== adapt ===============================
  #
  # @brief  Adapt.  This is a no-op.
  #
  def adapt(self):
    pass


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
  def displayState(self, dstate = None, ax=None):

    if ax is None:
      ax = plt.gca()

    if isinstance(dstate, State):
      if dstate.haveMeas:
        # Change to OpenCV style
        ax.plot(dstate.tpt[0,:], dstate.tpt[1,:], self.tparams.plotStyle)
    else:
      if self.haveMeas:
        # Change to OpenCV style WHICH IS WHICH?????
        ax.plot(self.tpt[0,:], self.tpt[1,:], self.tparams.plotStyle)



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
    if not isinstance(params, Params):
      params = Params()
    setattr(params, pname, pval)
    return params

#
#================================ centroid ===============================
