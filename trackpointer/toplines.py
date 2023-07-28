#================================ toplines ===============================
#
# @brief    Used to track the centroid of a target object.
#
# Toplines track pointer interface object.  Rather than use the centroid,
# this track pointer presumes some form of directionality for the target
# and tracks only the top pixel measurements based on the direction.
# For example, if the direction is towards the top of the image, then
# toplines will snag a track point from a topwards biased subset of the
# tracked object blob.
#
#================================ toplines ===============================

#
# @file     centroid.m
#
# @author   Patricio A. Vela,       pvela@gatech.edu
# @date     2023/07/28              [created]
#
#!NOTE:
#!  set indent to 2 spaces.
#!  do not indent function code.
#!  set tab to 4 spaces with conversion to spaces.
#
#================================ toplines ===============================

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

import trackpointer.centroid as tp

@dataclass
class Params(object):
  """!
  @brief    Parameters for the toplines tracker

  @param  plotStyle (str): The plot style from the matplotlib for the centroid.
                           Defaults to "rx". \

  Detailed choices see: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
  """
  plotStyle:str = "rx"
  numLines = 15

class fromTop(tp.centroid):

  #============================== fromTop ==============================
  #
  # @brief      Track-pointer constructor.
  #
  # @param[in]  iPt      The initial track point coordinates.
  # @param[in]  params   The parameter structure.
  #
  def __init__(self, iPt=None, params=Params()):

    if not isinstance(params, Params):
      params = self.setIfMissing(params,'plotStyle','rx')

    self.tparams = params
    self.haveMeas = False

    if iPt:
      self.tpt = iPt
      self.haveMeas = True
    else:
      self.tpt = None

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

    #--[1] Get the top-most non-empty row and ones before it. Compute row
    #      average of data.
    #      
    imsize   = np.shape(Ip)
    hitCount = np.sum(Ip, axis=1)
    hitInds  = np.argwhere(hitCount)

    if (np.size(hitInds) == 0):     # If nothing, then no measurement.

      self.tpt = None
      self.haveMeas = False

    else:

      topInd   = np.min(hitInds)
      if (topInd < (imsize[0] - self.tparams.numLines)):
        botInd = topInd + self.tparams.numLines 
      else:
        botInd = imsize[0] 

      useCount = hitCount[topInd:botInd]
      useInds  = range(topInd, botInd)
      crow     = np.inner(useCount,useInds) / np.sum(useCount)


      #--[2] Get the row and ones before it. Compute column average.
      #      
      Irows = Ip[topInd:botInd,:]
      ibin, jbin = np.nonzero(Irows)          # y,x in OpenCV

      self.tpt = np.array([np.mean(jbin), crow]).reshape(-1,1)
      self.haveMeas = True

    mstate = self.getState()
    return mstate

class fromBottom(tp.centroid):

  #============================== fromTop ==============================
  #
  # @brief      Track-pointer constructor.
  #
  # @param[in]  iPt      The initial track point coordinates.
  # @param[in]  params   The parameter structure.
  #
  def __init__(self, iPt=None, params=Params()):

    if not isinstance(params, Params):
      params = self.setIfMissing(params,'plotStyle','rx')

    self.tparams = params
    self.haveMeas = False

    if iPt:
      self.tpt = iPt
      self.haveMeas = True
    else:
      self.tpt = None

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

    #--[1] Get the top-most non-empty row and ones before it. Compute row
    #      average of data.
    #      
    imsize   = np.shape(Ip)
    hitCount = np.sum(Ip, axis=1)
    hitInds  = np.argwhere(hitCount)

    if (np.size(hitInds) == 0):     # If nothing, then no measurement.

      self.tpt = None
      self.haveMeas = False

    else:

      botInd   = np.max(hitInds)+1
      if (botInd > self.tparams.numLines):
        topInd = botInd - self.tparams.numLines 
      else:
        topInd = 0

      useCount = hitCount[topInd:botInd]
      useInds  = range(topInd, botInd)
      crow     = np.inner(useCount,useInds) / np.sum(useCount)

      #--[2] Get the row and ones before it. Compute column average.
      #      
      Irows = Ip[topInd:botInd,:]
      ibin, jbin = np.nonzero(Irows)          # y,x in OpenCV

      self.tpt = np.array([np.mean(jbin), crow]).reshape(-1,1)
      self.haveMeas = True

    mstate = self.getState()
    return mstate

#
#================================ toplines ===============================
