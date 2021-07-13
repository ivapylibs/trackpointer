#============================== fakeTriangle =============================
#
# @brief    Class that synthesizes measurements of a triangular set
#           of markers to test out a tracking system for them.
#
#============================== fakeTriangle =============================

#
# @file     fakeTriangle.m
#
# @author   Patricio A. Vela,       pvela@gatech.edu
#           Yunzhi Lin,             yunzhi.lin@gatech.edu
# @date     2020/09/30 [created]
#           2021/07/13 [modified]
#! NOTE:
#!  indent in 2 spaces.
#!  tab is 4 spaces with conversion.
#
#============================== fakeTriangle =============================

# import Lie

import Lie.group.SE2.Homog

import numpy as np
import warnings
from skimage import draw


class fakeTriangle(object):

  #============================ fakeTriangle ===========================
  #
  # @brief  Instantiate a fakeTriangle object.
  #
  # @param[in]  pMark   The relative positions.
  # @param[in]  sMark   The shape of the markers (array or cell array).
  #                     Empty if shape is a solid, not multiple small markers.
  # @param[in]  imSize  The image size.
  # @param[in]  gamma   The distance to pixel conversion
  #                       Optional, default = 1)
  #
  def __init__(self, pMark=None, sMark=None, imSize=None, gamma=None):

    self.g = Lie.group.SE2.Homog()
    self.pMark = pMark
    self.isDirty = True

    if sMark is None:
      self.noMarker = True
    elif sMark.ndim!=3: 
      if pMark.shape[1]>1:
        # We put channel in the first dimension
        self.sMark = np.repeat(sMark[:, :, np.newaxis], pMark.shape[1], axis=2).transpose(2,0,1)
      else:
        self.sMark = sMark
      self.noMarker = False
    elif len(sMark) == pMark.shape[1]:
      self.sMark = sMark
      self.noMarker = False
    else:
      warnings('fakeTriangle: pMark and sMark are incompatible sizes.')

    self.imSize = imSize

    if gamma and np.isscalar(gamma):
      self.gamma = gamma


  #============================== setPose ==============================
  #
  # @brief  Set the current SE(2) pose of the fake target.
  #
  #
  def setPose(self, gSet):

    self.g = gSet
    self.isDirty = True


  #=============================== render ==============================
  #
  # @brief  Create a fake image
  #
  # @param[out] I   The rendered image.
  #

  def render(self):

    if self.isDirty:
      self.I = self.synthesize()

    I = self.I
    return I

  # @todo Eventually make methods protected protected. For now public.

  #============================= synthesize ============================
  #
  # @brief  Dirty flag indicates we should create new version.
  #
  def synthesize(self):

    if (not self.isDirty):    #! Check just in case, to avoid recomputing.
      return

    sW = []
    #! Step 1: Map marker centers to new locations. (maybe not be needed)
    pW = self.g * self.pMark # 3*N

    image_shape = (self.imSize[0], self.imSize[1])

    #! Step 2: Map shape (polygon) to new locations.
    if self.noMarker:
      # # @todo
      # # The image coordinate system is different from Matlab
      # # Comment it if we do not care about Matlab
      # pW[[0, 1], :] = pW[[1, 0], :]
      I = draw.polygon2mask(image_shape, pW[:2,:].T)
    else:
      for ii in range(len(self.sMark)):
        sW.append(self.g * (self.pMark[:,ii].reshape(-1,1)) + self.sMark[ii]) # 3*N

      I = np.zeros(self.imSize).astype('bool')
      for ii in range(len(sW)):
        # # @todo
        # # The image coordinate system is different from Matlab
        # # Comment it if we do not care about Matlab
        # sW[ii][[0,1],:] = sW[ii][[1,0],:]
        I = I | draw.polygon2mask(image_shape, sW[ii][:2,:].T)

    return I.astype('uint8')
#
#============================== fakeTriangle =============================