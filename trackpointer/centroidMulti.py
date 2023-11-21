#============================== centroidMulti ==============================
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
#============================== centroidMulti ==============================

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
#!  
#
#
#============================== centroidMulti ==============================

import numpy as np
import cv2

import skimage.morphology as morph
from skimage.measure import regionprops, label
from trackpointer.centroid import centroid, State, CfgCentroid



#
#---------------------------------------------------------------------------
#==================== Configuration Node : centroidMulti ===================
#---------------------------------------------------------------------------
#

class CfgCentMulti(CfgCentroid):
  '''!
  @brief  Configuration setting specifier for centroidMulti.

  minArea   - Minimum area acceptable (anything less is not a target).
  maxArea   - Maximum area acceptable (anything more is not a target).
  measProps - Flag to determine whether to keep the region properties.
  keepLabel - Flag to keep the label image, in case needed later.

  '''
  #============================= __init__ ============================
  #
  '''!
  @brief        Constructor of configuration instance.

  @param[in]    cfg_files   List of config files to load to merge settings.
  '''
  def __init__(self, init_dict=None, key_list=None, new_allowed=True):

    if (init_dict == None):
      init_dict = CfgCentMulti.get_default_settings()

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
    default_dict = dict(minArea = 0, maxArea = float('inf'), \
                        regConn = 1, \
                        measProps = False, keepLabel = False)
    return default_dict

#========================== builtForPuzzles =========================
#
#
  @staticmethod
  def builtForLearning():
    puzzleCfg = CfgCentMulti();
    puzzleCfg.minArea = 60
    puzzleCfg.maxArea = 300
    return puzzleCfg


#
#---------------------------------------------------------------------------
#============================== centroidMulti ==============================
#---------------------------------------------------------------------------
#

class centroidMulti(centroid):

  #============================ centroidMulti ============================
  #
  # @brief      Centroid track-pointer constructor.
  #
  # @param[in]  iPt     The initial track point coordinates.
  #             params   The parameter structure.
  #
  def __init__(self, iPt=None, params=CfgCentMulti()):

    super(centroidMulti,self).__init__(iPt, params)

    self.labelIm    = None
    self.trackProps = None


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
      Ip = np.copy(I)

    # [08/30 PAV: CODE BELOW COMMENTED OUT DUE TO BEING SLOW AND KINDA CRAPPY.]
    # [09/07 PAV: Also seems redundant since it runs regionprops anyhow.
    #             Looks like uses openCV for labels, but method is no good.   ]
    #binReg = centroidMulti.regionProposal(Ip)
    #self.tpt = np.array(binReg).T # from N x 2 to 2 x N

    # Link to scikit [region props](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops)

    if (self.tparams.minArea > 0):
      morph.remove_small_objects(Ip, self.tparams.minArea, 1, out = Ip)

    (Il, nl) = label(Ip, None, True, self.tparams.regConn)
    # @todo Consider how might use nl return value.

    if self.tparams.keepLabel:
      self.labelImage = Il

    if self.tparams.measProps:
      self.trackProps = regionprops(Il)

      # @todo   Address the maxArea option.
      binReg = []
      for ri in self.trackProps:
        if (ri.area < self.tparams.maxArea):
          binReg.append([ri.centroid[1], ri.centroid[0]])
    else:
      binReg = []
      for ri in regionprops(Il):
        binReg.append([ri.centroid[1], ri.centroid[0]])

    self.tpt = np.array(binReg).T # from N x 2 to 2 x N

    if len(self.tpt) == 0:
      self.haveMeas = False
    else:
      self.haveMeas = self.tpt.shape[1] > 0

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

  #========================= regionProposal =========================
  #
  # @brief  Find out the centroid for multiple objects
  #
  # @param[in]  I   The input mask image.
  #
  @ staticmethod
  def regionProposal(I):
    # [08/03 PAV: Code below is horrendous.  What is purpose here? There are better methods.]
    # [             Even the stackoverflow link below has one.                              ]
    Ip = I.astype(np.uint8)
    mask = np.zeros_like(Ip)
    cnts = cv2.findContours(Ip, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    # For OpenCV 4+
    cnts = cnts[0]
    
    for idx, cnt in enumerate(cnts):
      cv2.drawContours(mask, cnt, -1, (idx+1), 1)

    # [ 09/07 PAV: Above code can be implemented with scikit ``labels``. ]
    # Note that regionprops assumes different areas are with different labels
    # See https://stackoverflow.com/a/61591279/5269146

    # Have been transferred to the OpenCV style as regionprops is from skimage
    binReg = [[i.centroid[1], i.centroid[0]] for i in regionprops(mask)]

    return binReg
#
#================================ centroidMulti ===============================
