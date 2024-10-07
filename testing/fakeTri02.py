#!/usr/bin/python
#=============================== fakeTri02 ===============================
#
# @brief    Code to create an image sequence of a rigid body moving along a
#           twist trajectory (constant body velocity) with the presumed
#           marker measurements that would result. Marker can be a filled
#           area or a
#
#
# This script helps to understand how to generate fake marker imagery to
# test out codebase for processing marker movement. There are two options
# provided, one is with a moving set of three markers in a triangle shape.
# They move rigidly together.  The other is no markers and the triangle is
# a solid shape.
#
#=============================== fakeTri02 ===============================

#
# @file     fakeTri02.m
#
# @author   Patricio A. Vela,   pvela@gatech.edu
#           Yunzhi Lin,         yunzhi.lin@gatech.edu
# @date     2020/09/30 [created]
#           2021/07/13 [modified]
#!NOTE:
#!  Indent is set to 2 spaces.
#!  Tab is set to 4 spaces with conversion to spaces.
#
# @quit
#=============================== fakeTri02 ===============================

#==[0] Prep environment.
#
import numpy as np
import matplotlib.pyplot as plt

from trackpointer.utils.fakeTriangle import fakeTriangle
import Lie.group.SE2.Homog

#==[1] Specify the marker geometry on the rigid body, and other related
#       parameters. Instantiate the simulated image generator.
#
# Currently, x,y follow the OpenCV coordinate system but not the Matlab one
pMark  = np.array([[-12, -12, 12],[18, -18, 0],[1, 1, 1]])+np.array([[120],[120],[0]]) # Define a triangle
sMark  = 2*np.array([[ -2, -2, 2, 2],[-2, 2, 2, -2],[0, 0, 0, 0]]) # For each vertice
imSize = np.array([200, 200])

# @todo comment this line to enable useMarkers or not
# useMarkers = False
useMarkers = True

if useMarkers:
  ftarg = fakeTriangle(pMark, sMark, imSize)
else:
  ftarg = fakeTriangle(pMark, None, imSize)

#==[2] Define initial condition, set the pose and the relative
#       transformation, then render the image sequence in a for loop.
#
theta = 0
R = Lie.group.SE2.Homog.rotationMatrix(theta)
x = np.array([[200],[200]])
g = Lie.group.SE2.Homog(R=R, x=x)
ftarg.setPose(g)

plt.ion()

# Render first frame @ initial pose.
I = ftarg.render()
plt.imshow(I, cmap='Greys')

for ii in range(25):# Loop to udpate pose and re-render.
  theta = np.pi/50
  R = Lie.group.SE2.Homog.rotationMatrix(theta)
  g = g * Lie.group.SE2.Homog(x=np.array([[0],[0]]), R=R)
  ftarg.setPose(g)
  I = ftarg.render()

  plt.cla()

  plt.imshow(I, cmap='Greys')
  plt.pause(0.001)

plt.ioff()
plt.draw()

#
#=============================== fakeTri02 ===============================
