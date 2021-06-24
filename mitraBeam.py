#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 00:57:12 2018

@author: yannis
"""

## Calculate the SQUARE equation (32) in Cornish 2001


import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from detecDir import GW_Detector, GravWave, beamPatternF
from matplotlib import cm
from scipy import constants as const
from scipy import integrate
import healpy as hp


DecIndices = 300j
PhiIndices = 450j

decGrid, phiGrid = np.mgrid[-90 : 90 : DecIndices , 0 : 360 : PhiIndices]
LambdaInvEl = 0 * decGrid
bigIntegral = 0 * decGrid

Dec0 = 12
rightA0 = 0
Omega0 = -GravWave( dec= Dec0 , rightA=rightA0 ).kVec

Omega0GridIndices = [np.int(np.around(Dec0/360 * np.imag(DecIndices))),np.int(np.around(rightA0/360 * np.imag(PhiIndices)))] 
print(Omega0GridIndices)

print(Omega0)
OmegaVec = -GravWave(dec= decGrid , rightA=phiGrid).kVec

siderealDay = 86400 # in seconds
DeltaT = 192 # in seconds
aIntervals = np.int(siderealDay / DeltaT) #### 86400/192 = 450


#### This code works for now only if the timeArray has length decGrid.shape[1]
### because only in this case we can use the symmetry between the passage of
### siderial time and rotations in the right ascenion
### Thus:
alphaRay = np.linspace(0,2*np.pi,decGrid.shape[1])

#alphaRay = np.linspace(0,2*np.pi,aIntervals)

print(len(alphaRay))


#mySiderealTime =+np.pi/5
frequencyUpper = 1024

multiShape = (len(alphaRay),decGrid.shape[0],decGrid.shape[1])
Fplus1 = np.zeros(multiShape)
Fcross1 = np.zeros(multiShape)
Fplus2 = np.zeros(multiShape)
Fcross2 = np.zeros(multiShape)
myInnerProduct = np.zeros(multiShape) 
myBeamGamma = np.zeros(multiShape)
GammaSquare = np.zeros(multiShape)
mySinc = np.zeros(multiShape)
DeltaXvec = np.zeros((len(alphaRay),3))

myNum = 0
for itTime in np.arange(len(alphaRay)): # len(alphaRay) ~ 1400
    LHO = GW_Detector(46.45528,119.4078,-90+36,alphaRay[itTime],'LIGO Hanford Observatory')
    LLO = GW_Detector(30.56278,90.77417,18, alphaRay[itTime],'LIGO Livingston Observatory')
#   auxDet = GW_Detector(90.0,0.0,0.0, alphaRay[itTime],'Auxiliary Observatory')     
    currentDet1 = LHO
    currentDet2 = LLO
    
    DeltaXvec[itTime,:] = 2998000*(LHO.detectorPos - LLO.detectorPos)/np.linalg.norm(LHO.detectorPos - LLO.detectorPos)
    
    for myDecIt in np.arange(decGrid.shape[0]):
        for myPhiIt in np.arange(decGrid.shape[1]):
            if itTime == 0:

                Fplus1[itTime,myDecIt,myPhiIt] = beamPatternF( currentDet1 , GravWave(decGrid[myDecIt,0],phiGrid[0,myPhiIt],0), '+')
                #print(Fplus1.shape)
                Fcross1[itTime,myDecIt,myPhiIt] = beamPatternF( currentDet1 , GravWave(decGrid[myDecIt,0],phiGrid[0,myPhiIt],0), 'x')
                #print(Fcross1.shape)
                
                Fplus2[itTime,myDecIt,myPhiIt] = beamPatternF( currentDet2 , GravWave(decGrid[myDecIt,0],phiGrid[0,myPhiIt],0), '+')
                Fcross2[itTime,myDecIt,myPhiIt] = beamPatternF( currentDet2 , GravWave(decGrid[myDecIt,0],phiGrid[0,myPhiIt],0), 'x')
    
    
                myBeamGamma[itTime,myDecIt,myPhiIt] = (Fplus1[itTime,myDecIt,myPhiIt] * Fplus2[itTime,myDecIt,myPhiIt] +
                                                       Fcross1[itTime,myDecIt,myPhiIt] * Fcross2[itTime,myDecIt,myPhiIt])
                
                GammaSquare[itTime,myDecIt,myPhiIt] = myBeamGamma[itTime,myDecIt,myPhiIt] * myBeamGamma[itTime,myDecIt,myPhiIt]
                
                #print(GravWave(dec= decGrid , rightA=phiGrid).kVec.shape)
                
                #print(DeltaXvec)
                
    #            #computing Omega dot Dx for all theta (declinations) and phi
    #            myInnerProduct = 0 * decGrid 
    #            for decIt in range(decGrid.shape[0]):
    #                for phiIt in range(decGrid.shape[1]):
    #                    myInnerProduct[decIt,phiIt] += np.dot(OmegaVec[:,decIt,phiIt] - Omega0,DeltaXvec)
                           
            else:
                Fplus1[itTime,myDecIt,myPhiIt] = Fplus1[0,myDecIt,(myPhiIt-itTime)%len(alphaRay)]
                Fcross1[itTime,myDecIt,myPhiIt] = Fcross1[0,myDecIt,(myPhiIt-itTime)%len(alphaRay)]
                Fplus2[itTime,myDecIt,myPhiIt] = Fplus2[0,myDecIt,(myPhiIt-itTime)%len(alphaRay)]
                Fcross2[itTime,myDecIt,myPhiIt] = Fcross2[0,myDecIt,(myPhiIt-itTime)%len(alphaRay)]
                myBeamGamma[itTime,myDecIt,myPhiIt] = myBeamGamma[0,myDecIt,(myPhiIt-itTime)%len(alphaRay)]
                GammaSquare[itTime,myDecIt,myPhiIt] = GammaSquare[0,myDecIt,(myPhiIt-itTime)%len(alphaRay)]
            
            myInnerProduct[itTime,myDecIt,myPhiIt] = np.dot(OmegaVec[:,myDecIt,myPhiIt] - Omega0 , DeltaXvec[itTime])
            
            mySinc[itTime,myDecIt,myPhiIt] = np.sinc( 2 * frequencyUpper * myInnerProduct[itTime,myDecIt,myPhiIt] / const.c)
            
            myNum += 1
            if not myNum % 10000 :
                print(myNum)


print(myInnerProduct.shape)

for myDecIt in np.arange(decGrid.shape[0]):
    for myPhiIt in np.arange(decGrid.shape[1]):
        
        if myPhiIt ==0:
            LambdaInvEl[myDecIt,myPhiIt] = integrate.simps(GammaSquare[:,myDecIt,myPhiIt], x=alphaRay)
        else:
            LambdaInvEl[myDecIt,myPhiIt] = LambdaInvEl[myDecIt,0]
            
        bigIntegral[myDecIt,myPhiIt] = integrate.simps(myBeamGamma[:,Omega0GridIndices[0],Omega0GridIndices[1]] * 
                                                   myBeamGamma[:,myDecIt,myPhiIt] * mySinc[:,myDecIt,myPhiIt],x=alphaRay)

LambdaMatrix = np.reciprocal(LambdaInvEl)


myTsidIndex = 0
#R = GammaSquare[myTsidIndex,:,:]
#R = myInnerProduct[myTsidIndex,:,:]
R = bigIntegral * LambdaMatrix
#R = LambdaMatrix


#fig = plt.figure(27,figsize=(12,6))
#fig.suptitle(r'{0} and {1} $\Gamma (\Omega , t)$ function at $t_{{sid}} = {2}$ rad'.format(currentDet1.name,currentDet2.name,alphaRay[myTsidIndex])) 
#ax = fig.add_subplot(1,1,1, projection='3d')
#ax.set_xlabel('xAxis', fontsize=10)
#ax.set_ylabel('yAxis', fontsize=9)
#ax.set_zlabel('zAxis', fontsize=9)
#
#plt.gca().set_aspect('equal', adjustable='box')
##X = R * np.cos(np.radians(decGrid)) * np.cos(np.radians(phiGrid))
##Y = R * np.cos(np.radians(decGrid)) * np.sin(np.radians(phiGrid))
##Z = R * np.sin(np.radians(decGrid))
#
##coloured unit shpere plot
#X = np.cos(np.radians(decGrid)) * np.cos(np.radians(phiGrid))
#Y = np.cos(np.radians(decGrid)) * np.sin(np.radians(phiGrid))
#Z = np.sin(np.radians(decGrid))
#
#N = ( R + np.abs(R.min())) / (R + np.abs(R.min())).max()
#
#plot = ax.plot_surface(
#    X, Y, Z, rstride=1, cstride=1, facecolors=cm.gnuplot2(N),
#    linewidth=0, antialiased=False, alpha=0.5)
#
#plt.show()

#you can even use nside that is not a power of 2
nside = 16

npix = hp.nside2npix(nside)
thetas =np.pi / 2 - decGrid.flatten() * np.pi / 180.0
phis = phiGrid.flatten() * np.pi * 2 / 360.0
fs = R.flatten()

indices = hp.ang2pix(nside, thetas,phis)
hpxmap = np.zeros(npix, dtype=np.float)
numContrib = np.ones(npix, dtype=np.int8)

for myIter in range(len(indices)):
    numContrib[indices[myIter]] += 1
    hpxmap[indices[myIter]] += fs[myIter]

for ii in range(len(numContrib)):
    if numContrib[ii] > 1:
        numContrib[ii] -= 1
        
hpxmap = hpxmap / numContrib

mycmap = cm.gnuplot2
mycmap.set_under('w')

#hpxmap_smooth = hp.smoothing(hpxmap, fwhm=np.radians(1.))

hp.mollview(hpxmap,cmap = mycmap)
plt.show()


with open("GammaMatrix_{}times_{}decs_{}phis.txt".format(len(alphaRay),decGrid.shape[0],decGrid.shape[1]),'w+b') as myFile_G:
    for itTime in range(len(alphaRay)):
        R_G = myBeamGamma[itTime,:,:]
        fs_G = R_G.flatten()
        hpxmap_G = np.zeros(npix, dtype=np.float)
        numContrib_G = np.ones(npix, dtype=np.uint16)
        
        for myIter in range(len(indices)):
            numContrib_G[indices[myIter]] += 1
            hpxmap_G[indices[myIter]] += fs_G[myIter]
        
        for ii in range(len(numContrib_G)):
            if numContrib_G[ii] > 1:
                numContrib_G[ii] -= 1
        hpxmap_G = hpxmap_G / numContrib_G
        np.savetxt(myFile_G,hpxmap_G,delimiter=' ',newline=' ', fmt='%.6f')
        myFile_G.write(b'\n')
        
#with open("GammaMatrix_{}times_{}decs_{}phis.txt".format(len(alphaRay),decGrid.shape[0],decGrid.shape[1]),'rb') as myFile_G:
#    for ii in list(myFile_G):
#        print(ii)


