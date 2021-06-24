#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 20:38:12 2018

@author: yannis
"""


## Calculate the equation (32) in Cornish 2001


import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from detecDir import GW_Detector, GravWave, beamPatternF
from matplotlib import cm
from scipy import constants as const
import healpy as hp

#cornish seems to evaluate everything at sidereal time pi/5 (or do I have a bug?)
mySiderealTime =+np.pi/5
frequency = 200
LHO = GW_Detector(46.45528,119.4078,-90+36,mySiderealTime,'LIGO Hanford Observatory')
LLO = GW_Detector(30.56278,90.77417,18, mySiderealTime,'LIGO Livingston Observatory')
auxDet = GW_Detector(90.0,0.0,0.0, mySiderealTime,'Auxiliary Observatory')

decGrid, phiGrid = np.mgrid[-90 : 90 : 80j , 0 : 360 : 70j]

currentDet1 = LHO
currentDet2 = LLO
Fplus1 = beamPatternF( currentDet1 , GravWave(decGrid,phiGrid,0), '+')
print(Fplus1.shape)
Fcross1 = beamPatternF( currentDet1 , GravWave(decGrid,phiGrid,0), 'x')
print(Fcross1.shape)

Fplus2 = beamPatternF( currentDet2 , GravWave(decGrid,phiGrid,0), '+')
Fcross2 = beamPatternF( currentDet2 , GravWave(decGrid,phiGrid,0), 'x')

print(GravWave(dec= decGrid,rightA=phiGrid).kVec.shape)
OmegaVec = -GravWave(dec= decGrid,rightA=phiGrid).kVec
DeltaXvec = 2998000*(LHO.detectorPos - LLO.detectorPos)/np.linalg.norm(LHO.detectorPos - LLO.detectorPos)
#print(DeltaXvec)

myInnerProdut = 0 * decGrid 
for decIt in range(decGrid.shape[0]):
    for phiIt in range(decGrid.shape[1]):
        myInnerProdut[decIt,phiIt] += np.dot(OmegaVec[:,decIt,phiIt],DeltaXvec)


myBeamP = (Fplus1 * Fplus2 + Fcross1 * Fcross2) * np.cos( 2 * np.pi* frequency * myInnerProdut / const.c)


#### I am also taking the SQUARE to avoid negative values

#myBeamPSq = np.square(myBeamP)

R = myBeamP


#fig = plt.figure(22,figsize=(12,6))
#fig.suptitle(r'{0} narrowband beam response function at $t_{{sid}} = {1}$'.format(currentDet1.name,mySiderealTime)) 
#ax = fig.add_subplot(1,1,1, projection='3d')
#ax.set_xlabel('xAxis', fontsize=10)
#ax.set_ylabel('yAxis', fontsize=9)
#ax.set_zlabel('zAxis', fontsize=9)
#
#plt.gca().set_aspect('equal', adjustable='box')
#X = R * np.cos(np.radians(decGrid)) * np.cos(np.radians(phiGrid))
#Y = R * np.cos(np.radians(decGrid)) * np.sin(np.radians(phiGrid))
#Z = R * np.sin(np.radians(decGrid))
#N = ( R + np.abs(R.min())) / (R + np.abs(R.min())).max()
#
#plot = ax.plot_surface(
#    X, Y, Z, rstride=1, cstride=1, facecolors=cm.jet(N),
#    linewidth=0, antialiased=False, alpha=0.5)
#
#plt.show()


nside =8
npix = hp.nside2npix(nside)
thetas =np.pi / 2 - decGrid.flatten() * np.pi / 180.0
phis = phiGrid.flatten() * np.pi * 2 / 360.0
fs = R.flatten()

indices = hp.ang2pix(nside, thetas,phis)

hpxmap = np.zeros(npix, dtype=np.float)
numContrib = np.ones(npix, dtype=np.float)

for myIter in range(len(indices)):
    numContrib[indices[myIter]] += 1
    hpxmap[indices[myIter]] += fs[myIter]
    
hpxmap = hpxmap / numContrib






mycmap = cm.jet
mycmap.set_under('w')

#hpxmap_smooth = hp.smoothing(hpxmap, fwhm=np.radians(1.))
hp.mollview(hpxmap,cmap = mycmap)
plt.show()




#compute Cl's
# we need both even (cos) and odd (sin) cntributions to Cl so we compute also the qI=1 angular pattern
myBeamP2 = (Fplus1 * Fplus2 + Fcross1 * Fcross2) * np.sin( 2 * np.pi* frequency * myInnerProdut / const.c)
fs2 = myBeamP2.flatten()

hpxmap2 = np.zeros(npix, dtype=np.float)
numContrib2 = np.ones(npix, dtype=np.float)

for myIter2 in range(len(indices)):
    numContrib2[indices[myIter2]] += 1
    hpxmap2[indices[myIter2]] += fs2[myIter2]
    
hpxmap2 = hpxmap2 / numContrib2

LMAX=50
cl1 = hp.anafast(hpxmap, lmax=LMAX)
cl2 = hp.anafast(hpxmap2, lmax=LMAX)
cl = cl1+cl2
ell = np.arange(len(cl))

#
#plt.figure(43)
#plt.plot(ell, np.sqrt(cl))
#plt.xlabel('ell'); plt.ylabel('cl');
#plt.show

plt.figure(44)
ellplotted = ell - 0.5
ellplotted = np.append(ellplotted,ellplotted[-1]+1)
clplotted = np.insert(cl, 0, cl[0])
ax = plt.step(ellplotted,np.sqrt(clplotted), label='200Hz')
plt.legend(loc='upper right')
plt.xlim(0,20)
plt.ylim(0,0.2)
plt.show()












