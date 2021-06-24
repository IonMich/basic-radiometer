# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 15:40:55 2018

@author: Ioannis Mich
"""
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from detecDir import GW_Detector, GravWave, beamPatternF
from matplotlib import cm

mySiderealTime =0
LHO = GW_Detector(46.45528,119.4078,-90+36,mySiderealTime,'LIGO Hanford Observatory')
LLO = GW_Detector(30.56278,90.77417,18, mySiderealTime,'LIGO Livingston Observatory')
auxDet = GW_Detector(90.0,0.0,0.0, mySiderealTime,'Auxiliary Observatory')

decGrid, phiGrid = np.mgrid[-90 : 90 : 30j , 0 : 360 : 40j]

currentDet = auxDet

FplusSq = beamPatternF( currentDet , GravWave(decGrid,phiGrid,0), '+')**2
print(FplusSq.shape)
FcrossSq = beamPatternF( currentDet , GravWave(decGrid,phiGrid,0), 'x')**2
print(FcrossSq.shape)

# theoretical values for dec=90 , rightAsc=0
FplusThSq = np.power(0.5 * np.multiply( (1 + np.sin(np.radians(decGrid))**2) , np.cos( 2*np.radians(phiGrid) ) ) , 2.0 )
FcrossThSq = np.power( np.multiply( np.sin(np.radians(decGrid)) , np.sin(2*np.radians(phiGrid)) ) , 2.0 )

R1 = FplusSq
R2 = FcrossSq
R3 = np.multiply( 0.5 , np.add(FplusSq,FcrossSq) ) 

#R1 = FplusSq - FplusThSq
#R2 = FcrossSq - FcrossThSq
#R3 = np.multiply( 0.5 , np.add(FplusSq,FcrossSq) ) - np.multiply( 0.5 , np.add(FplusThSq,FcrossThSq) )

fig = plt.figure(21,figsize=(12,6))
fig.suptitle(r'{0} beam response functions at $t_{{sid}} = {1}$'.format(currentDet.name,mySiderealTime))
#beamList =[R1, R2]
beamList =[R1, R2, R3]



for rIt in range(0, 3):
    R = beamList[rIt]
    ax = fig.add_subplot(1,3,rIt + 1, projection='3d')
#    ax.set_xlim(-0.5, 0.5)
#    ax.set_ylim(-0.5, 0.5)
#    ax.set_zlim(-1, 1)
#    ax.view_init(elev=0, azim=0)
    ax.set_xlabel('xAxis', fontsize=10)
    ax.set_ylabel('yAxis', fontsize=9)
    ax.set_zlabel('zAxis', fontsize=9)
    if rIt == 0:
        ax.set_title('Plus Polarization')
    elif rIt==1:
        ax.set_title('Cross Polarization')
    else:
        ax.set_title('Averaged Polarization')
    plt.gca().set_aspect('equal', adjustable='box')
    X = R * np.cos(np.radians(decGrid)) * np.cos(np.radians(phiGrid))
    Y = R * np.cos(np.radians(decGrid)) * np.sin(np.radians(phiGrid))
    Z = R * np.sin(np.radians(decGrid))
    
    N = R/R.max()
    plot = ax.plot_surface(
        X, Y, Z, rstride=1, cstride=1, facecolors=cm.jet(N),
        linewidth=0, antialiased=False, alpha=0.5)
    
plt.show()

#fig.savefig('beamResponseAux.pdf' )