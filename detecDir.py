# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 22:37:50 2018

@author: Ioannis Mich
"""

## Earth Fixed

import numpy as np
#import astropy.constants as ap
#import scipy.constants as sp
import matplotlib.pyplot as plt
import math as m
#from pynverse import inversefunc
from mpl_toolkits.mplot3d import Axes3D
#from itertools import product, combinations
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


class GW_Detector:
    def __init__(self, latitude = 0, longitude = 0, azimuthWest = 0, gamma=0, name='LIGO Detector'):
        self.long = np.radians(longitude)
        self.lat = np.radians(latitude)
        ###here azimuthW =0 in the West and increases to the South
        ##fix cosines/sines so that there is only self.azim defined
        self.azimW = np.radians(azimuthWest)
        
        self.azim = (270 - self.azimW) % 360
        
        self.name = name
        
        self.rVec = np.array([np.cos(self.long) * np.cos(self.lat), - np.sin(self.long) * np.cos(self.lat) , np.sin(self.lat) ])
        uVecAux = np.array([-np.sin(self.long),-np.cos(self.long),0])
        vVecAux = np.array([np.cos(self.long) * np.sin(self.lat), - np.sin(self.long) * np.sin(self.lat) , - np.cos(self.lat) ])
        
        self.uVec = uVecAux * np.cos(self.azimW) + vVecAux * np.sin(self.azimW)
        self.vVec = vVecAux * np.cos(self.azimW) - uVecAux * np.sin(self.azimW)
      
        
        # rotation from sidereal time (radians)
        self.gamma = gamma;
        gstRot = np.matrix([[np.cos(self.gamma), -np.sin(self.gamma), 0], [np.sin(self.gamma), np.cos(self.gamma), 0],[0, 0, 1]])
        self.uVec = np.asarray(gstRot.dot(self.uVec)).reshape(-1)
        self.vVec = np.asarray(gstRot.dot(self.vVec)).reshape(-1)
        self.rVec = np.asarray(gstRot.dot(self.rVec)).reshape(-1)
        
        self.detectorPos = self.rVec
        
        self.detTensor = 0.5 * ( np.outer(self.uVec,self.uVec) - np.outer(self.vVec,self.vVec) )
        
    def draw(self,plotAxis,scale=0.3,thecolor = 'k'):
        self.scale = scale
        self.color = thecolor
        self.XarmX,self.YarmX,self.ZarmX = zip(self.detectorPos,np.add(self.uVec * self.scale,self.detectorPos))
        self.XarmY,self.YarmY,self.ZarmY = zip(self.detectorPos,np.add(self.vVec * self.scale,self.detectorPos))
        mya = Arrow3D( self.XarmX , self.YarmX , self.ZarmX , mutation_scale=10,
            lw=2, arrowstyle="-|>", color=thecolor)
        myb = Arrow3D( self.XarmY , self.YarmY , self.ZarmY , mutation_scale=10,
            lw=2, arrowstyle="-|>", color=thecolor)
        plotAxis.add_artist(mya)
        plotAxis.add_artist(myb)
        
        # draw a point
        self.xx, self.yy , self.zz = zip( self.rVec)
        plotAxis.scatter(self.xx, self.yy, self.zz, color="g", s=100)


class GravWave:
    def __init__(self, dec=0, rightA=0 , polAngle=0):
        ##TODO: edit to allow for dec , rightA to be np Arrays or np Grids
        self.dec = np.radians(dec)
        self.rightA = np.radians(rightA)
        self.psi = np.radians(polAngle) 
        self.kVec = np.array([np.multiply( -np.cos(self.rightA) , np.cos(self.dec) ),
                              np.multiply( -np.sin(self.rightA) , np.cos(self.dec) ),
                              - np.sin(self.dec) ])
        
        self.iVec = np.array([np.sin(self.rightA),
                              -np.cos(self.rightA),
                              0*np.sin(self.rightA) ]) ## replaced 0 by 0 * np.sin(...)= 0 in order to fix dimensions in the case of Grid Angles
        self.jVec = - np.array([np.multiply( np.cos(self.rightA) , np.sin(self.dec) ),
                                np.multiply( np.sin(self.rightA) , np.sin(self.dec) ),
                                -np.cos(self.dec) ])
        
        if self.iVec[0].shape !=():
            self.matrixAngles = True
        else:
            self.matrixAngles = False
            
        # done (correct): Check if sign of psi is correct
#        print(np.multiply( self.iVec , np.cos(self.psi) ).shape)
#        print(np.multiply( self.jVec , np.sin(self.psi) ).shape)
        self.lVec = np.multiply( self.iVec , np.cos(self.psi) ) + np.multiply( self.jVec , np.sin(self.psi) )
        self.mVec = np.multiply( self.jVec , np.cos(self.psi) ) - np.multiply( self.iVec , np.sin(self.psi) )
        

        if self.matrixAngles == True:
            self.decSize = self.lVec[0,:,0].size
            self.rightSize = self.lVec[0,0,:].size
            self.tensorPlus = np.zeros( (3 , 3 , self.decSize , self.rightSize) )
            self.tensorCross = np.zeros( (3 , 3 , self.decSize , self.rightSize) )
            #print(self.tensorPlus.shape)
            for i in range(3):
                for j in range(3):
                    for k in range(self.decSize):
                        for n in range(self.rightSize):
                            self.tensorPlus[i,j,k,n] = self.lVec[i,k,n] * self.lVec[j,k,n] - \
                                                    self.mVec[i,k,n] * self.mVec[j,k,n]
                            self.tensorCross[i,j,k,n] = self.lVec[i,k,n] * self.mVec[j,k,n] + \
                                                    self.mVec[i,k,n] * self.lVec[j,k,n]
        else:
            self.tensorPlus = ( np.outer(self.lVec,self.lVec) - np.outer(self.mVec,self.mVec) )
            self.tensorCross = ( np.outer(self.lVec,self.mVec) + np.outer(self.mVec,self.lVec) )            
#            print(self.tensorPlus.shape)

        
    def draw(self, plotAxis, scale=0.5, thecolor = 'g'):
        self.scale = scale
        self.color = thecolor
        dirOrigin = -1.5 * self.kVec
        self.impactPoint = - 1.0 *self.kVec
        self.xx, self.yy , self.zz = zip(dirOrigin,np.add( self.kVec * self.scale,dirOrigin))
        self.xxi,self.yyi,self.zzi = zip(self.impactPoint,np.add(self.iVec * self.scale,self.impactPoint))
        self.xxj,self.yyj,self.zzj = zip(self.impactPoint,np.add(self.jVec * self.scale,self.impactPoint))
        
        self.xxl,self.yyl,self.zzl = zip(self.impactPoint,np.add(self.lVec * self.scale,self.impactPoint))
        self.xxm,self.yym,self.zzm = zip(self.impactPoint,np.add(self.mVec * self.scale,self.impactPoint))
        
        myi = Arrow3D( self.xxi , self.yyi , self.zzi , mutation_scale=10,
            lw=2, arrowstyle="-|>", color=thecolor)
        myj = Arrow3D( self.xxj , self.yyj , self.zzj , mutation_scale=10,
            lw=2, arrowstyle="-|>", color=thecolor)
        myl = Arrow3D( self.xxl , self.yyl , self.zzl , mutation_scale=10,
            lw=2, arrowstyle="-|>", color='b')
        mym = Arrow3D( self.xxm , self.yym , self.zzm , mutation_scale=10,
            lw=2, arrowstyle="-|>", color='b')
        
        myk = Arrow3D( self.xx , self.yy , self.zz , mutation_scale=10,
            lw=2, arrowstyle="-|>", color=thecolor)
        
        plotAxis.add_artist(myi)
        plotAxis.add_artist(myj)
        plotAxis.add_artist(myk)
        plotAxis.add_artist(myl)
        plotAxis.add_artist(mym)


def beamPatternF(detector, gravWave, polarization=None):
    ####TODO: I should change this part to make it more readable: 
    ## the first method uses the polarizationTensor, while the second one uses the basis polarization vectors l m
    ## both methods work in the case of matrix angles but the for loop method fails for scalar angles 
    ## the second method (one liner) is probably faster but I should check
    if polarization == '+':
#        print(detector.detTensor.shape)
#        print(gravWave.lVec.shape)
#        
#        auxTen = np.tensordot (detector.detTensor , gravWave.lVec, axes=1)
#        print(auxTen.shape)
#        auxMulti = np.multiply( gravWave.lVec , auxTen )
#        print(auxMulti.shape)
#        auxMultiF = np.sum(auxMulti,axis =0)
#        print(auxMultiF.shape)

        if gravWave.matrixAngles == True :
            fPlus = np.zeros( (gravWave.decSize , gravWave.rightSize) )
            for i in range(3):
                for j in range(3):
                    for k in range(gravWave.decSize):
                        for n in range(gravWave.rightSize):
                            fPlus[k,n] += detector.detTensor[i,j] * gravWave.tensorPlus[i,j,k,n]

            return fPlus
        else:
            fPlusSimple = np.sum(np.multiply( gravWave.lVec , np.tensordot (detector.detTensor , gravWave.lVec, axes=1) ) , axis =0) - \
                        np.sum(np.multiply( gravWave.mVec , np.tensordot (detector.detTensor , gravWave.mVec, axes=1) ) , axis =0)
            return fPlusSimple     
#        return np.tensordot( gravWave.lVec , np.tensordot (detector.detTensor , gravWave.lVec,axes=1) , axes=1 ) - \
#            np.tensordot( gravWave.mVec, np.tensordot( detector.detTensor , gravWave.mVec,axes=1) , axes=1)
    elif polarization == 'x':
        if gravWave.matrixAngles == True :
            fCross = np.zeros( (gravWave.decSize , gravWave.rightSize) )
            for i in range(3):
                for j in range(3):
                    for k in range(gravWave.decSize):
                        for n in range(gravWave.rightSize):
                            fCross[k,n] += detector.detTensor[i,j] * gravWave.tensorCross[i,j,k,n]

            return fCross
        else:
            fCrossSimple = np.sum(np.multiply( gravWave.lVec , np.tensordot (detector.detTensor , gravWave.mVec, axes=1) ) , axis =0) + \
                        np.sum(np.multiply( gravWave.mVec , np.tensordot (detector.detTensor , gravWave.lVec, axes=1) ) , axis =0)
            return fCrossSimple 
#            return gravWave.mVec.dot( detector.detTensor.dot(gravWave.mVec) ) + gravWave.mVec.dot( detector.detTensor.dot(gravWave.lVec) )
    else:
        raise ValueError('Please specify the polarization to be either \'+\' or \'x\'.')
        
if __name__ == "__main__":
    
    fig = plt.figure(figsize=(12,12))
    ax = fig.gca(projection='3d')
    ###### TODO: make it work for mollweide projection
    ##idea : (x,y,z) -> (xMoll,yMoll) wolfram article
    
    ##No view_init and NO wireframe and NO zlabel in Mollweide Projection
#    ax = fig.gca(projection='mollweide')

    ax.view_init(elev=30, azim=45)
    
    ax.set_aspect("equal")
    
    # draw sphere
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="r")
    ax.set_xlabel('x',fontsize=20)
    ax.set_ylabel('y',fontsize=20)
    ax.set_zlabel('z',fontsize=20)
    
    mySiderealTime = np.pi/2 #RADIANS
    
    ###TODO: fix azimuth definition
    LHO = GW_Detector(46.45528,119.4078,-90+35.9994,mySiderealTime,'LIGO Hanford Observatory')
    LLO = GW_Detector(30.56278,90.77417,17.7166, mySiderealTime,'LIGO Livingston Observatory')
    auxDet = GW_Detector(90.0,0.0,90.0, mySiderealTime,'Auxiliary Observatory')
    LHO.draw(ax,scale=0.3)
    LLO.draw(ax,scale=0.3)
#    auxDet.draw(ax,scale=0.3)
    
    myGW = GravWave(0.0,0.0,0.0)
    myGW.draw(ax,scale=0.3)
    
    detAngle = np.arccos(LLO.rVec.dot(LHO.rVec))
    # NOT great circle
    detDist = 2.0 * 6371 * np.sin( detAngle / 2.0)
    
    print( beamPatternF(auxDet,myGW,'+')**2 )
    
    print('Detector angle difference is',  '{:.2f}'.format(m.degrees(detAngle)), 'degrees')
    print('Detector distance is',  '{:.2f}'.format(detDist), 'km')

    print('\n',LHO.name)
    print('X arm',LHO.uVec)
    print('Y arm',LHO.vVec)
    print('Detector tensor\n',LHO.detTensor)
    print('Detector tensor trace\n',np.trace(LHO.detTensor) )
    
    print('\n',LLO.name)
    print('X arm',LLO.uVec)
    print('Y arm',LLO.vVec)
    print('Detector tensor\n',LLO.detTensor)
    print('Detector tensor trace\n',np.trace(LLO.detTensor) )
    
    print('\n',auxDet.name)
    print('X arm',auxDet.uVec)
    print('Y arm',auxDet.vVec)
    print('Detector tensor\n',auxDet.detTensor)
    
    print('')
    print('GW l vector',myGW.lVec)
    print('GW m vector',myGW.mVec)
    
    
#    fig.savefig("earthSphere.pdf", bbox_inches='tight')
    
    plt.show()


