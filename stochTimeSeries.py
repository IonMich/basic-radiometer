#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 13:38:25 2018

@author: yannis
"""

import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.nan)
from matplotlib import cm
import healpy as hp
from scipy import integrate
from detecDir import GW_Detector


def frequency_noise_from_psd(psd, delta_f, seed=None):
    """ Create noise with a given psd.

    Return noise coloured with the given psd. The returned noise
    FrequencySeries has the same length and frequency step as the given psd.
    Note that if unique noise is desired a unique seed should be provided.

    Parameters
    ----------
    psd : FrequencySeries
        The noise weighting to color the noise.
    seed : {0, int} or None
        The seed to generate the noise. If None specified,
        the seed will not be reset.

    Returns
    --------
    noise : FrequencySeriesSeries
        A FrequencySeries containing gaussian noise colored by the given psd.
        # Copyright (C) 2012  Alex Nitz
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!EDITED!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
    """
    sigma = 0.5 * ( psd / delta_f ) ** (0.5)
    if seed is not None:
        np.random.seed(seed)

    not_zero = (sigma != 0)
    #    notZeroTwo = np.append(not_zero, not_zero[-2:-len(psd):-1])
    
    sigma_red = sigma[not_zero]
    noise_re = np.random.normal(0, sigma_red)
    noise_co = np.random.normal(0, sigma_red)
    noise_red = noise_re + 1j * noise_co
    noise= np.zeros(len(psd),dtype = 'complex')

    noise[not_zero] = noise_red
    noise = np.append(noise, np.conjugate(noise[-2:-len(psd):-1]))
    return noise


def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f)-1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real

def colorfft(f):
    f = np.array(f, dtype='complex')
    Np = (len(f)-1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return f

def resamplePSDtoOne(freqs,thePSD):
    #Get the frequencies from log two sided to linearly two-sided
    #Resample PSD at these values
    finalOneFreqs = np.linspace(0,512, 256 + 1 )
#    print(len(finalOneFreqs))
    tmpOneFreqs = np.concatenate(([0], np.linspace(0.1 , freqs[0]-1E-1) , freqs ), axis=None)
    tmpOnePSD =np.concatenate(([0], 0*np.linspace(0.1,freqs[0]-1E-1), thePSD ),axis=None)
    oneRePSD = np.interp(finalOneFreqs,tmpOneFreqs,tmpOnePSD)
    
    return finalOneFreqs , oneRePSD
    

    

def FFTfromPSD(freqs,myPSD):
    #Get TWO-sided ASD from ONE-sided PSD (sampled linearly) 
    # from freq=zero up to some fUpper
    
    #first generate one-sided ASD 
    newFreqs = np.concatenate(( freqs, -freqs[-2:0:-1] ), axis=None)
#    print(len(newFreqs))
    oneASD = np.sqrt(2*myPSD)
    myTwoASD = np.concatenate( ( oneASD/2, oneASD[-2:0:-1]/2 ), axis=None)
    
    return newFreqs,  myTwoASD



#def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
#    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
#    f = np.zeros(samples)
#    idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
#    f[idx] = 1
#    return fftnoise(f)

speedOfLight = 299792458.0

fUpper = 512
deltaF = 2
deltaT = 192

totalTime = 86400

nside = 16
npix = hp.nside2npix(nside)

freqBins = fUpper / deltaF


plt.figure(72)
#Read in LIGO Noise Curve 
Freq_LIGO = np.loadtxt('aLIGO_late_psd.txt', usecols = 0)
PSD_LIGO = np.loadtxt('aLIGO_late_psd.txt', usecols = 1)
#print(len(PSD_LIGO))
plt.plot(Freq_LIGO, np.sqrt(PSD_LIGO) ,'b',label='LIGO_Late Noise Curve')

plt.show()



oneReFreqs , onePSD  = resamplePSDtoOne(Freq_LIGO[601:] , PSD_LIGO[601:])
df = oneReFreqs[1] - oneReFreqs[0]


plt.plot(oneReFreqs ,np.sqrt(2*onePSD),'k')



twoFreqs , twoASD = FFTfromPSD(oneReFreqs , onePSD)

twoPSD = twoASD * twoASD

plt.plot(twoFreqs[0:257],twoASD[0:257],'r')
plt.plot(twoFreqs[258:],twoASD[258:],'r')

plt.yscale('log')
plt.xlim(-512,512)


##A few lines full of mistakes
#myNoiseTimeseries = fftnoise( twoASD )

#plt.figure(42)
#plt.plot(myNoiseTimeseries)





#DONE: make PSDs to have 512 data points between -512 Hz and +512 Hz



timesInSecs = 192*np.arange(450)

mySFTnoiseMatrix1 = np.zeros( (450,512) ,dtype='complex');
mySFTnoiseMatrix2 = np.zeros( (450,512) ,dtype='complex'); 
noiseTSproduct = np.zeros( (450,512) );
for itTime in range(len(timesInSecs)):
#    mySFTnoiseMatrix1[itTime,:] = colorfft(twoASD)
#    mySFTnoiseMatrix2[itTime,:] = colorfft(twoASD)

    mySFTnoiseMatrix1[itTime,:] = frequency_noise_from_psd(onePSD,df)
    mySFTnoiseMatrix2[itTime,:] = frequency_noise_from_psd(onePSD,df)
    
noiseProduct = np.matrix.conjugate(mySFTnoiseMatrix1) * mySFTnoiseMatrix2
print(noiseProduct.shape)
plt.figure(73)
plt.plot(twoFreqs[0:257],noiseProduct[1,0:257].real,'r')
plt.plot(twoFreqs[258:],noiseProduct[1,258:].real,'r')
plt.figure(74)
plt.plot(twoFreqs[0:257],noiseProduct[1,0:257].imag,'b')
plt.plot(twoFreqs[258:],noiseProduct[1,258:].imag,'b')
plt.figure(75)
plt.plot(twoFreqs[0:257],np.sqrt(noiseProduct[1,0:257].real**2+noiseProduct[1,0:257].imag**2),'k')
plt.plot(twoFreqs[258:],np.sqrt(noiseProduct[1,258:].real**2+noiseProduct[1,258:].imag**2),'k' )
plt.figure(76)
plt.plot(np.fft.ifft(noiseProduct[1,:]).real)
plt.figure(77)
plt.plot(np.fft.ifft(noiseProduct[1,:]).imag)

for itTime in range(len(timesInSecs)):
    noiseTSproduct[itTime,:] = np.fft.ifft(noiseProduct[itTime,:]).real

## Averaged noise turns out to be 2 orders of magnitude smaller 
#averagedTSNoise = np.sum(noiseTSproduct,axis=0)/len(timesInSecs)
#    
#plt.figure(23)
#plt.plot(twoFreqs[0:257],averagedTSNoise[0:257],'k')
#plt.plot(twoFreqs[258:],averagedTSNoise[258:],'k' )
    
averagedNoiseProduct = np.sum(np.abs(noiseProduct),axis=0)/len(timesInSecs)
averagedNoise = np.sum(np.abs(mySFTnoiseMatrix1),axis=0)/len(timesInSecs)

plt.figure(72)
plt.plot(twoFreqs[0:257],np.sqrt(2)*averagedNoise[0:257],'grey')
plt.plot(twoFreqs[258:],np.sqrt(2)*averagedNoise[258:],'grey' )

plt.plot(twoFreqs[0:257],np.sqrt(2)*np.sqrt(averagedNoiseProduct[0:257]),'k')
plt.plot(twoFreqs[258:],np.sqrt(2)*np.sqrt(averagedNoiseProduct[258:]),'k' )

    
#Gamma(Omega_k,t)
BeamGamma = np.zeros((npix,450))
with open("GammaMatrix_450times_300decs_450phis.txt",'r') as myFile_G:

    lines = [line.rstrip('\n') for line in myFile_G]
    
    for ii in range(len(lines)):       
        BeamGamma[:,ii] = np.fromstring(lines[ii], dtype=float, sep=' ')
#        print(BeamGamma[ii,:])
        
#print(BeamGamma[:,0])


# P_k assignments

realBackground = np.zeros(npix)

myIndex = 928
IndexW,IndexNW,IndexN = hp.get_all_neighbours(16, myIndex)[2:5]
sourceIndices = [myIndex, IndexW , IndexNW, IndexN]
realBackground[sourceIndices] += 1
#print(realBackground)

#realBackground += np.random.uniform(size=npix)


#DeltaXvec(t,i) where i = x , y , z
alphaRay = np.linspace(0,2*np.pi,450)
DeltaXvec = np.zeros((len(alphaRay),3))

for itTime in np.arange(len(alphaRay)): # len(alphaRay) ~ 1400
    LHO = GW_Detector(46.45528,119.4078,-90+36,alphaRay[itTime],'LIGO Hanford Observatory')
    LLO = GW_Detector(30.56278,90.77417,18, alphaRay[itTime],'LIGO Livingston Observatory')
#   auxDet = GW_Detector(90.0,0.0,0.0, alphaRay[itTime],'Auxiliary Observatory')     
    currentDet1 = LHO
    currentDet2 = LLO
    
    DeltaXvec[itTime,:] = 2998000*(LHO.detectorPos - LLO.detectorPos)/np.linalg.norm(LHO.detectorPos - LLO.detectorPos)

OmegaVec =np.zeros((npix,3))
for k in range(npix):
    OmegaVec[k,0], OmegaVec[k,1], OmegaVec[k,2] = hp.pix2vec(nside,k)[0] , hp.pix2vec(nside,k)[1] , hp.pix2vec(nside,k)[2]
    
#print(OmegaVec[0,:])

#gamma(t,f)
lowercaseGamma = np.zeros( (450, len(twoFreqs)), dtype='complex' )
myNum=0
for itFreq in range(len(twoFreqs)):
    for itTime in np.arange(len(alphaRay)):
        lowercaseGamma[itTime,itFreq] = np.sum(BeamGamma[:,itTime] * realBackground[:] * np.exp(2 * np.pi * 1j * twoFreqs[itFreq] * np.matmul(OmegaVec,DeltaXvec[itTime,:])/ speedOfLight))
        myNum+=1
        if not myNum % 10000 :
            print(myNum)
    
Hspectrum = 1E-50
signalProduct = deltaT * Hspectrum * lowercaseGamma

print(signalProduct.shape)
#print(signalProduct[0,:].max())
   
totalProduct = signalProduct + noiseProduct  

totalProduct = totalProduct  





outsideIntegral = integrate.simps( 2 * Hspectrum**2 /(twoPSD[18:257]*twoPSD[18:257]) , x=twoFreqs[18:257])

## k=496 itTime = 61 seems to be exactly zero, and it causes problems in the inversion
BeamGamma[496, 61] =10E-8
outsideTotal = (deltaT * outsideIntegral * BeamGamma)**(-1)


myNum=0
ultimateIntegral = np.zeros((npix,len(alphaRay)))
for k in range(npix):
    for itTime in range(len(alphaRay)):
        myIntegrand = np.real(totalProduct[itTime,18:257] * Hspectrum * np.exp(- 2 * np.pi * 1j * twoFreqs[18:257] * np.matmul(OmegaVec[k,:],DeltaXvec[itTime,:])/ speedOfLight) / (twoPSD[18:257]*twoPSD[18:257]))
        ultimateIntegral[k,itTime] = 2 * integrate.simps(  myIntegrand , x=twoFreqs[18:257])
        myNum += 1
        if not myNum % 50000 :
            print(myNum)

DeltaS = outsideTotal * ultimateIntegral


numeratorS = np.sum(DeltaS * BeamGamma**2 , axis=1)

denominatorS = np.sum(BeamGamma**2 , axis=1)

S = numeratorS / denominatorS
print(S.shape)

hpxmap = S

mycmap = cm.gnuplot2
mycmap.set_under('w')

#hpxmap_smooth = hp.smoothing(hpxmap, fwhm=np.radians(1.))

hp.mollview(hpxmap,cmap = mycmap)
plt.show()




