#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 13:40:09 2023

@author: naveenj
"""

#libraries
import rampy as rp
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import gcvspline
import lmfit
from lmfit.models import GaussianModel
import random


#data
spectrum = np.genfromtxt("LS4.txt",skip_header=20, skip_footer=43) 

#plot
plt.figure(figsize=(5,5))
plt.plot(spectrum[:,0],spectrum[:,1],'k.',markersize=1)
plt.xlabel("Wn, cm$^{-1}$", fontsize = 12)
plt.ylabel("Normalized intensity, a. u.", fontsize = 12)
plt.title("Fig. 1: The raw data",fontsize = 12,fontweight="bold")
plt.show()

#preprocessing
spectrum_increasing = rp.flipsp(spectrum)
x_new = np.arange(400., 4000., 1.0) #you can change accordingly
y_new = rp.resample(spectrum[:,0], spectrum[:,1], x_new, fill_value="extrapolate")
inputsp = np.vstack((x_new,y_new)).T

bir = np.array([(860,874),(1300,1330)]) # The regions where the baseline will be fitted
y_corr, y_base = rp.baseline(inputsp[:,0],inputsp[:,1],bir,'poly',polynomial_order=3)# We fit a polynomial background.

# spectra selection
lb = 867 # The lower boundary of interest
hb = 1300 # The upper boundary of interest
x = inputsp[:,0]
x_fit = x[np.where((x > lb)&(x < hb))]
y_fit = y_corr[np.where((x > lb)&(x < hb))]
ese0 = np.sqrt(abs(y_fit[:,0]))/abs(y_fit[:,0]) # the relative errors after baseline subtraction
y_fit[:,0] = y_fit[:,0]/np.amax(y_fit[:,0])*10 # normalise spectra to maximum intensity, easier to handle 
sigma = abs(ese0*y_fit[:,0]) #calculate good ese


#a new plot for showing the spectrum
plt.figure()
plt.subplot(1,2,1)
inp = plt.plot(x,inputsp[:,1],'k-',label='Original')
corr = plt.plot(x,y_corr,'b-',label='Corrected') 
bas = plt.plot(x,y_base,'r-',label='Baseline')
plt.xlim(lb,1300)
plt.ylim(0,40000)
plt.xlabel("Wn, cm$^{-1}$", fontsize = 14)
plt.ylabel("Normalized intensity, a. u.", fontsize = 14)
plt.legend()
plt.title('A) Baseline removal')

plt.subplot(1,2,2)
plt.plot(x_fit,y_fit,'k.')
plt.xlabel("Wn, cm$^{-1}$", fontsize = 14)
plt.title('B) signal to fit')
#plt.tight_layout()
plt.suptitle('Figure 2', fontsize = 14,fontweight = 'bold')
plt.show()

def residual(pars, x, data=None, eps=None): #Function definition
    # unpack parameters, extract .value attribute for each parameter
    a1 = pars['a1'].value
    a2 = pars['a2'].value
    a3 = pars['a3'].value
    a4 = pars['a4'].value
    a5 = pars['a5'].value
    
    f1 = pars['f1'].value
    f2 = pars['f2'].value
    f3 = pars['f3'].value
    f4 = pars['f4'].value
    f5 = pars['f5'].value 
    
    l1 = pars['l1'].value
    l2 = pars['l2'].value
    l3 = pars['l3'].value
    l4 = pars['l4'].value
    l5 = pars['l5'].value
    
    # Using the Gaussian model function from rampy
    peak1 = rp.gaussian(x,a1,f1,l1)
    peak2 = rp.gaussian(x,a2,f2,l2)
    peak3 = rp.gaussian(x,a3,f3,l3)
    peak4 = rp.gaussian(x,a4,f4,l4)
    peak5 = rp.gaussian(x,a5,f5,l5)
    
    model = peak1 + peak2 + peak3 + peak4 + peak5 # The global model is the sum of the Gaussian peaks
    
    if data is None: # the function only returns the direct calculation
        return model, peak1, peak2, peak3, peak4, peak5
    if eps is None: # without errors, no ponderation
        return (model - data)
    return (model - data)/eps # with errors, the difference is ponderated


params = lmfit.Parameters()
#               (Name,  Value,  Vary,   Min,  Max,  Expr)
params.add_many(('a1',   2.4,   True,  0,      None,  None),
                ('f1',   946,   True, 910,    970,  None),
                ('l1',   26,   True,  20,      50,  None),
                ('a2',   3.5,   True,  0,      None,  None),
                ('f2',   1026,  True, 990,   1070,  None),
                ('l2',   39,   True,  20,   55,  None),  
                ('a3',   8.5,    True,    7,      None,  None),
                ('f3',   1082,  True, 1070,   1110,  None),
                ('l3',   31,   True,  25,   35,  None),  
                ('a4',   2.2,   True,  0,      None,  None),
                ('f4',   1140,  True, 1110,    1160,  None),
                ('l4',   35,   True,  20,   50,  None),  
                ('a5',   2.,   True,  0,      None,  None),
                ('f5',   1211,  True, 1180,   1220,  None),
                ('l5',   28,   True,  20,   45,  None))

# constrain the positions
params['f1'].vary = False
params['f2'].vary = False
params['f3'].vary = False
params['f4'].vary = False
params['f5'].vary = False

algo = 'nelder'  
    
result = lmfit.minimize(residual, params, method = algo, args=(x_fit, y_fit[:,0]))

params['f1'].vary = True
params['f2'].vary = True
params['f3'].vary = True
params['f4'].vary = True
params['f5'].vary = True

#we fit twice
result2 = lmfit.minimize(residual, params,method = algo, args=(x_fit, y_fit[:,0])) 

model = lmfit.fit_report(result2.params)
yout, peak1,peak2,peak3,peak4,peak5 = residual(result2.params,x_fit) # the different peaks
rchi2 = (1/(float(len(y_fit))-15-1))*np.sum((y_fit - yout)**2/sigma**2) # calculation of the reduced chi-square  

plt.plot(x_fit,y_fit,'k-')
plt.plot(x_fit,yout,'r-')
plt.plot(x_fit,peak1,'b-')
plt.plot(x_fit,peak2,'b-')
plt.plot(x_fit,peak3,'b-')
plt.plot(x_fit,peak4,'b-')
plt.plot(x_fit,peak5,'b-')
    
plt.xlim(lb,hb)
plt.ylim(-0.5,10.5)
plt.xlabel("Wn, cm$^{-1}$", fontsize = 14)
plt.ylabel("Normalized intensity, a. u.", fontsize = 14)
plt.title("Fig. 3: Fit of the x  vibrations\n in B with \nthe Nelder Mead algorithm ",fontsize = 14,fontweight = "bold")
print("rchi-2 = \n"+str(rchi2))
plt.show()

algo = 'leastsq' # We will use the Levenberg-Marquart algorithm  
#               (Name,  Value,  Vary,   Min,  Max,  Expr) Here I directly initialize with fixed frequencies
params.add_many(('a1',   2.4,   True,  0,      None,  None),
                ('f1',   946,   True, 910,    970,  None),
                ('l1',   26,   True,  20,      50,  None),
                ('a2',   3.5,   True,  0,      None,  None),
                ('f2',   1026,  True, 990,   1070,  None),
                ('l2',   39,   True,  20,   55,  None),  
                ('a3',   8.5,    True,    7,      None,  None),
                ('f3',   1082,  True, 1070,   1110,  None),
                ('l3',   31,   True,  25,   35,  None),  
                ('a4',   2.2,   True,  0,      None,  None),
                ('f4',   1140,  True, 1110,    1160,  None),
                ('l4',   35,   True,  20,   50,  None),  
                ('a5',   2.,   True,  0,      None,  None),
                ('f5',   1211,  True, 1180,   1220,  None),
                ('l5',   28,   True,  20,   45,  None))

result = lmfit.minimize(residual, params, method = algo, args=(x_fit, y_fit[:,0]))
# we release the positions but contrain the FWMH and amplitude of all peaks 
params['f1'].vary = True
params['f2'].vary = True
params['f3'].vary = True
params['f4'].vary = True
params['f5'].vary = True

result2 = lmfit.minimize(residual, params,method = algo, args=(x_fit, y_fit[:,0]))
model = lmfit.fit_report(result2.params) # the report 
yout, peak1,peak2,peak3,peak4,peak5 = residual(result2.params,x_fit) # the different peaks
rchi2 = (1/(float(len(y_fit))-15-1))*np.sum((y_fit - yout)**2/sigma**2) # calculation of the reduced chi-square 

##### WE DO A NICE FIGURE THAT CAN BE IMPROVED FOR PUBLICATION
plt.plot(x_fit,y_fit,'k-')
plt.plot(x_fit,yout,'r-')
plt.plot(x_fit,peak1,'b-')
plt.plot(x_fit,peak2,'b-')
plt.plot(x_fit,peak3,'b-')
plt.plot(x_fit,peak4,'b-')
plt.plot(x_fit,peak5,'b-')
    
plt.xlim(lb,hb)
plt.ylim(-0.5,10.5)
plt.xlabel("Wn, cm$^{-1}$", fontsize = 14)
plt.ylabel("Normalized intensity, a. u.", fontsize = 14)
plt.title("Fig. 4: Fit of the x  vibrations\n in B with \nthe LM algorithm",fontsize = 14,fontweight = "bold")
print("rchi-2 = \n"+str(rchi2))
plt.show()


algo = 'cg'  
#               (Name,  Value,  Vary,   Min,  Max,  Expr) Here I directly initialize with fixed frequencies
params.add_many(('a1',   2.4,   True,  0,      None,  None),
                ('f1',   946,   True, 910,    970,  None),
                ('l1',   26,   True,  20,      50,  None),
                ('a2',   3.5,   True,  0,      None,  None),
                ('f2',   1026,  True, 990,   1070,  None),
                ('l2',   39,   True,  20,   55,  None),  
                ('a3',   8.5,    True,    7,      None,  None),
                ('f3',   1082,  True, 1070,   1110,  None),
                ('l3',   31,   True,  25,   35,  None),  
                ('a4',   2.2,   True,  0,      None,  None),
                ('f4',   1140,  True, 1110,    1160,  None),
                ('l4',   35,   True,  20,   50,  None),  
                ('a5',   2.,   True,  0,      None,  None),
                ('f5',   1211,  True, 1180,   1220,  None),
                ('l5',   28,   True,  20,   45,  None))

result = lmfit.minimize(residual, params, method = algo, args=(x_fit, y_fit[:,0]))

params['f1'].vary = True
params['f2'].vary = True
params['f3'].vary = True
params['f4'].vary = True
params['f5'].vary = True

result2 = lmfit.minimize(residual, params,method = algo, args=(x_fit, y_fit[:,0]))
model = lmfit.fit_report(result2.params) # the report 
yout, peak1,peak2,peak3,peak4,peak5 = residual(result2.params,x_fit) # the different peaks
rchi2 = (1/(float(len(y_fit))-15-1))*np.sum((y_fit - yout)**2/sigma**2) # calculation of the reduced chi-square 

plt.plot(x_fit,y_fit,'k-')
plt.plot(x_fit,yout,'r-')
plt.plot(x_fit,peak1,'b-')
plt.plot(x_fit,peak2,'b-')
plt.plot(x_fit,peak3,'b-')
plt.plot(x_fit,peak4,'b-')
plt.plot(x_fit,peak5,'b-')
    
plt.xlim(lb,hb)
plt.ylim(-0.5,10.5)
plt.xlabel("Wn, cm$^{-1}$", fontsize = 14)
plt.ylabel("Normalized intensity, a. u.", fontsize = 14)
plt.title("Fig. 5: Fit of the x stretch vibrations\n in B with \nthe CG algorithm",fontsize = 14,fontweight = "bold")
print("rchi-2 = \n"+str(rchi2))
plt.show()


algo = 'powell' 
#               (Name,  Value,  Vary,   Min,  Max,  Expr) Here I directly initialize with fixed frequencies
params.add_many(('a1',   2.4,   True,  0,      None,  None),
                ('f1',   946,   True, 910,    970,  None),
                ('l1',   26,   True,  20,      50,  None),
                ('a2',   3.5,   True,  0,      None,  None),
                ('f2',   1026,  True, 990,   1070,  None),
                ('l2',   39,   True,  20,   55,  None),  
                ('a3',   8.5,    True,    7,      None,  None),
                ('f3',   1082,  True, 1070,   1110,  None),
                ('l3',   31,   True,  25,   35,  None),  
                ('a4',   2.2,   True,  0,      None,  None),
                ('f4',   1140,  True, 1110,    1160,  None),
                ('l4',   35,   True,  20,   50,  None),  
                ('a5',   2.,   True,  0,      None,  None),
                ('f5',   1211,  True, 1180,   1220,  None),
                ('l5',   28,   True,  20,   45,  None))

result = lmfit.minimize(residual, params, method = algo, args=(x_fit, y_fit[:,0]))

params['f1'].vary = True
params['f2'].vary = True
params['f3'].vary = True
params['f4'].vary = True
params['f5'].vary = True

result2 = lmfit.minimize(residual, params,method = algo, args=(x_fit, y_fit[:,0]))
model = lmfit.fit_report(result2.params) # the report 
yout, peak1,peak2,peak3,peak4,peak5 = residual(result2.params,x_fit) # the different peaks
rchi2 = (1/(float(len(y_fit))-15-1))*np.sum((y_fit - yout)**2/sigma**2) # calculation of the reduced chi-square 

plt.plot(x_fit,y_fit,'k-')
plt.plot(x_fit,yout,'r-')
plt.plot(x_fit,peak1,'b-')
plt.plot(x_fit,peak2,'b-')
plt.plot(x_fit,peak3,'b-')
plt.plot(x_fit,peak4,'b-')
plt.plot(x_fit,peak5,'b-')
    
plt.xlim(lb,hb)
plt.ylim(-0.5,10.5)
plt.xlabel("Wn, cm$^{-1}$", fontsize = 14)
plt.ylabel("Normalized intensity, a. u.", fontsize = 14)
plt.title("Fig. 6: Fit of the x stretch vibrations\n in B with \nthe Powell algorithm",fontsize = 14,fontweight = "bold")
print("rchi-2 = \n"+str(rchi2))
plt.show()

def bootstrap(data, ese,nbsample):
    N = len(data)
    bootsamples = np.zeros((N,nbsample))
    
    for i in range(nbsample):
        for j in range(N):
            bootsamples[j,i] = np.random.normal(data[j], ese[j], size=None)
    return bootsamples

nboot = 10 # Number of bootstrap samples, I set it to a low value for the example but usually you want thousands there
data_resampled = bootstrap(y_fit[:,0],sigma,nboot)# resampling of data + generate the output parameter tensor

para_output = np.zeros((5,3,nboot)) # 5 x 3 parameters x N boot samples
bootrecord = np.zeros((nboot)) # For recording boot strap efficiency
        
for nn in range(nboot):
    algos = ['powell','nelder']
    algo = random.choice(algos) 
    params = lmfit.Parameters()
    #               (Name,  Value,  Vary,   Min,  Max,  Expr) Here I directly initialize with fixed frequencies
    params.add_many(('a1',   24,   True,  0,      None,  None),
                ('f1',   946,   True, 910,    970,  None),
                ('l1',   26,   True,  20,      50,  None),
                ('a2',   35,   True,  0,      None,  None),
                ('f2',   1026,  True, 990,   1070,  None),
                ('l2',   39,   True,  20,   55,  None),  
                ('a3',   85,    True,    70,      None,  None),
                ('f3',   1082,  True, 1070,   1110,  None),
                ('l3',   31,   True,  25,   35,  None),  
                ('a4',   22,   True,  0,      None,  None),
                ('f4',   1140,  True, 1110,    1160,  None),
                ('l4',   35,   True,  20,   50,  None),  
                ('a5',   4,   True,  0,      None,  None),
                ('f5',   1211,  True, 1180,   1220,  None),
                ('l5',   28,   True,  20,   45,  None))

    result = lmfit.minimize(residual, params, method = algo, args=(x_fit, data_resampled[:,nn],sigma))
    params['f1'].vary = True
    params['f2'].vary = True
    params['f3'].vary = True
    params['f4'].vary = True
    params['f5'].vary = True

    result2 = lmfit.minimize(residual, params,method = algo, args=(x_fit, data_resampled[:,nn], sigma))
                           
    vv = result2.params.valuesdict()
    para_output[0,0,nn] = vv['a1']
    para_output[1,0,nn] = vv['a2']
    para_output[2,0,nn] = vv['a3']
    para_output[3,0,nn] = vv['a4']
    para_output[4,0,nn] = vv['a5']
            
    para_output[0,1,nn] = vv['f1']
    para_output[1,1,nn] = vv['f2']
    para_output[2,1,nn] = vv['f3']
    para_output[3,1,nn] = vv['f4']
    para_output[4,1,nn] = vv['f5']
            
    para_output[0,2,nn] = vv['l1']
    para_output[1,2,nn] = vv['l2']
    para_output[2,2,nn] = vv['l3']
    para_output[3,2,nn] = vv['l4']
    para_output[4,2,nn] = vv['l5']
                
para_mean = np.mean(para_output,axis=2)
para_ese = np.std(para_output,axis=2)
for kjy in range(nboot):
    if kjy == 0:
        bootrecord[kjy] = 0
    else:
        bootrecord[kjy] = np.sum(np.std(para_output[:,:,0:kjy],axis=2))

print(para_mean)

print(para_ese)

plt.plot(np.arange(nboot)+1,bootrecord,'ko')
plt.xlim(0,nboot+1)
plt.xlabel("Number of iterations",fontsize = 14)
plt.ylabel("Summed errors of parameters",fontsize = 14)
plt.title("Fig. 7: Bootstrap iterations for convergence",fontsize = 14, fontweight = 'bold')
plt.show()