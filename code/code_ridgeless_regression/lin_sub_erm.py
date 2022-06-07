#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from linrisk import *
 
'''
Subsampling risk - ERM risk plot
'''
#%%

'''
Plot for pi = 0.8 and varying angles
norm of beta0 and beta1 is equal to 1
theta = angle between beta0 and beta1
delta = sqrt(2*(1 - cos(theta))) by geometry
beta0.delta = - sqrt(1-cos(theta))
'''
d = 200
 
sigma = np.sqrt(0.1)
pi = 0.8
beta0 = 1 # norm of minority coeff
beta1 = 1 # norm of majority coeff
 
thetas = np.arange(0, np.pi + np.pi/4, step = np.pi/4)
length = 200
gammas = np.concatenate([np.logspace(-1, -0.0001, num = length), np.logspace(0.0001, 1, num = length)]) # N/n
 
trisks = np.zeros(shape = (gammas.shape[0], thetas.shape[0]))
 
for i, gamma in enumerate(gammas):
    for j,theta in enumerate(thetas):
        delta = np.sqrt((1 - np.cos(theta))*2)
        beta_delta = -np.sqrt((1 - np.cos(theta)))
        trisk = -t_linrisk(beta0 = beta0, delta = delta, beta_delta = beta_delta , sigma = sigma, gamma = gamma, pi = pi)\
                + t_linrisk_sub(beta0 = beta0, delta = delta, beta_delta = beta_delta , sigma = sigma, gamma = gamma, pi = pi)
        trisks[i,j] = trisk    

labels = ['Emp', 'Th']
cols = ['green', 'red', 'blue', 'orange', 'purple']
ltys = ['-', ':']
markers = ['x', 'o', "^", '+', "*" ]
lines = []

plt.figure()
for j,_ in enumerate(thetas):
    
    
    plt.plot(gammas[length:], trisks[length:, j], linestyle = '-', alpha = 0.5, color = cols[j], label = "Th")
    plt.plot(gammas[:length], trisks[:length, j], linestyle = '-', alpha = 0.5, color = cols[j])
    
   
plt.xscale('log')
plt.xlabel(r"$\gamma$", fontsize = 25)
plt.ylabel("Subsampling - ERM", fontsize = 23)
plt.ylim(-5, 7)
plt.axhline(y=0, color='black', linestyle='--')
labels, lines = [], []
labels1 = [r'$0^\circ$', r'$45^\circ$',r'$90^\circ$',r'$135^\circ$', r'$180^\circ$']
for i, col in enumerate(cols):
    lines.append(Line2D([0], [0], color = col))
    t = thetas[i]
    labels.append(r'$\theta =$'+labels1[i])

plt.legend(lines, labels,fontsize = 15, loc = 'best')
#name = 'sub_pi=' + str(pi) + "_varying_angles" 
#plt.savefig("lin-figs/" + name + ".pdf", bbox_inches = 'tight')   
plt.show() 

#%%

'''
Plot for theta = 180 and varying proportion
norm of beta0 and beta1 is equal to 1
theta = angle between beta0 and beta1
delta = sqrt(2*(1 - cos(theta))) by geometry
beta0.delta = - sqrt(1-cos(theta))
'''
d = 200
 
pi_list = np.array([.6,.7,.8,.9])
beta0 = 1 # norm of minority coeff
beta1 = 1 # norm of majority coeff

theta = np.pi
length = 200
gammas = np.concatenate([np.logspace(-1, -0.0001, num = length), np.logspace(0.0001, 1, num = length)]) # N/n

trisks = np.zeros(shape = (gammas.shape[0], pi_list.shape[0]))

for i, gamma in enumerate(gammas):
    for j,pi in enumerate(pi_list):
        delta = np.sqrt((1 - np.cos(theta))*2)
        beta_delta = -np.sqrt((1 - np.cos(theta)))
        trisk = -t_linrisk(beta0 = beta0, delta = delta, beta_delta = beta_delta , sigma = 1, gamma = gamma, pi = pi)\
                + t_linrisk_sub(beta0 = beta0, delta = delta, beta_delta = beta_delta , sigma = 1, gamma = gamma, pi = pi) 
        trisks[i,j] = trisk    

labels = ['Emp', 'Th']
cols = ['green', 'red', 'blue', 'orange']
ltys = ['-', ':']
markers = ['x', 'o', "^", '+', "*" ]
lines = []

plt.figure()
for j,_ in enumerate(pi_list):
    
    
    plt.plot(gammas[length:], trisks[length:, j], linestyle = '-', alpha = 0.5, color = cols[j], label = "Th")
    plt.plot(gammas[:length], trisks[:length, j], linestyle = '-', alpha = 0.5, color = cols[j])
    
   
plt.xscale('log')
plt.xlabel(r"$\gamma$", fontsize = 25)
plt.ylabel("Subsampling - ERM", fontsize = 23)
plt.ylim(-5, 7)
plt.axhline(y=0, color='black', linestyle='--')

labels, lines = [], []
labels1 = [r'$0.6$', r'$0.7$',r'$0.8$',r'$0.9$']
for i, col in enumerate(cols):
    lines.append(Line2D([0], [0], color = col))
    t = thetas[i]
    labels.append(r'$\pi =$'+labels1[i])

plt.legend(lines, labels,fontsize = 15, loc = 'upper right')
#name = 'sub_theta=' + str(180) + "_varying_pi" 
#plt.savefig("lin-figs/" + name + ".pdf", bbox_inches = 'tight')   
plt.show() 
