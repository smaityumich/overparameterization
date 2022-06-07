#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%


import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

# import matplotlib, re
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.preamble'] = [
#     r'\usepackage{amsmath}',
#     r'\usepackage{amssymb}',
#     r'\usepackage{mathrsfs}']


# from matplotlib.ticker import LogLocator, LogFormatterMathtext, ScalarFormatter




from risks import get_trisks, get_erisks

#%%

'''
plot of sub-sampling error term for fixed pi and varying angles
'''

d = 200
n = 3 * d
p = 0.8
beta0 = 1 # norm of minority coeff, SNR if sigma = 1
beta1 = 1
thetas = np.arange(0, np.pi + np.pi/4, step = np.pi/4)
#delta = np.sqrt( 2 * (1 - np.cos(theta))) # norm of shift, SNR-shift if sigma = 1
sigma = 0 # error std, keep it 1 (or 0 for noiseless case) 
length = 5
length_th = 100
ITER = 10
gammas = np.concatenate([np.logspace(-1, -0.1, num = length), np.logspace(0.1, 1, num = length)]) # N/n
gammas_th = np.concatenate([np.logspace(-1, -0.01, num = length_th), np.logspace(0.01, 1, num = length_th)]) # N/nITER = 10
risks = np.zeros(shape = (gammas.shape[0], thetas.shape[0], ITER))
trisks = np.zeros(shape = (gammas_th.shape[0], thetas.shape[0]))

#risks = np.zeros(shape = (gammas.shape[0], thetas.shape[0], ITER))

for i, gamma in enumerate(gammas):
    for j, theta in enumerate(thetas):
        for k in range(ITER):
            eta = 3 * gamma # eta = N/d = gamma * (n/d)
            gamma_sub = gamma / (2 - 2 * p)
            eta_sub = eta * (2 - 2 * p)
            erisk = get_erisks(p = 0.5, gamma = gamma_sub, eta = eta_sub, sigma = sigma, beta = beta0, delta = np.sqrt((1 - np.cos(theta))*2), d = d, beta_delta = -np.sqrt((1 - np.cos(theta))/2)) - get_erisks(p = p, gamma = gamma, eta = eta, sigma = sigma, beta = beta0, delta = np.sqrt((1 - np.cos(theta))*2), d = d, beta_delta = -np.sqrt((1 - np.cos(theta))/2))
            #trisk = get_trisks(p = 0.5, gamma = gamma_sub, eta = eta_sub, sigma = sigma, beta = beta0, delta = np.sqrt((1 - np.cos(theta))*2), d = d, beta_delta = -np.sqrt((1 - np.cos(theta))/2)) - get_trisks(p = p, gamma = gamma, eta = eta, sigma = sigma, beta = beta0, delta = np.sqrt((1 - np.cos(theta))*2), d = d, beta_delta = -np.sqrt((1 - np.cos(theta))/2))
            risks[i, j, k] = erisk
#%%            
mean_risks = risks.mean(axis = 2)
std_risks = risks.std(axis=2)

for i, gamma in enumerate(gammas_th):
    for j, theta in enumerate(thetas):
        
        eta = 3 * gamma # eta = N/d = gamma * (n/d)
        gamma_sub = gamma / (2 - 2 * p)
        eta_sub = eta * (2 - 2 * p)
        #erisk = get_erisks(p = 0.5, gamma = gamma_sub, eta = eta_sub, sigma = sigma, beta = beta0, delta = np.sqrt((1 - np.cos(theta))*2), d = d, beta_delta = -np.sqrt((1 - np.cos(theta))/2)) - get_erisks(p = p, gamma = gamma, eta = eta, sigma = sigma, beta = beta0, delta = np.sqrt((1 - np.cos(theta))*2), d = d, beta_delta = -np.sqrt((1 - np.cos(theta))/2))
        trisk = get_trisks(p = 0.5, gamma = gamma_sub, eta = eta_sub, sigma = sigma, beta = beta0, delta = np.sqrt((1 - np.cos(theta))*2), d = d, beta_delta = -np.sqrt((1 - np.cos(theta))/2)) - get_trisks(p = p, gamma = gamma, eta = eta, sigma = sigma, beta = beta0, delta = np.sqrt((1 - np.cos(theta))*2), d = d, beta_delta = -np.sqrt((1 - np.cos(theta))/2))
        trisks[i, j] = trisk
#%%
labels = ['Emp', 'Th']
cols = ['green', 'red', 'blue', 'orange','purple']
ltys = ['-', ':']
markers = ['x', 'o', "^", '+', "*" ]
lines = []


plt.figure()
for j,_ in enumerate(thetas):
    
    
    plt.plot(gammas_th[length_th:], trisks[length_th:, j], linestyle = '-', alpha = 0.5, color = cols[j], label = "Th")
    plt.plot(gammas_th[:length_th], trisks[:length_th, j], linestyle = '-', alpha = 0.5, color = cols[j])
    
    # plt.scatter(gammas[length:], mean_risks[length:,j], color = cols[j], marker = "o", label = "Emp")
    # plt.scatter(gammas[:length], mean_risks[:length, j], color = cols[j], marker = "o")
    
    plt.errorbar(gammas[:length], mean_risks[:length,j], std_risks[:length, j], linestyle = '', color = cols[j], marker = 'o')
    plt.errorbar(gammas[length:], mean_risks[length:, j], std_risks[length:, j], linestyle = '', color = cols[j], marker = 'o')
    #lines.append(Line2D([0], [0], linestyle = lty, color = col, marker = marker))
    
plt.xscale('log')
plt.xlabel(r"$\gamma$", fontsize = 15)
plt.ylabel("Sub sampling error - ERM error", fontsize = 12)
plt.ylim(-5, 7)
plt.axhline(y=0, color='black', linestyle='--')
labels, lines = [], []
for i, col in enumerate(cols):
    lines.append(Line2D([0], [0], marker = 'o', color = col))
    t = thetas[i]
    labels.append(r'$\theta =$'+str(i/4)+r'$\times\pi$')

plt.legend(lines, labels,fontsize = 12, loc = 'best')
#plt.show()
name = 'pi=' + str(p) + "_varying_angles" 
plt.savefig("figs/ss_risk_fixed_pi/" + name + ".pdf")   
plt.show()     
#%%
    
    
'''
plot of sub-sampling error term for fixed angle and varying pi
'''

d = 200
n = 3 * d
p = np.array([.6,.7,.8, .9])
beta0 = 1 # norm of minority coeff, SNR if sigma = 1
beta1 = 1
theta = np.pi
#delta = np.sqrt( 2 * (1 - np.cos(theta))) # norm of shift, SNR-shift if sigma = 1
sigma = 0 # error std, keep it 1 (or 0 for noiseless case) 
length = 5
length_th = 100
ITER = 10
gammas = np.concatenate([np.logspace(-1, -0.1, num = length), np.logspace(0.1, 1, num = length)]) # N/n
gammas_th = np.concatenate([np.logspace(-1, -0.01, num = length_th), np.logspace(0.01, 1, num = length_th)]) # N/nITER = 10
risks = np.zeros(shape = (gammas.shape[0], p.shape[0], ITER))
trisks = np.zeros(shape = (gammas_th.shape[0], p.shape[0]))



for i, gamma in enumerate(gammas):
    for j, pi in enumerate(p):
        for k in range(ITER):
            eta = 3 * gamma # eta = N/d = gamma * (n/d)
            gamma_sub = gamma / (2 - 2 * pi)
            eta_sub = eta * (2 - 2 * pi)
            erisk = get_erisks(p = .5, gamma = gamma_sub, eta = eta_sub, sigma = sigma, beta = beta0, delta = np.sqrt((1 - np.cos(theta))*2), d = d, beta_delta = -np.sqrt((1 - np.cos(theta))/2))\
                - get_erisks(p = pi, gamma = gamma, eta = eta, sigma = sigma, beta = beta0, delta = np.sqrt((1 - np.cos(theta))*2), d = d, beta_delta = -np.sqrt((1 - np.cos(theta))/2))
            #trisk = get_trisks(p = .5, gamma = gamma_sub, eta = eta_sub, sigma = sigma, beta = beta0, delta = np.sqrt((1 - np.cos(theta))*2), d = d, beta_delta = -np.sqrt((1 - np.cos(theta))/2))- get_trisks(p = pi, gamma = gamma, eta = eta, sigma = sigma, beta = beta0, delta = np.sqrt((1 - np.cos(theta))*2), d = d, beta_delta = -np.sqrt((1 - np.cos(theta))/2))
            risks[i, j, k] = erisk
#%%            
mean_risks = risks.mean(axis = 2)
std_risks = risks.std(axis=2)

for i, gamma in enumerate(gammas_th):
    for j, pi in enumerate(p):
        
        eta = 3 * gamma # eta = N/d = gamma * (n/d)
        gamma_sub = gamma / (2 - 2 * pi)
        eta_sub = eta * (2 - 2 * pi)
        #erisk = get_erisks(p = .5, gamma = gamma_sub, eta = eta_sub, sigma = sigma, beta = beta0, delta = np.sqrt((1 - np.cos(theta))*2), d = d, beta_delta = -np.sqrt((1 - np.cos(theta))/2))- get_erisks(p = pi, gamma = gamma, eta = eta, sigma = sigma, beta = beta0, delta = np.sqrt((1 - np.cos(theta))*2), d = d, beta_delta = -np.sqrt((1 - np.cos(theta))/2))
        trisk = get_trisks(p = .5, gamma = gamma_sub, eta = eta_sub, sigma = sigma, beta = beta0, delta = np.sqrt((1 - np.cos(theta))*2), d = d, beta_delta = -np.sqrt((1 - np.cos(theta))/2))\
            - get_trisks(p = pi, gamma = gamma, eta = eta, sigma = sigma, beta = beta0, delta = np.sqrt((1 - np.cos(theta))*2), d = d, beta_delta = -np.sqrt((1 - np.cos(theta))/2))
        trisks[i, j] =  trisk
#%%
labels = ['Emp', 'Th']
cols = ['green', 'red', 'blue', 'orange']
ltys = ['-', ':']
markers = ['x', 'o', "^", '+', "*" ]
lines = []

for j,_ in enumerate(p):
    
    plt.plot(gammas_th[length_th:], trisks[length_th:, j], linestyle = '-', alpha = 0.5, color = cols[j], label = "Th")
    plt.plot(gammas_th[:length_th], trisks[:length_th, j], linestyle = '-', alpha = 0.5, color = cols[j])
    
    # plt.scatter(gammas[length:], mean_risks[length:,j], color = cols[j], marker = "o", label = "Emp")
    # plt.scatter(gammas[:length], mean_risks[:length, j], color = cols[j], marker = "o")
    
    plt.errorbar(gammas[:length], mean_risks[:length,j], std_risks[:length, j], linestyle = '', color = cols[j], marker = 'o')
    plt.errorbar(gammas[length:], mean_risks[length:, j], std_risks[length:, j], linestyle = '', color = cols[j], marker = 'o')
    #lines.append(Line2D([0], [0], linestyle = lty, color = col, marker = marker))
    
plt.xscale('log')
plt.xlabel(r"$\gamma$", fontsize = 15)
plt.ylabel("Sub sampling error - ERM error", fontsize = 12)
plt.ylim(-5, 7)
plt.axhline(y=0, color='black', linestyle='--')
labels, lines = [], []
for i, col in enumerate(cols):
    lines.append(Line2D([0], [0], marker = 'o', color = col))
    t = thetas[i]
    labels.append(r'$\pi =$'+str(p[i]))

plt.legend(lines, labels,fontsize = 12, loc = 'best')
#plt.show()
name = 'theta=pi'  + "_varying_prop" 
plt.savefig("figs/ss_risk_fixed_angle/" + name + ".pdf")   
plt.show()  
