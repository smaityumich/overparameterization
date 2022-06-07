#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

import matplotlib, re
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = [
    r'\usepackage{amsmath}',
    r'\usepackage{amssymb}',
    r'\usepackage{mathrsfs}']


from matplotlib.ticker import LogLocator, LogFormatterMathtext, ScalarFormatter



from risks import get_trisks, get_erisks




def plot(ax, delta = 1, fontsize = 20, labelsize = 12, titlesize = 25):
    gammas = np.logspace(-4, 4, num = 100, base = 2)
    pis = np.array([0.6, 0.7, 0.9])
    beta = 0
    # delta = 2
    beta_delta = -1
    sigma = 0.1


    trisk_vec = np.zeros(shape = (gammas.shape[0], pis.shape[0]))
    for i, gamma in enumerate(gammas):
        for j, pi in enumerate(pis):
            eta = gamma
            trisk_vec[i, j] = get_trisks(p = pi, gamma = gamma,
                                           eta = eta, sigma = sigma,
                                           beta = beta, delta = delta,
                                           beta_delta=beta_delta, d = 50)
            
            
    trisk_vec_sub = np.zeros(shape = (gammas.shape[0], pis.shape[0]))
    for i, gamma in enumerate(gammas):
        for j, pi in enumerate(pis):
            eta = gamma
            gamma_sub = gamma / (2 - 2 * pi)
            eta_sub = eta * (2 - 2* pi)
            pi_sub = 0.5
            trisk_vec_sub[i, j] = get_trisks(p = pi_sub, gamma = gamma_sub,
                                           eta = eta_sub, sigma = sigma,
                                           beta = beta, delta = delta,
                                           beta_delta=beta_delta, d = 50)
            
            

    ltys = ['-', '--', ':']
    lws = [1, 2, 3]


    a = 0.01


    for j, (pi, lty, lw) in enumerate(zip(pis, ltys, lws)):
        
        ind = gammas < 1 - a
        ax.plot(gammas[ind], trisk_vec[ind, j], linestyle=lty, color = 'r', linewidth=lw, alpha = 1)
        ind = gammas > 1 + a
        ax.plot(gammas[ind], trisk_vec[ind, j], linestyle=lty, color = 'r', linewidth=lw, alpha = 1)
        
        # ax.errorbar(gammas2, erisk_mean[:, j], erisk_std[:, j], linestyle='', color = 'r', linewidth=lw, alpha = 0.5)
         
        
        
        
        ind = gammas < (1 - a) * 2 * (1 - pi)
        ax.plot(gammas[ind], trisk_vec_sub[ind, j], linestyle=lty, color = 'b', linewidth=lw, alpha = 1)
        ind = gammas > (1 + a) * 2 * (1 - pi)
        ax.plot(gammas[ind], trisk_vec_sub[ind, j], linestyle=lty, color = 'b', linewidth=lw, alpha = 1)
        
        # ax.errorbar(gammas3, erisk_sub_mean[:, j], erisk_sub_std[:, j],
        #             linestyle='', color = 'b', linewidth=lw, alpha = 0.5)
        
    ax.set_xscale('log')
    ax.set_ylim(0, 4)
    ticks = np.logspace(-4, 4, num = 9, base = 2)
    ax.set_xticks(ticks)
    lf = LogFormatterMathtext(base = 2)
    lf.locs = list(gammas)
    ax.xaxis.set_major_formatter(lf)
    ax.tick_params(axis = 'both', labelsize=labelsize)
    ax.set_xlabel(r'$\gamma=N/n$', fontsize = fontsize)
    ax.set_title(r'$F_\delta=' + str(delta) + r'$', fontsize = titlesize)
    
    
def get_legend(ax, fontsize = 20):
    labels = ['ERM', 'subsample', r'$\pi = 0.6$', r'$\pi = 0.7$', r'$\pi = 0.9$']
    lines = [
        Line2D([0], [0], color = 'r', linestyle='-'), 
        Line2D([0], [0], color = 'b', linestyle='-')
        ]
    
    for lty, lw in zip(ltys, lws):
        lines.append(Line2D([0], [0], color = 'k', linestyle=lty, linewidth=lw))
    ax.legend(lines, labels, fontsize = fontsize)
    

fig, axs = plt.subplots(1, 2, figsize = (12, 5))
deltas = [0, 2]
for ax, delta in zip(axs, deltas):
    plot(ax, delta=delta, labelsize=15, fontsize=20, titlesize=25)

get_legend(axs[0], fontsize = 17)
fig.savefig('ERMvSS.pdf')













gammas = np.logspace(-4, 4, num = 100, base = 2)
pis = np.array([0.6, 0.7, 0.9])
beta = 1
delta = 2
beta_delta = -1
sigma = 0.1


trisk_vec = np.zeros(shape = (gammas.shape[0], pis.shape[0]))
for i, gamma in enumerate(gammas):
    for j, pi in enumerate(pis):
        eta = gamma
        trisk_vec[i, j] = get_trisks(p = pi, gamma = gamma,
                                       eta = eta, sigma = sigma,
                                       beta = beta, delta = delta,
                                       beta_delta=beta_delta, d = 50)
        
        
length = 5
gammas2 = np.concatenate([np.logspace(-3.5, -0.5, num = length, base = 2), np.logspace(0.5, 3.5, num = length, base = 2)]) # N/n
ITER = 20
erisk = np.zeros(shape = (gammas2.shape[0], pis.shape[0], ITER))

for i, gamma in enumerate(gammas2):
    for j, pi in enumerate(pis):
        for k in range(ITER):
            eta = gamma
            erisk[i, j, k] = get_erisks(p = pi, gamma = gamma,
                                            eta = eta, sigma = sigma,
                                            beta = beta, delta = delta,
                                            beta_delta=beta_delta, d = 50)
            
erisk_mean = erisk.mean(axis = 2)     
erisk_std = erisk.std(axis = 2)        

        

trisk_vec_sub = np.zeros(shape = (gammas.shape[0], pis.shape[0]))
for i, gamma in enumerate(gammas):
    for j, pi in enumerate(pis):
        eta = gamma
        gamma_sub = gamma / (2 - 2 * pi)
        eta_sub = eta * (2 - 2* pi)
        pi_sub = 0.5
        trisk_vec_sub[i, j] = get_trisks(p = pi_sub, gamma = gamma_sub,
                                       eta = eta_sub, sigma = sigma,
                                       beta = beta, delta = delta,
                                       beta_delta=beta_delta, d = 50)
        
        
gammas3 = np.logspace(-3.5, 3.5, num = 2 * length, base = 2)        
erisk_sub = np.zeros(shape = (gammas3.shape[0], pis.shape[0], ITER))



for i, gamma in enumerate(gammas3):
    for j, pi in enumerate(pis):
        for k in range(ITER):
            eta = gamma
            gamma_sub = gamma / (2 - 2 * pi)
            eta_sub = eta * (2 - 2* pi)
            pi_sub = 0.5
            if np.absolute(gamma_sub - 1) > 0.5:
                erisk_sub[i, j, k] = get_erisks(p = pi_sub, gamma = gamma_sub,
                                                eta = eta_sub, sigma = sigma,
                                                beta = beta, delta = delta,
                                                beta_delta=beta_delta, d = 100)
            else:
                erisk_sub[i, j, k] = np.NaN
            
erisk_sub_mean = erisk_sub.mean(axis = 2)     
erisk_sub_std = erisk_sub.std(axis = 2)        

ltys = ['-', '--', ':']
lws = [1, 2, 3]
cols = ['r', 'b']  


fig, axs = plt.subplots(1, 1, figsize = (12, 5))
ax = axs

delta = 0.01


for j, (pi, lty, lw) in enumerate(zip(pis, ltys, lws)):
    
    ind = gammas < 1 - delta
    ax.plot(gammas[ind], trisk_vec[ind, j], linestyle=lty, color = 'r', linewidth=lw, alpha = 1)
    ind = gammas > 1 + delta
    ax.plot(gammas[ind], trisk_vec[ind, j], linestyle=lty, color = 'r', linewidth=lw, alpha = 1)
    
    ax.errorbar(gammas2, erisk_mean[:, j], erisk_std[:, j], linestyle='',
                color = 'r', linewidth=lw, alpha = 1)
     
    
    
    
    ind = gammas < (1 - delta) * 2 * (1 - pi)
    ax.plot(gammas[ind], trisk_vec_sub[ind, j], linestyle=lty, color = 'b', linewidth=lw, alpha = 1)
    ind = gammas > (1 + delta) * 2 * (1 - pi)
    ax.plot(gammas[ind], trisk_vec_sub[ind, j], linestyle=lty, color = 'b', linewidth=lw, alpha = 1)
    
    ax.errorbar(gammas3, erisk_sub_mean[:, j], erisk_sub_std[:, j],
                linestyle='', color = 'b', linewidth=lw, alpha = 1)
    
ax.set_xscale('log')
ax.set_ylim(0, 8)
ticks = np.logspace(-4, 4, num = 9, base = 2)
ax.set_xticks(ticks)
lf = LogFormatterMathtext(base = 2)
lf.locs = list(gammas)
ax.xaxis.set_major_formatter(lf)
ax.tick_params(axis = 'both', labelsize=22)
ax.set_xlabel(r'$\gamma=N/n$', fontsize = 35)
# ind = gammas2 < 1-delta
# ax.errorbar(gammas2[ind], brisks_mean[ind], brisks_std[ind], linestyle='', color = col, marker = mk)
# ind = gammas2 > 1+ delta
# ax.errorbar(gammas2[ind], brisks_mean[ind], brisks_std[ind], linestyle='', color = col, marker = mk)
      
      
