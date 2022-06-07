#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = [
    r'\usepackage{amsmath}',
    r'\usepackage{amssymb}',
    r'\usepackage{mathrsfs}']


from matplotlib.ticker import LogFormatterMathtext, ScalarFormatter



from risks import *


# ## Overparametrization plot

# In[2]:


d = 200
n = 3 * d
p = 0.9
beta = 5 # norm of minority coeff, SNR if sigma = 1
delta = 1 # norm of shift, SNR-shift if sigma = 1
sigma = 1 # error std, keep it 1 (or 0 for noiseless case) 
length = 5
gammas = np.concatenate([np.logspace(-1, -0.1, num = length), np.logspace(0.1, 1, num = length)]) # N/n
ITER = 10
risks = np.zeros(shape = (gammas.shape[0], 2, ITER))

for i, gamma in enumerate(gammas):
    for j in range(ITER):
        eta = 3 * gamma # eta = N/d = gamma * (n/d)
        erisk, trisk = get_risks(p = p, gamma=gamma, eta = eta, sigma = sigma, beta = beta, delta = delta, d = d)
        risks[i, :, j] = erisk, trisk


# In[3]:


mean_risks = risks.mean(axis = 2)
std_risks = risks.std(axis=2)
labels = ['Emp', 'Th']
cols = ['orange', 'green']
ltys = ['-', ':']
markers = ['x', 'o']
lines = []

for i, (label, col, lty, marker) in enumerate(zip(labels, cols, ltys, markers)):
    plt.errorbar(gammas[:length], mean_risks[:length, i], std_risks[:length, i], linestyle = lty, color = col, marker = marker)
    plt.errorbar(gammas[length:], mean_risks[length:, i], std_risks[length:, i], linestyle = lty, color = col, marker = marker)
    lines.append(Line2D([0], [0], linestyle = lty, color = col, marker = marker))

plt.xscale('log')
plt.xlabel(r'$\gamma$', fontsize = 15)
plt.legend(lines, labels, fontsize = 15)


# ## Different $\pi$

# In[4]:


d = 200
n = 2 * d
ps = np.linspace(0.5, 0.9, num = 2 * length)
beta = 5 # norm of minority coeff, SNR if sigma = 1
delta = 1 # norm of shift, SNR-shift if sigma = 1
sigma = 1 # error std, keep it 1 (or 0 for noiseless case) 
length = 5
gammas = 3 # N/n
ITER = 10
risks = np.zeros(shape = (ps.shape[0], 2, ITER))

for i, p in enumerate(ps):
    for j in range(ITER):
        eta = (n/d) * gamma # eta = N/d = gamma * (n/d)
        erisk, trisk = get_risks(p = p, gamma=gamma, eta = eta, sigma = sigma, beta = beta, delta = delta, d = d)
        risks[i, :, j] = erisk, trisk


# In[5]:


mean_risks = risks.mean(axis = 2)
std_risks = risks.std(axis=2)
labels = ['Emp', 'Th']
cols = ['orange', 'green']
ltys = ['-', ':']
markers = ['x', 'o']
lines = []

for i, (label, col, lty, marker) in enumerate(zip(labels, cols, ltys, markers)):
    plt.errorbar(ps, mean_risks[:, i], std_risks[:, i], linestyle = lty, color = col, marker = marker)
    lines.append(Line2D([0], [0], linestyle = lty, color = col, marker = marker))


plt.xlabel(r'$\pi$', fontsize = 15)
plt.legend(lines, labels, fontsize = 15)


# In[ ]:

d = 200
n = 2 * d
ps = 0.5
beta = 1 # norm of minority coeff, SNR if sigma = 1
delta = 0 # norm of shift, SNR-shift if sigma = 1
sigma = 0 # error std, keep it 1 (or 0 for noiseless case) 
length = 5
gammas = np.logspace(-4, 4, num = 100, base = 2)
trisk_vec_bias = np.zeros_like(gammas)
for i, gamma in enumerate(gammas):
    trisk_vec_bias[i] = get_trisks(p = 0.9, gamma = gamma, eta = gamma, sigma = 0, beta = 1, delta = 0, d = 50)
    
length = 5
gammas2 = np.concatenate([np.logspace(-1, -0.1, num = length), np.logspace(0.1, 1, num = length)]) # N/n
ITER = 20
risks_bias = np.zeros(shape = (gammas2.shape[0], ITER))

for i, gamma in enumerate(gammas2):
    for j in range(ITER):
        erisk = get_erisks(p = 0.9, gamma = gamma, eta = gamma, sigma = 0, beta = 1, delta = 0, d = 50)
        risks_bias[i, j] = erisk
brisks_mean = risks_bias.mean(axis = 1)     
brisks_std = risks_bias.std(axis = 1)        



d = 200
n = 2 * d
ps = 0.5
beta = 1 # norm of minority coeff, SNR if sigma = 1
delta = 0 # norm of shift, SNR-shift if sigma = 1
sigma = 0 # error std, keep it 1 (or 0 for noiseless case) 
length = 5
gammas = np.logspace(-4, 4, num = 100, base = 2)
trisk_vec_var = np.zeros_like(gammas)
for i, gamma in enumerate(gammas):
    trisk_vec_var[i] = get_trisks(p = 0.9, gamma = gamma, eta = gamma, sigma = 1, beta = 0, delta = 0, d = 50)
    
length = 5
gammas2 = np.concatenate([np.logspace(-1, -0.1, num = length), np.logspace(0.1, 1, num = length)]) # N/n
ITER = 20
risks_var = np.zeros(shape = (gammas2.shape[0], ITER))

for i, gamma in enumerate(gammas2):
    for j in range(ITER):
        erisk = get_erisks(p = 0.9, gamma = gamma, eta = gamma, sigma = 1, beta = 0, delta = 0, d = 50)
        risks_var[i, j] = erisk
vrisks_mean = risks_var.mean(axis = 1)     
vrisks_std = risks_var.std(axis = 1)        



d = 200
n = 2 * d
ps = 0.5
beta = 1 # norm of minority coeff, SNR if sigma = 1
delta = 0 # norm of shift, SNR-shift if sigma = 1
sigma = 0 # error std, keep it 1 (or 0 for noiseless case) 
length = 5
gammas = np.logspace(-4, 4, num = 100, base = 2)
trisk_vec_m = np.zeros_like(gammas)
for i, gamma in enumerate(gammas):
    trisk_vec_m[i] = get_trisks(p = 0.9, gamma = gamma, eta = gamma, sigma = 0, beta = 0, delta = 1, d = 50)
    
length = 5
gammas2 = np.concatenate([np.logspace(-1, -0.1, num = length), np.logspace(0.1, 1, num = length)]) # N/n
ITER = 20
risks_m = np.zeros(shape = (gammas2.shape[0], ITER))

for i, gamma in enumerate(gammas2):
    for j in range(ITER):
        erisk = get_erisks(p = 0.9, gamma = gamma, eta = gamma, sigma = 0, beta = 0, delta = 1, d = 50)
        risks_m[i, j] = erisk
mrisks_mean = risks_m.mean(axis = 1)     
mrisks_std = risks_m.std(axis = 1)      


fig, axs = plt.subplots(1, 3, figsize = (35, 7.5))

ax = axs[0]
lty, col, mk = '-', 'blue', 'o'
delta = 0.01
ind = gammas < 1-delta
ax.plot(gammas[ind], trisk_vec_bias[ind], linestyle=lty, color = col, linewidth=1, alpha = 1)
ind = gammas > 1+ delta
ax.plot(gammas[ind], trisk_vec_bias[ind], linestyle=lty, color = col, linewidth=1, alpha = 1)

ind = gammas2 < 1-delta
ax.errorbar(gammas2[ind], brisks_mean[ind], brisks_std[ind], linestyle='', color = col, marker = mk)
ind = gammas2 > 1+ delta
ax.errorbar(gammas2[ind], brisks_mean[ind], brisks_std[ind], linestyle='', color = col, marker = mk)

lty, col, mk = '-', 'green', 'x'
ind = gammas < 1-delta
ax.plot(gammas[ind], trisk_vec_var[ind], linestyle=lty, color = col, linewidth=1, alpha = 1)
ind = gammas > 1+ delta
ax.plot(gammas[ind], trisk_vec_var[ind], linestyle=lty, color = col, linewidth=1, alpha = 1)

ax.set_xscale('log')
# lf = LogFormatter(base = 2)
# lf.locs = list(gammas)
ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))

ind = gammas2 < 1-delta
ax.errorbar(gammas2[ind], vrisks_mean[ind], vrisks_std[ind], linestyle='', color = col, marker = mk)

ind = gammas2 > 1+ delta
ax.errorbar(gammas2[ind], vrisks_mean[ind], vrisks_std[ind], linestyle='', color = col, marker = mk)

ticks = np.logspace(-4, 4, num = 9, base = 2)
ax.set_xticks(ticks)
lf = LogFormatterMathtext(base = 2)
lf.locs = list(gammas)
ax.xaxis.set_major_formatter(lf)
ax.tick_params(axis = 'both', labelsize=22)
ax.set_ylim((0, 8))

# lty, col, mk = '-', 'red', '+'
# ind = gammas < 1-delta
# ax.plot(gammas[ind], trisk_vec_m[ind], linestyle=lty, color = col, linewidth=1, alpha = 1)
# ind = gammas > 1+ delta
# plt.plot(gammas[ind], trisk_vec_m[ind], linestyle=lty, color = col, linewidth=1, alpha = 1)

# ind = gammas2 < 1-delta
# plt.errorbar(gammas2[ind], mrisks_mean[ind], mrisks_std[ind], linestyle='', color = col, marker = mk)
# ind = gammas2 > 1+ delta
# plt.errorbar(gammas2[ind], mrisks_mean[ind], mrisks_std[ind], linestyle='', color = col, marker = mk)

ax.set_xlabel(r'$\gamma=N/n$', fontsize = 35)

lines = [
    Line2D([0], [0], linestyle='-', color = 'blue', marker = 'o'),
    Line2D([0], [0], linestyle='-', color = 'g', marker = 'x'),
    ]

labels = [
    r'$\mathcal{B}^\star$', r'$\mathcal{V}^\star$',
    ]
ax.legend(lines, labels, fontsize = 35)
# fig.savefig('bias-var.pdf')


pi = np.linspace(0.5, 0.99, num = 100)
length = 50
gammas = np.concatenate([np.logspace(-3, -0.1, num = length, base = 2), np.logspace(0.1, 3, num = length, base = 2)])
PI, GAMMA = np.meshgrid(pi, gammas)

M = np.zeros_like(PI)
for i in range(PI.shape[0]):
    for j in range(PI.shape[1]):
        M[i, j] = get_trisks(p = PI[i, j], gamma = GAMMA[i, j], eta = GAMMA[i, j], sigma = 0, beta = 0, delta = 1, d = 50)

# fig, ax = plt.subplots(1, 1, figsize = (12, 7.5))   
ax = axs[1]     
# contours = plt.contour(GAMMA, PI, M, 3, colors='black')
cntr = ax.contourf(GAMMA, PI, np.log(M), 20, cmap='Spectral', alpha = 0.5, vmax = 2.1, vmin = -3.1)
# plt.clabel(contours, inline=True, fontsize=8)
ax.plot(np.ones_like(pi), pi, color = 'k', linewidth = 2)
ax.set_xscale('log')
ticks = np.logspace(-3, 3, num = 7, base = 2)
ax.set_xticks(ticks)
lf = LogFormatterMathtext(base = 2)
lf.locs = list(gammas)
ax.xaxis.set_major_formatter(lf)
ax.set_xlabel(r'$\gamma=N/n$', fontsize = 35)
ax.set_ylabel(r'$\pi=n_1/n$', fontsize = 35)
ax.tick_params(axis = 'both', labelsize=22)
cbar = fig.colorbar(cntr, ax=ax)
cbar.ax.set_ylabel(r'$\log(\mathcal{M}^\star)$', rotation=270, fontsize = 30, labelpad=32)
cbar.ax.tick_params(labelsize=22)
# fig.savefig('misspecification.pdf')






pi = np.linspace(0.5, 0.99, num = 100)
length = 50
gammas = np.concatenate([np.logspace(-3, -0.1, num = length, base = 2), np.logspace(0.1, 3, num = length, base = 2)])
PI, GAMMA = np.meshgrid(pi, gammas)

M = np.zeros_like(PI)
for i in range(PI.shape[0]):
    for j in range(PI.shape[1]):
        M[i, j] = get_trisks(p = PI[i, j], gamma = GAMMA[i, j], eta = GAMMA[i, j], sigma = 1, beta = 5, delta = 1, d = 50)

# fig, ax = plt.subplots(1, 1, figsize = (12, 7.5))  
ax = axs[2]      
# contours = plt.contour(GAMMA, PI, M, 3, colors='black')
cntr = ax.contourf(GAMMA, PI, np.log(M), 20, cmap='Spectral', alpha = 0.5)
ax.plot(np.ones_like(pi), pi, color = 'k', linewidth = 2)
# plt.clabel(contours, inline=True, fontsize=8)
ax.set_xscale('log')
ticks = np.logspace(-3, 3, num = 7, base = 2)
ax.set_xticks(ticks)
lf = LogFormatterMathtext(base = 2)
lf.locs = list(gammas)
ax.xaxis.set_major_formatter(lf)
ax.set_xlabel(r'$\gamma=N/n$', fontsize = 35)
ax.set_ylabel(r'$\pi=n_1/n$', fontsize = 35)
ax.tick_params(axis = 'both', labelsize=22)
cbar = fig.colorbar(cntr, ax=ax)
cbar.ax.set_ylabel(r'$\log(R_0)$', rotation=270, fontsize = 30, labelpad=32)
cbar.ax.tick_params(labelsize=22)     

fig.savefig('combined-plot.pdf', bbox_inches = 'tight')


pi = np.linspace(0.5, 0.99, num = 100)
length = 50
gammas = np.concatenate([np.logspace(-3, -0.01, num = length, base = 2), np.logspace(0.01, 3, num = length, base = 2)])
PI, GAMMA = np.meshgrid(pi, gammas)

B, V, M = np.zeros_like(PI), np.zeros_like(PI), np.zeros_like(PI)
for i in range(PI.shape[0]):
    for j in range(PI.shape[1]):
        
        p, g, eta = PI[i, j], GAMMA[i, j], GAMMA[i, j]
        etas = eta
        ps = 0.5
        gs = 0.5 * g / (1 - p)
        
        B[i, j] = get_trisks(p = ps, gamma = gs,
                             eta = etas, sigma = 0, beta = 1, delta = 0, d = 50)
        
        V[i, j] = get_trisks(p = ps, gamma = gs,
                             eta = etas, sigma = 1, beta = 0, delta = 0, d = 50)
        
        M[i, j] = get_trisks(p = ps, gamma = gs,
                             eta = etas, sigma = 0, beta = 0, delta = 1, d = 50)

fig, ax = plt.subplots(1, 3, figsize = (30, 7.5))  
# contours = plt.contour(GAMMA, PI, M, 3, colors='black')
fig.tight_layout(pad = 4)

TITLES = [r'$\log(\mathcal{B}^\star)$', r'$\log(\mathcal{V}^\star)$', r'$\log(\mathcal{M}^\star)$']
ms = [B, V, M]
for i, (t, m)  in enumerate(zip(TITLES, ms)):
    cntr = ax[i].contourf(GAMMA, PI, np.log(m), 20, cmap='Spectral', alpha = 0.5, vmin = -2, vmax = 2)
    # plt.clabel(contours, inline=True, fontsize=8)
    ax[i].set_xscale('log')
    ticks = np.logspace(-3, 3, num = 7, base = 2)
    ax[i].set_xticks(ticks)
    lf = LogFormatterMathtext(base = 2)
    lf.locs = list(gammas)
    ax[i].xaxis.set_major_formatter(lf)
    ax[i].set_title(t, fontsize = 40)
    g = np.linspace(2 ** (-3), 1,  num = 100)
    
    ax[i].plot(g, 1 - g/2, color = 'k', linestyle = '-', linewidth = 3)
    
    ax[i].set_xlabel(r'$\gamma=N/n$', fontsize = 35)
    ax[i].tick_params(axis = 'both', labelsize=22)
    
ax[0].set_ylabel(r'$\pi=n_1/n$', fontsize = 35)


cbar = fig.colorbar(cntr, ax=ax.ravel().tolist(), pad = 0.01)
cbar.ax.tick_params(labelsize=22)
fig.savefig('subsample.pdf', bbox_inches = 'tight')

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.contour3D(np.log2(GAMMA), PI, M, 50, cmap='binary')



pi = np.linspace(0.5, 0.99, num = 100)
length = 50
gammas = np.concatenate([np.logspace(-3, -0.01, num = length, base = 2), np.logspace(0.01, 3, num = length, base = 2)])
PI, GAMMA = np.meshgrid(pi, gammas)

M = np.zeros_like(PI)
for i in range(PI.shape[0]):
    for j in range(PI.shape[1]):
        
        p, g, eta = PI[i, j], GAMMA[i, j], GAMMA[i, j]
        etas = eta
        ps = 0.5
        gs = 0.5 * g / (1 - p)
        
        M[i, j] = get_trisks(p = ps, gamma = gs,
                             eta = etas, sigma = 1, beta = 5, delta = 1, d = 50)

fig, ax = plt.subplots(1, 1, figsize = (10, 7.5))  
# contours = plt.contour(GAMMA, PI, M, 3, colors='black')
fig.tight_layout(pad = 4)
m = M
title = r'$\log(R_0)$'
cntr = ax.contourf(GAMMA, PI, np.log(m), 20, cmap='Spectral', alpha = 0.5)
# plt.clabel(contours, inline=True, fontsize=8)
ax.set_xscale('log')
ticks = np.logspace(-3, 3, num = 7, base = 2)
ax.set_xticks(ticks)
lf = LogFormatterMathtext(base = 2)
lf.locs = list(gammas)
ax.xaxis.set_major_formatter(lf)
ax.set_title(title, fontsize = 40)
g = np.linspace(2 ** (-3), 1,  num = 100)

ax.plot(g, 1 - g/2, color = 'k', linestyle = '-', linewidth = 3)

ax.set_xlabel(r'$\gamma=N/n$', fontsize = 35)
ax.tick_params(axis = 'both', labelsize=22)
    
ax.set_ylabel(r'$\pi=n_1/n$', fontsize = 35)


cbar = fig.colorbar(cntr, ax=ax, pad = 0.01)
cbar.ax.tick_params(labelsize=22)

