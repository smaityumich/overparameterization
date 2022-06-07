#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

'''
Theoretical risk for linear model
'''
def t_linrisk(beta0, delta, beta_delta, sigma , gamma, pi):
    r1 = max(beta0 *(1 - 1/gamma), 0)
    
    r2 =0
    if (gamma <1): 
        r2 = sigma**2 * (gamma/(1 - gamma))
    else:
        r2 = sigma**2/(gamma -1)
    
    r3 = 0
    if (gamma <1): 
        r3 = delta* pi * gamma/(1- gamma) + delta * pi**2 * (1 - 2*gamma)/(1 - gamma)
    else:
        r3 = delta* pi /(gamma -1) + delta * pi**2 * (gamma -2)/(gamma**2 - gamma)
    
    r4 = beta_delta * pi * min(1/gamma, 1)
    
    risk = r1+ r2 + r3 + r4
    return risk

'''
Theoreticla differnce between subsampling and ERM
'''
def t_linrisk_sub(beta0, delta, beta_delta, sigma , gamma, pi):
    gamma = gamma/(2 * (1 - pi))
    pi = 0.5
    
    r1 = max(beta0 *(1 - 1/gamma), 0)
    
    r2 =0
    if (gamma <1): 
        r2 = sigma**2 * (gamma/(1 - gamma))
    else:
        r2 = sigma**2/(gamma -1)
    
    r3 = 0
    if (gamma <1): 
        r3 = delta* pi * gamma/(1- gamma) + delta * pi**2 * (1 - 2*gamma)/(1 - gamma)
    else:
        r3 = delta* pi /(gamma -1) + delta * pi**2 * (gamma -2)/(gamma**2 - gamma)
    
    r4 = beta_delta * pi * min(1/gamma, 1)
    
    risk = r1+ r2 + r3 + r4
    return risk
    
