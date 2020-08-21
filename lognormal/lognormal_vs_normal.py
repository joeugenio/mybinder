#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 15:53:41 2020

@author: Joel Eugênio - joelecjr@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt

# Lognormal parameters
S = 1
MU = 0
log_mean = np.exp(MU+(S**2/2))
log_sd = np.sqrt(np.exp(2*MU+S**2)*(np.exp(S**2)-1))

# Lognormal PDF
def p_lognorm(x, sigma=1, mu=0):
    pdf = np.zeros(x.shape)
    i = 0
    for j in x:
        if (j > 0):
            pdf[i] = (1/(j*sigma*np.sqrt(2*np.pi)))*np.exp(-.5*((np.log(j)-mu)/sigma)**2)
        i += 1
    return pdf

# Normal PDF
def p_norm(x, sd=1, m=0):
    return (1/(sd*np.sqrt(2*np.pi)))*np.exp(-.5*((x-m)/sd)**2)

# Laplace PDF
def p_lap(x, sd=1, m=0):
    a = np.sqrt(2)/sd
    return  (a/2)*np.exp(-a*np.abs(x-m))

n = np.linspace(-10,10,1000)

print('Lognormal pdf with mean {:.2f} ({:.2f}) and SD {:.2f}'.format(0,log_mean, log_sd))
p1 = p_lognorm(n+log_mean)
print('Gaussian pdf with mean {:.2f} and SD {:.2f}'.format(0, log_sd))
p2 = p_norm(n, sd=log_sd)
print('Laplace pdf with mean {:.2f} and SD {:.2f}'.format(0, log_sd))
p3 = p_lap(n, sd=log_sd)

plt.plot(n,p1,label='lognormal')
plt.plot(n,p2,label='gaussiana')
plt.plot(n,p3,label='laplace')
plt.axvline(0, 0, 1, ls='--', color='k',linewidth=1)
plt.legend()
plt.show()