# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 11:05:02 2024

@author: maxmu
"""
import numpy as np
import matplotlib.pyplot as plt
import emcee.autocorr as mc
from scipy.ndimage import convolve, generate_binary_structure
from tqdm import tqdm
from collections import Counter

L = 3
B_crit = 0.1
N = 10
q = 3
num = 10000

lattice = np.random.randint(0, q, (L, L))
betas = np.logspace(0.1 * B_crit, 10 * B_crit, num)

plt.imshow(lattice)
plt.title('Initial Lattice')
plt.show()

def initial_H(q, lattice, B, N):
    H = 0
    for y in range(L):
        for x in range(L):
            if lattice[x][y] == lattice[x][(y+1) % L]:
                    H += 1
            if lattice[x][y] == lattice[(x+1) % L][y]:
                    H += 1
    return -H

print("Initial H: ", initial_H(q, lattice, B_crit, N))

def most_common(lst):
    return max(set(lst), key=lst.count)

def metropolis(q, lattice, B, N):
    H = initial_H(q, lattice, B_crit, N)
    for n in range(N):
        f = []
        for y in range(L):
            for x in range(L):
                dH = 0
                spin_i = lattice[x][y]
                spin_f = np.random.randint(0, q)
                while spin_i == spin_f:
                    spin_f = np.random.randint(0, q)
                
                if spin_i == lattice[x][(y+1)%L]:
                    dH += 1
                if spin_i == lattice[(x+1)%L][y]:
                    dH += 1
                if spin_i == lattice[(x-1)%L][y]:
                    dH += 1
                if spin_i == lattice[x][(y-1)%L]:
                    dH += 1
                
                if spin_f == lattice[x][(y+1)%L]:
                    dH -= 1
                if spin_f == lattice[(x+1)%L][y]:
                    dH -= 1
                if spin_f == lattice[(x-1)%L][y]:
                    dH -= 1
                if spin_f == lattice[x][(y-1)%L]:
                    dH -= 1
                    
                if dH <= 0:
                    lattice[x][y] = spin_f
                    H += dH
                    f.append(spin_f)

                else:
                    u = np.random.random(1)
                    if u <= np.exp(-B * dH):
                        lattice[x][y] = spin_f
                        H += dH
                        f.append(spin_f)
                        #update f
        plt.imshow(lattice)
        f_max = Counter(f).most_common(1)[0][1]
        M = ((q * f_max) - 1) / (q - 1)

        print("Final H: ",H)
        print("M: ", M)


metropolis(q, lattice, B_crit, N)