#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V: number of vocabulary size
L: number of message length
A: number of attributes
N: number of types in each attribute
"""
import numpy as np
import matplotlib.pyplot as plt

def num_compositional(V, L, A, N):
    '''
        N!*(V!/(V-A)!)^N
    '''
    V, L, A, N = int(V), int(L), int(A), int(N)
    left = np.math.factorial(N)
    right = (np.math.factorial(V)/np.math.factorial(V-A))**N
    return left*right
    
def num_holi_comp(V, L, A, N):
    '''
        (V^L)!/(V^L-A^N)! - num_compositional(N, V, A)
    '''
    V, L, A, N = int(V), int(L), int(A), int(N)
    all_lan = np.math.factorial(V**L)/np.math.factorial(V**L-A**N)
    tmp_comp= num_compositional(V, L, A, N)
    tmp_holi = all_lan - tmp_comp
    return tmp_holi, tmp_comp

# ========= For most of current examples ============
SIZE = 8
START = 2
x_axis = np.arange(START,START+SIZE,1)
V = x_axis#*np.ones((SIZE,))
L = START*np.ones((SIZE,))
A = START*np.ones((SIZE,))
N = START*np.ones((SIZE,))

y_comp = []
y_holi = []
for i in range(SIZE):
    tmp_holi, tmp_comp = np.log(num_holi_comp(V[i], L[i], A[i], N[i]))
    y_holi.append(tmp_holi)
    y_comp.append(tmp_comp)

plt.plot(x_axis,y_holi,'r-x')
plt.plot(x_axis,y_comp, 'g-s')
plt.show()


'''
def test(V, L=2, A1=4, A2=8, N=2):
    left = np.math.factorial(V**L)/np.math.factorial(V**L-A1*A2)
    right = np.math.factorial(N)*(np.math.factorial(V)/np.math.factorial(V-A1))* \
    (np.math.factorial(V)/np.math.factorial(V-A2))
    return left-right, right

SIZE = 8
START = 8
x_axis = np.arange(START,START+SIZE,1)
V = x_axis#*np.ones((SIZE,))

y_comp = []
y_holi = []
for i in range(SIZE):
    tmp_holi, tmp_comp = np.log(test(V[i], L=2, A1=4, A2=8, N=2))
    y_holi.append(tmp_holi)
    y_comp.append(tmp_comp)

plt.plot(x_axis,y_holi,'r-x')
plt.plot(x_axis,y_comp, 'g-s')
plt.show()   
'''