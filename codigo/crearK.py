# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 18:07:12 2025

@author: pedro
"""
import numpy as np

#primero creo K a partir de la matriz A
def crearK (A): 
    n = A.shape[0]
    m = A.shape[1]
    K = np.zeros((m, n)) 
    sumaFilasA = np.sum(A, axis = 1)
    
    if m!=n:
        print('Estamos trabajando con una matriz no cuadrada')
        return
    for i in range (len (sumaFilasA)):
        K[i, i] = sumaFilasA[i] 
        
    return K    

A = np.array([
    [0, 1, 1, 1],
    [0, 0, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 1, 0]
], dtype=float) 
    
#print (crearK(A))

#Ahora busco K^-1
def crearKmenosuno (A):
    n = A.shape[0]
    m = A.shape[1]
    Kmenos = np.zeros((m, n)) 
    sumaFilasA = np.sum(A, axis = 1)
    
    if m!=n:
        print('Estamos trabajando con una matriz no cuadrada')
        return
    for i in range (len (sumaFilasA)):
        if sumaFilasA[i] != 0:
            Kmenos[i, i] = 1 / sumaFilasA[i] 
        
    return Kmenos 

#print(crearKmenosuno(A))

#Ahora puedo usar lo anterior para conseguir C
def crearC (A):
    A_traspuesta = np.transpose(A)
    Kmenos = crearKmenosuno(A)
    C = A_traspuesta @ Kmenos
    return C

print (crearC(A))
    

#Probe con el ejemplo del power y la C queda joya


    
    
