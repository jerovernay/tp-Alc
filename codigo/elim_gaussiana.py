#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eliminacion Gausianna
"""
import numpy as np

def elim_gaussiana(A):
    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    P, A_copia = construir_P(A)
    U = A_copia
    L = np.identity(n)

    if m!=n:
        print('Matriz no cuadrada')
        return

    for j in range(n):
        for i in range(j+1, n):
            L[i,j] = U[i,j] / U[j,j]
            U[i,:] = U[i,:] - L[i,j] * U[j,:]

            
#    L = np.tril(U,-1) + np.eye(A.shape[0]) 
#    U = np.triu(U)
    
    return L, U, P



def construir_P(A):
    n = A.shape[0]
    P = np.eye(n)
    A_copia = A.copy()

    for k in range(n):
        # Buscamos el índice del mayor pivote desde fila k hacia abajo
        p = k
        for i in range(k+1, n):
            if abs(A_copia[p, k]) == 0:
                p = i

        # Intercambiamos filas si es necesario
        if p != k:
            A_copia[[k, p], :] = A_copia[[p, k], :]
            P[[k, p], :] = P[[p, k], :]

    return P,A_copia



A = np.array([
    [0, 2, 1],
    [1, 1, 0],
    [2, 0, 1]
], dtype=float)   

def main():


    L,U,P = elim_gaussiana(A)

    print('Matriz A \n', P @ A)
    print('Matriz L \n', L)
    print('Matriz U \n', U)
    print('dobleP: \n', P )
    print('A=LU? ' , 'Si!' if np.allclose(np.linalg.norm(P @ A - L@U, 1), 0) else 'No!')
    print('Norma infinito de U: ', np.max(np.sum(np.abs(U), axis=1)) )

if __name__ == "__main__":
    main()

def hacer_b(n):

    B = np.eye(n) - np.tril(np.ones((n,n)),-1) 
    B[:n,n-1] = 1

    return B


W = hacer_b(3)          


def solve_invm(K, s):

    K_inv = np.linalg.inv(K)

    o = K_inv @ s

    return o


WL = np.array( [[ 1,  0,  0], [-1,  1,  0], [-1, -1,  1]]) 
WU = np.array([[1,0,1], [0,1,2], [0,0,4]])

b = np.array([[1],[2],[3]])

y = solve_invm(WL, b)

x = solve_invm(WU, y)

def solve_sistema(A, b):

    L,U,cant_oper = elim_gaussiana(A)

    y = solve_invm(L,b)
    x = solve_invm(U, y)

    return x



