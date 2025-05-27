import numpy as np
from scipy.linalg import lu, solve_triangular

def inversa_por_lu(A):
    n = A.shape[0]

    # Realizamos la factorización LU de la matriz A
    P, L, U = lu(A)

    # Inicializamos la matriz identidad I
    I = np.eye(n)
    A_inv = np.zeros_like(A, dtype=float)

    # Resolvemos para cada columna de la matriz inversa
    for i in range(n):
        
        b = I[:, i]  # La columna i de la identidad

        # Resolvemos L y U
        y = solve_triangular(L, P @ b, lower=True)
        
        x = solve_triangular(U, y)

        A_inv[:, i] = x  # Guardamos el resultado en la columna i de A_inv

    return A_inv


# Primero creo K a partir de la matriz A

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
    
print (crearK(A), '\n')
print(inversa_por_lu(crearK(A)), '\n')
inversaDeK = inversa_por_lu(crearK(A))



def matriz_Transicion1 (A):
    A_traspuesta = np.transpose(A)
    K = crearK(A)
    K_inversa = inversa_por_lu(K)
    C = A_traspuesta @ K_inversa
    
    return C

print (matriz_Transicion1(A), '\n')

# Calculo del PageRank

def calculo_Page_Rank(A, alpha, N):
    
    # Genero la matriz de transicion en base a A y una identidad en base a n de A
    C = matriz_Transicion1(A)
    I = np.eye(A.shape[0])
    
    P = (alpha/N) * (inversa_por_lu( I - (1-alpha) * C ))
    
    return P 
    
