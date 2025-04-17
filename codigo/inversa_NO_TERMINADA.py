import numpy as np
from scipy.linalg import lu, solve_triangular


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

    return P, A_copia


def elim_gaussiana(A):
    #cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    P, A_copia = construir_P(A)
    U = A_copia
    L = np.identity(n)

    if m!=n:
        print('Estamos trabajando con una matriz no cuadrada')
        return
    

    for j in range(n):
        for i in range(j+1, n):
            L[i,j] = U[i,j] / U[j,j]
            U[i,:] = U[i,:] - L[i,j] * U[j,:]
 
    return L, U, P



def encontrar_unos_en_filas(P):
    n = P.shape[0]
    indices = [0] * n
    for i in range(n):
        for j in range(n):
            if P[i, j] == 1:
                indices[i] = j
                break
    return indices

def inversa_por_lu(A):
    n = A.shape[0]
    P, L, U = lu(A)

    I = np.eye(n)
    A_inv = np.zeros_like(A, dtype=float)

    orden_columnas = encontrar_unos_en_filas(P)

    for i in range(n):
        b = P @ I[:, i]

        y = solve_triangular(L, b, lower=True)
        x = solve_triangular(U, y)

        A_inv[:, orden_columnas[i]] = x

    return A_inv

# A = np.array([
#     [2, 1, 3],
#     [1, 0, 2],
#     [4, 1, 8]
# ], dtype=float)

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
    print('A=LU? ' , 'Si! \n' if np.allclose(np.linalg.norm(P @ A - L@U, 1), 0) else 'No! \n')
    print('Norma infinito de U: ', np.max(np.sum(np.abs(U), axis=1)) )

    A_inv = inversa_por_lu(A)
    print(A_inv)
    print("Chequeo: A @ A_inv =")
    print(A @ A_inv)  # Debería dar casi la identidad

if __name__ == "__main__":
    main()