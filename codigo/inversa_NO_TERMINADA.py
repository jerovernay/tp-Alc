import numpy as np
from scipy.linalg import lu, solve_triangular


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

A = np.array([
    [2, 1, 3],
    [1, 0, 2],
    [4, 1, 8]
], dtype=float)

A_inv = inversa_por_lu(A)
print(A_inv)
print("Chequeo: A @ A_inv =")
print(A @ A_inv)  # Debería dar casi la identidad
