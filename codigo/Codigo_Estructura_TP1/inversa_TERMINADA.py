import numpy as np
from scipy.linalg import lu, solve_triangular


def inversa_por_lu(A):
    n = A.shape[0]
    P,L,U = lu(A)

    I = np.eye(n)
    A_inv = np.zeros_like(A, dtype=float)

    for i in range(n):
        
        b = I[:, i]
        
        y = solve_triangular(L, P @ b, lower=True)
        
        x = solve_triangular(U, y)

        A_inv[:, i] = x

    return A_inv

A = np.array([
    [4, 7, 2],
    [3, 5, 1],
    [2, 3, 6]
], dtype=float) 

def main():
    
    A_inv = inversa_por_lu(A)
    print(A_inv)
    print("Chequeo: A @ A_inv =")
    print(A @ A_inv)  # Debería dar casi la identidad

if __name__ == "__main__":
    main()