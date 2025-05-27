import numpy as np

# Matriz A de ejemplo
A_ejemplo = np.array([
   [0, 1, 1, 1, 0, 0, 0, 0],
   [1, 0, 1, 1, 0, 0, 0, 0],
   [1, 1, 0, 1, 0, 1, 0, 0],
   [1, 1, 1, 0, 1, 0, 0, 0],
   [0, 0, 0, 1, 0, 1, 1, 1],
   [0, 0, 1, 0, 1, 0, 1, 1],
   [0, 0, 0, 0, 1, 1, 0, 1],
   [0, 0, 0, 0, 1, 1, 1, 0]
])

def crearK (A):
    
    # A: Matriz de adyacencia

    n = A.shape[0]
    K = np.zeros((n, n)) #Armo una matriz de ceros para rellenar sus casilleros
    sumaFilasA = np.sum(A, axis = 1)
    
    
    for i in range (len (sumaFilasA)):
        K[i, i] = sumaFilasA[i]
    
    return K

k_demostracion = crearK(A_ejemplo)

print(f"Visualizacion de K: {k_demostracion}")
print(f"\nVisualizacion de A: {A_ejemplo}")
L = k_demostracion - A_ejemplo  
print(f"\nVisualizacion de L: {L}", '\n')



vector_1 = np.array([1,1,1,1,1,1,1,1])

print(A_ejemplo @ vector_1, '\n')
print(k_demostracion @ vector_1, '\n')