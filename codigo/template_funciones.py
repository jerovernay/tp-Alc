# Carga de paquetes necesarios para graficar
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Para leer archivos
import geopandas as gpd # Para hacer cosas geográficas
import seaborn as sns # Para hacer plots lindos
import networkx as nx # Construcción de la red en NetworkX
import scipy


def construye_adyacencia(D,m): 
    
    # Función que construye la matriz de adyacencia del grafo de museos
    # D matriz de distancias, m cantidad de links por nodo
    # Retorna la matriz de adyacencia como un numpy.
    
    D = D.copy()
    l = [] # Lista para guardar las filas
    
    for fila in D: # recorriendo las filas, anexamos vectores lógicos
        l.append(fila<=fila[np.argsort(fila)[m]] ) # En realidad, elegimos todos los nodos que estén a una distancia menor o igual a la del m-esimo más cercano
    A = np.asarray(l).astype(int) # Convertimos a entero
    np.fill_diagonal(A,0) # Borramos diagonal para eliminar autolinks
    
    return(A)


#comentar



def calculaLU(A):
    
    # matriz es una matriz de NxN
    # Retorna la factorización LU a través de una lista con dos matrices L y U de NxN.
    
    def construir_P(A):
        
        n = A.shape[0]
        P = np.eye(n) # comentar aca
        A_permutada = A.copy()
    
        for k in range(n):
            #Tomamos los valores de la columna k desde la fila k  hasta el final
            columna = A_permutada[k:, k]
    
            #Hacemos que todos los valores de la columna sean su absoluto
            largo_columna_abs = np.abs(columna)
    
            #Buscamos el indice de la columna al que le pertenece el valor mas grande
            max_indice_columna = 0
            maxValor = largo_columna_abs[0]
    
            for i in range(1, len(columna)):
    
                if largo_columna_abs[i] > maxValor:
                    maxValor = largo_columna_abs[i]
                    max_indice_columna = i
    
            #Calculamos el indice correcto de la fila en A
            p = k + max_indice_columna
    
    
            # Intercambiamos filas en A_permutada y en P si es necesario
            if p != k:
    
                #Intercambiamos en A_copia
                A_permutada[[k, p], :] = A_permutada[[p, k], :]
    
                #Intercambiamos en P
                P[[k, p], :] = P[[p, k], :]
    
        return P, A_permutada
    
    
    P, A_permutada = construir_P(A) #Consigo la P, y en caso de que P != I la A con la filas reordenadas
    m = A.shape[0]
    n = A.shape[1]
    
    if m!=n:
        print('Matriz no cuadrada')
        return
    
    U = A_permutada # Comienza siendo una copia de A y deviene en U (triangulado superiormente)
    L = np.identity(n)  # comentar aca !!!
    
    

    for j in range(n):
        for i in range(j+1,n):
            L[i,j] = U[i,j] / U[j,j]
            U[i,:] = U[i,:] - L[i,j] * U[j,:]

    return L, U, P



# Calculo de la inversa usando descomposicion de LU para cualquier matriz inversible

def inversa_por_lu(A):
    n = A.shape[0]

    # Realizamos la factorización LU de la matriz A
    L, U, P = calculaLU(A)

    # Inicializamos la matriz identidad I
    I = np.eye(n)
    A_inv = np.zeros_like(A, dtype=float) #comentar aca !!!!

    # Resolvemos para cada columna de la matriz inversa
    for i in range(n):

        b = I[:, i]  # La columna i de la identidad

        # Resolvemos L y U
        y = scipy.linalg.solve_triangular(L, P @ b, lower=True)

        x = scipy.linalg.solve_triangular(U, y)

        A_inv[:, i] = x  # Guardamos el resultado en la columna i de A_inv

    return A_inv




def calcula_matriz_C(A): 
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C
    
    
    # Primero creo K a partir de la matriz A

    def crearK (A):
    
        n = A.shape[0]
        m = A.shape[1]
        K = np.zeros((m, n))
        sumaFilasA = np.sum(A, axis = 1)
    
    
        for i in range (len (sumaFilasA)):
            K[i, i] = sumaFilasA[i]
    
        return K
    
    # Nuestra matriz de transicion esta definida por A_traspuesta y K_inv, como nos pide la ecuacion (2)
    
    A_traspuesta = np.transpose(A) # Trasponemos A 
    K = crearK(A) # Creamos K
    K_inv = inversa_por_lu(K) # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de A
    C = A_traspuesta @ K_inv # Calcula C multiplicando A_traspuesta y K_inv
    
    return C


    
def calcula_pagerank(A,alpha):
    
    # Función para calcular PageRank usando LU
    # A: Matriz de adyacencia
    # d: coeficientes de damping
    # Retorna: Un vector p con los coeficientes de page rank de cada museo
    
    
    # Genero la matriz de transicion en base a A y una identidad en base a n de A
    
    n = A.shape[0]  # Dimension de A
    C = calcula_matriz_C(A)
    N = n   # Obtenemos el número de museos N a partir de la estructura de la matriz A
    I = np.eye(n)
    
    # Variamos la ecuacion dad, ya que tenemos una formula general para invertir con LU
    
    M = (N/alpha) * ( I - (1-alpha) * C )
    b = np.ones(n)  # Vector de 1s, multiplicado por el coeficiente correspondiente usando d y N.
    
    p = inversa_por_lu(M) @ b 
    
    return p

def calcula_matriz_C_continua(D): 
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C en versión continua
    D = D.copy()
    F = 1/D
    np.fill_diagonal(F,0)
    Kinv = ... # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de F 
    C = ... # Calcula C multiplicando Kinv y F
    return C

def calcula_B(C,cantidad_de_visitas):
    # Recibe la matriz T de transiciones, y calcula la matriz B que representa la relación entre el total de visitas y el número inicial de visitantes
    # suponiendo que cada visitante realizó cantidad_de_visitas pasos
    # C: Matirz de transiciones
    # cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
    # Retorna:Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v
    B = np.eye(C.shape[0])
    for i in range(cantidad_de_visitas-1):
        # Sumamos las matrices de transición para cada cantidad de pasos
    return B