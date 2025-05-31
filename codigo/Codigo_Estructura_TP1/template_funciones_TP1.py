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


# Funcion del calculo de LU

def calculaLU(A):
    
    # A es una matriz de NxN
    # Retorna la factorización LU a través de una lista con dos matrices L y U de NxN.
    
    m = A.shape[0]
    n = A.shape[1]
    
    if m!=n:
        print('Matriz no cuadrada')
        return
    
    U = A.copy() # Comienza siendo una copia de A y deviene en U (triangulado superiormente)
    L = np.identity(n)  # Parto desde una matriz identidad de dimension nxn, que va a mutar a L
    
    

    for j in range(n): 
        for i in range(j+1,n):
            L[i,j] = U[i,j] / U[j,j]
            U[i,:] = U[i,:] - L[i,j] * U[j,:]

    return L, U



# Calculo de la inversa usando descomposicion de LU para cualquier matriz inversible

def inversa_por_lu(A):
    n = A.shape[0]

    # Realizamos la factorización LU de la matriz A
    L, U = calculaLU(A)

    # Inicializamos la matriz identidad I y la inversa
    I = np.eye(n)
    A_inv = np.zeros_like(A) 

    # Resolvemos para cada columna de la matriz inversa
    for i in range(n):

        b = I[:, i]  # La columna i de la identidad

        # Resolvemos L y U
        y = scipy.linalg.solve_triangular(L, b, lower=True)

        x = scipy.linalg.solve_triangular(U, y)

        A_inv[:, i] = x  # Guardamos el resultado en la columna i de A_inv

    return A_inv


# Funcion auxiliar para el calculo de K (matriz de grado)
# Creo K a partir de la matriz A

def crearK (A):
    
    # A: Matriz de adyacencia

    n = A.shape[0]
    K = np.zeros((n, n)) #Armo una matriz de ceros para rellenar sus casilleros
    sumaFilasA = np.sum(A, axis = 1)
    
    
    for i in range (len (sumaFilasA)):
        K[i, i] = sumaFilasA[i]
    
    return K


# Función para calcular la matriz de trancisiones C

def calcula_matriz_C(A): 

    # A: Matriz de adyacencia
    # Retorna la matriz C
    
    # Nuestra matriz de transicion esta definida por A_traspuesta y K_inv, como nos pide la ecuacion (2)
    
    A_traspuesta = np.transpose(A) # Trasponemos A 
    K = crearK(A) # Creamos K
    K_inv = inversa_por_lu(K) # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de A
    C = A_traspuesta @ K_inv # Calcula C multiplicando A_traspuesta y K_inv
    
    return C


# Función para calcular PageRank usando LU

def calcula_pagerank(A,alpha):

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



# Función para calcular la matriz de trancisiones C

def calcula_Cji(D, j , i): #Funcion auxiliar para calcular los casilleros Cji
    
    #Recibe la matriz de distancias y las coordenadas del casillero que necesito utilizar para rellenar el casillero en C (con esas mismas coordenadas)
    
        N = D.shape[0]
    
    #Como nos presentacon el casillero como una division, separo procesos entre el numerador y el denominador 
    
        num = 1 / D[i,j] #defino el numerador ( F(dij) )
        den = 0

        for k in range(1, N): #Armo la sumatoria del denominador
            if k!= i: 
                den = den + (1 / D[i,k])

        return num / den

def calcula_matriz_C_continua(D):

    #D: Recibe la matriz de distancias
    
    n = D.shape[0]
    C = np.zeros((n,n)) # armo una matriz de ceros para rellenar los casilleros, dejando la diagonal de ceros

    for j in range(n):
        for i in range(n):
            if i != j:
              C[j,i] = calcula_Cji(D, j, i)

    return C

def calcula_B(C,cantidad_de_visitas):

    # Recibe la matriz C de transiciones, y calcula la matriz B que representa la relación entre el total de visitas y el número inicial de visitantes
    # suponiendo que cada visitante realizó cantidad_de_visitas pasos
    # C: Matirz de transiciones
    # cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
    # Retorna:Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v

    n = C.shape[0]
    B = np.eye(n)
    for k in range(1, cantidad_de_visitas):      # Comienza a iterar desde 1 porque B = I = C ^ 0
        C_elevada_k = np.linalg.matrix_power(C,k)
        B = B + C_elevada_k                       # Sumamos las matrices de transición para cada cantidad de pasos
    return B

#Funcion para calcular en numero Condicion de normal 1 de B
def nro_condicion_norma1(B):

    B_inv = inversa_por_lu(B)

    norma1_de_B = np.linalg.norm(B, 1)
    norma1_de_B_inv = np.linalg.norm(B_inv, 1)

    return norma1_de_B * norma1_de_B_inv
