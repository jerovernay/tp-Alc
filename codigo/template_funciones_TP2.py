# Matriz A de ejemplo
#A_ejemplo = np.array([
#    [0, 1, 1, 1, 0, 0, 0, 0],
#    [1, 0, 1, 1, 0, 0, 0, 0],
#    [1, 1, 0, 1, 0, 1, 0, 0],
#    [1, 1, 1, 0, 1, 0, 0, 0],
#    [0, 0, 0, 1, 0, 1, 1, 1],
#    [0, 0, 1, 0, 1, 0, 1, 1],
#    [0, 0, 0, 0, 1, 1, 0, 1],
#    [0, 0, 0, 0, 1, 1, 1, 0]
#])

# Carga de paquetes necesarios para graficar
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Para leer archivos
import geopandas as gpd # Para hacer cosas geográficas
import seaborn as sns # Para hacer plots lindos
import networkx as nx # Construcción de la red en NetworkX
import scipy

#Funciones TP1 


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


                                                            # ======== Funciones TP 2 ======= #


def A_sim(A):  # calculo para simetrizar red de adyacencia
    A_Traspuesta = A.T

    if np.allclose(A, A_Traspuesta):
      return A
    else:
      return np.ceil((A + A_Traspuesta) / 2 )




def calcula_L(A): # L = K - A
    A = A_sim(A)
    K = crearK(A)
    return K - A




def calcular_2E(A,n): # calculo de 2E a partir de sumatorias de casilleros de A
    dosE = 0

    for i in range(n):
      for j in range(n):
        dosE += A[i,j]

    return dosE




def crear_P(A,n):  # armado de P a partir de P[i,j] = (k[i] * k[j] ) / 2E
    dosE = calcular_2E(A,n)
    P = np.zeros((n,n))
    k = np.diag(crearK(A))

    for i in range(n):
      for j in range(n):
        P[i,j] = (k[i] * k[j] ) / dosE

    return P



def calcula_R(A): # R = A - P
    A = A_sim(A)
    n = A.shape[0]
    P = crear_P(A,n)

    return A - P



def calcula_Lambda(L,v):
    s = np.sign(v)           # s[i] = signos de la posicion i del autovector del segundo menor autovalor de L

    return 0.25 * s.T @ L @ s



def calcula_Q(R,v):
    s = np.sign(v)          # s[i] = signos de la posicion i del autovector del mayor autovalor de R

    return s.T @ R @ s




def metpot1(M, niter=10000, tol=1e-8 ):
    n = M.shape[0]
    np.random.seed(np.random.int()) #fijamos el v para cada ejecucion 
    v_sin_normalizar = np.random.rand(n)
    v = v_sin_normalizar / np.linalg.norm(v_sin_normalizar, 2)      #agarramamos un vector para inicalizar el metodo

    for i in range(niter):
      v_viejo = v.copy()

      Mv = M @ v_viejo

      v = Mv / np.linalg.norm(Mv, 2)   #movemos el vector al multiplicar por A

      if np.linalg.norm(v - v_viejo) < tol:  #si la diferencia es insignificante salimos del ciclo y nos quedamos con el ultimo autovector con diferencia significante
        break

    autovalor_1 = (v.T @ M @ v) / (v.T @ v)
    autovector_1 = v

    return autovalor_1 , autovector_1



def deflaciona(M):
    a_1, v_1 = metpot1(M)
    v_1_norm_a_2 = v_1.T @ v_1
    M1 = M - a_1 * ( np.outer(v_1, v_1) / v_1_norm_a_2 ) # deflaciona M

    return M1 , a_1, v_1



def metpotI(M, mu, niter=10000, tol=1e-8):
    n = M.shape[0]
    I = np.eye(n)
    B = M + mu * I       
    Binv = inversa_por_lu(B)
    a, v = metpot1(Binv)      #aplicamos el metodo de la potencia a la inversa de B para obtener el autovalor de menor modulo de B y su autovector asociado
    return a, v



def metpotI2(M, mu, niter=10000, tol=1e-8):
    # Recibe la matriz A, y un valor mu y retorna el segundo autovalor y autovector de la matriz A,
    # suponiendo que sus autovalores son positivos excepto por el menor que es igual a 0
   # Retorna el segundo autovector, su autovalor, y si el metodo llegó a converger.
   n = M.shape[0]
   I = np.eye(n)
   B = M + mu * I
   Binv = inversa_por_lu(B)
   defBinv, _, _ = deflaciona(Binv) # Deflacionamos la inversa para obviar el autovalor de menor modulo y buscar el segundo
   a, v =  metpot1(defBinv) # Buscamos su segundo autovector
   a = 1/a # Reobtenemos el autovalor correcto
   a -= mu
   return a, v




def laplaciano_iterativo(A,niveles,nombres_s=None):
    # Recibe una matriz A, una cantidad de niveles sobre los que hacer cortes, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.
    # La función debe, recursivamente, ir realizando cortes y reduciendo en 1 el número de niveles hasta llegar a 0 y retornar.
    if nombres_s is None: # Si no se proveyeron nombres, los asignamos poniendo del 0 al N-1
        nombres_s = range(A.shape[0])
    if A.shape[0] == 1 or niveles == 0: # Si llegamos al último paso, retornamos los nombres en una lista
        return([nombres_s])
    else: # Sino:
        L = calcula_L(A) # Recalculamos el L
        a_L, v_L = metpotI2(L, 0.5) # Encontramos el segundo autovector de L
        s = np.sign(v_L)
        indices_pos = np.where(s >= 0)[0]
        indices_neg = np.where(s < 0)[0]

        if len(indices_pos) == 0 or len(indices_neg) == 0:
              return ([nombres_s])
        # Recortamos A en dos partes, la que está asociada a el signo positivo de v y la que está asociada al negativo
        Ap = A[np.ix_(indices_pos, indices_pos)] # Asociado al signo positivo
        Am = A[np.ix_(indices_neg, indices_neg)] # Asociado al signo negativo

        return(
                laplaciano_iterativo(Ap,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v_L) if vi >= 0])
                +
                laplaciano_iterativo(Am,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v_L) if vi < 0])
                )



def modularidad_iterativo(A=None,R=None,nombres_s=None):
    # Recibe una matriz A, una matriz R de modularidad, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.

    if A is None and R is None:
        print('Dame una matriz')
        return(np.nan)
    if R is None:
        R = calcula_R(A)
    if nombres_s is None:
        nombres_s = range(R.shape[0])
    # Acá empieza lo bueno
    if R.shape[0] == 1: # Si llegamos al último nivel
        return [nombres_s]
    else:
        a,v = metpot1(R) # Primer autovector y autovalor de R
        s = np.sign(v)
        indices_pos = np.where(s >= 0)[0]
        indices_neg = np.where(s < 0)[0]
        # Modularidad Actual:
        Q0 = np.sum(R[v>0,:][:,v>0]) + np.sum(R[v<0,:][:,v<0])
        if Q0<=0 or all(v>0) or all(v<0): # Si la modularidad actual es menor a cero, o no se propone una partición, terminamos
            return [nombres_s]
        else:
            ## Hacemos como con L, pero usando directamente R para poder mantener siempre la misma matriz de modularidad
            Rp = R[np.ix_(indices_pos, indices_pos)] # Parte de R asociada a los valores positivos de v
            Rm = R[np.ix_(indices_neg, indices_neg)] # Parte asociada a los valores negativos de v
            ap,vp = metpot1(Rp) # autovector principal de Rp
            am,vm = metpot1(Rm) # autovector principal de Rm

            # Calculamos el cambio en Q que se produciría al hacer esta partición
            Q1 = 0
            if not all(vp>0) or all(vp<0):
               Q1 = np.sum(Rp[vp>0,:][:,vp>0]) + np.sum(Rp[vp<0,:][:,vp<0])
            if not all(vm>0) or all(vm<0):
                Q1 += np.sum(Rm[vm>0,:][:,vm>0]) + np.sum(Rm[vm<0,:][:,vm<0])
            if Q0 >= Q1: # Si al partir obtuvimos un Q menor, devolvemos la última partición que hicimos
                return([[ni for ni,vi in zip(nombres_s,v) if vi>0],[ni for ni,vi in zip(nombres_s,v) if vi<0]])
            else:
                # Sino, repetimos para los subniveles
                return(
                    modularidad_iterativo(
            A[np.ix_(indices_pos, indices_pos)],
            R=Rp,
            nombres_s=[nombres_s[i] for i in indices_pos])
                    +
                    modularidad_iterativo(
            A[np.ix_(indices_neg, indices_neg)],
            R=Rm,
            nombres_s=[nombres_s[i] for i in indices_neg])
                )
    