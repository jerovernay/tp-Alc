# Carga de paquetes necesarios para graficar
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Para leer archivos
import geopandas as gpd # Para hacer cosas geográficas
import seaborn as sns # Para hacer plots lindos
import networkx as nx # Construcción de la red en NetworkX
from scipy.linalg import lu, solve_triangular


# Leemos el archivo, retenemos aquellos museos que están en CABA, y descartamos aquellos que no tienen latitud y longitud
museos = gpd.read_file('https://raw.githubusercontent.com/MuseosAbiertos/Leaflet-museums-OpenStreetMap/refs/heads/principal/data/export.geojson')
barrios = gpd.read_file('https://cdn.buenosaires.gob.ar/datosabiertos/datasets/ministerio-de-educacion/barrios/barrios.geojson')



#Matriz de Distancia

# En esta línea:
# Tomamos museos, lo convertimos al sistema de coordenadas de interés, extraemos su geometría (los puntos del mapa), 
# calculamos sus distancias a los otros puntos de df, redondeamos (obteniendo distancia en metros), y lo convertimos a un array 2D de numpy
D = museos.to_crs("EPSG:22184").geometry.apply(lambda g: museos.to_crs("EPSG:22184").distance(g)).round().to_numpy()


#Matriz de adyacencia

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


#Calculo de la inversa

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


#Matriz de Grado

def matriz_Grado (A): 
    
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


#Matriz de transicion

def matriz_Transicion1 (A):
    A_traspuesta = np.transpose(A)
    K = matriz_Grado(A)
    K_inversa = inversa_por_lu(K)
    C = A_traspuesta @ K_inversa
    
    return C


# Calculo del PageRank

def calculo_Page_Rank(A, alpha, N):
    n = A.shape[0]
    
    # Genero la matriz de transicion en base a A y una identidad en base a n de A
    C = matriz_Transicion1(A)
    I = np.eye(n)
    
    # Genero b = 1 y hago la cuenta de P
    
    M = (N/alpha) * ( I - (1-alpha) * C )
    b = np.ones(n)
    
    P = inversa_por_lu(M) @ b
    
    return P





# Aca variamos los datos, con m y alpha

m = 3 # Cantidad de links por nodo
alpha = 1/5 # Cantidas de conexiones 

A = construye_adyacencia(D,m) # Construimos la matriz de adyacencia

page_Rank = calculo_Page_Rank(A, alpha, m) # Realizamos el calculo

page_Rank = page_Rank / np.sum(page_Rank) # Normalizamos para hacer mas viable la visualizacion




# Construccion del Mapa sin nada

fig, ax = plt.subplots(figsize=(12, 12))
barrios.to_crs("EPSG:22184").boundary.plot(color='gray',ax=ax) # Graficamos Los barrios


# Armado del mapa

factor_escala = 2e4  # Escalamos los nodos para que sean visibles "(esto puedo variar)" !!!


# Construccion del mapa de redes
G = nx.from_numpy_array(A) # Construimos la red a partir de la matriz de adyacencia

# Construimos un layout a partir de las coordenadas geográficas
G_layout = {i:v for i,v in enumerate(zip(museos.to_crs("EPSG:22184").get_coordinates()['x'],museos.to_crs("EPSG:22184").get_coordinates()['y']))}


Nprincipales = 3 # Cantidad de principales
principales = np.argsort(page_Rank)[-Nprincipales:] # Identificamos a los N principales

# Imprimir información sobre los 3 museos principales
print("Los 3 museos con mayor PageRank son:", '\n')
for i, idx in enumerate(principales[::-1]):  # Invertir para mostrar en orden descendente
    print(f"{i+1}. Museo {idx}: PageRank = {page_Rank[idx]:.6f}")

labels = {n: str(n) if i in principales else "" for i, n in enumerate(G.nodes)} # Nombres para esos nodos


# Graficamos red

nx.draw_networkx(G,G_layout,
                 node_size = page_Rank*factor_escala,
                 node_color = page_Rank,
                 cmap = plt.cm.viridis,
                 ax=ax,
                 with_labels=False) 
nx.draw_networkx_labels(G, G_layout, labels=labels, font_size=10, font_color="k") # Agregamos los nombres


sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(page_Rank), vmax=max(page_Rank)))
sm._A = []
cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
cbar.set_label("PageRank")

# Añadir título y leyenda
plt.title(f'Red de Museos - PageRank (m = {m}, α = {alpha})')
plt.axis('off')


plt.show()