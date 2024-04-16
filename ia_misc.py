import numpy as np
import pandas as pd

# Distances
# Euclidean Distance
def euclidian_distance(p, q):
    return np.sqrt(np.sum((q - p)**2))

# Funcion de Regresion Lineal por Gradient Descent
def linRegGD(m,b,x,y,L):
    # Valores iniciales de m y b
    m_pred = 0
    b_pred = 0
    
    # Numero total de datos
    n = len(x)
    
    for i in range(len(x)):
        # Derivada parcial de MSE con respecto a m
        m_pred += -2/n * x[i] * (y[i] - (m*x[i] + b))
        # Derivada parcial de MSE con respecto a b
        b_pred += -2/n * (y[i] - (m*x[i] + b))
    
    # Aplicacion de L = Learning rate como paso descendente
    # para aproximar al error minimo
    return m - L*m_pred, b - L*b_pred

# Function de Regresion Lineal por Minimos Cuadrados
def linRegLS(x,y):
    # Numero total de datos
    n = len(x)
    
    # Funcion en terminos de m
    m = (n*np.sum(x*y) - np.sum(x)*np.sum(y)) / (n*np.sum(x**2) - np.sum(x)**2)
    # Funcion en terminos de b
    b = (np.sum(y) - m*np.sum(x)) / n
    
    return m, b

# Generación de tres clusters con datos aleatorios normalizados
def getClusters(n,xs2,xs3):
    x1 = (np.random.randn(1,n)/np.cos(2*np.pi)).flatten()
    x1 = x1 - np.min(x1)
    y1 = (np.random.randn(1,n)/np.cos(2*np.pi)).flatten()
    y1 = y1 - np.min(y1)
    x2 = (np.random.randn(1,n)/np.cos(2*np.pi)).flatten()
    x2 = x2 - np.min(x2) + xs2
    y2 = (np.random.randn(1,n)/np.cos(2*np.pi)).flatten()
    y2 = y2 - np.min(y2) + 2
    x3 = (np.random.randn(1,n)/np.cos(2*np.pi)).flatten()
    x3 = x2 - np.min(x2) + xs3
    y3 = (np.random.randn(1,n)/np.cos(2*np.pi)).flatten()
    y3 = y2 - np.min(y2) - 3
    
    c1 = np.zeros(n)
    
    df_clusters = {
        'x': np.concatenate([x1,x2,x3]),
        'y': np.concatenate([y1,y2,y3]),
        'c': np.concatenate(
            [np.zeros(n,int),
             np.zeros(n,int) + 1,
             np.zeros(n,int) + 2]
        )
    }
    
    return df_clusters

# Definición del objeto KNN (k-Nearest Neighbors)
class KNN:
    # Atributo classes
    classes = []
    
    # Constructor del objeto KNN que recibe como argumento
    # el número de vecinos cercanos que votarán
    # El argumento tendrá un valor predeterminado de 1
    def __init__(self, k = 1) -> None:
        self.k = k
    
    # Método del objeto KNN para checar la distancia euclidiana
    # las entradas son dos arreglos que representan 'x' y 'y' o
    # 'z' dependiendo si va a calcular una, dos o más dimensiones
    def euclidian_distance(self, p, q):
        return np.sqrt(np.sum((q - p)**2))
    
    # Método de entrenamiento donde vamos a llenar el atributo 
    # classes con los clústeres de entrenamiento, el método reshape
    # lo usaremos para cambiar los datos de renglones a columnas 
    # y usaremos concatenate para unir las tres columnas en una matriz
    # para tener los datos de la siguiente forma:
    # x    | y     | c
    # 1.2  | 0.8   | 0
    # 0.1  | 1.4   | 1
    # 4.5  | 3.1   | 2
    def training(self, train_data):
        self.classes = train_data
    
    # Método predict, recibe un punto q a comparar con sus vecinos para
    # realizar predecir a qué clase pertenece sin importar la dimensión.
    def predict(self, q):
        # Arreglo de distancias
        distances = []
        # Calculo de la distancia por cada renglón en el conjunto de datos
        # de entrenamiento.
        for row in self.classes:
            # Con append se agregan dos valores a la lista de distancias
            # la distancia entre los puntos de n dimensiones y a qué clase
            # corresponde del set de entrenamiento.
            # Ejemplo:
            # distancia | clase
            # 2.5       | 1
            # 2.8       | 2
            # 3.2       | 1
            # 0.8       | 1
            p = np.array(row[:len(q)])
            
            distances.append(
                [self.euclidian_distance(q,p),row[-1]]
            )
        
        # Usando el método sorted, ordenamos el arreglo con las distancias
        # y las clases de menor a mayor con respecto a la distancia:
        # Ejemplo:
        # distancia | clase
        # 0.8       | 1
        # 2.5       | 1
        # 2.8       | 2
        # 3.2       | 1
        sorted_dist = sorted(distances)
        
        # Hacemos el conteo de las clases que aparecen en los primeros
        # k más cercanos
        # Ejemplo donde k = 3
        # clase | frecuencia
        # 1     | 2
        # 2     | 1
        vote_cnt = dict()
        for ks, class_row in enumerate(sorted_dist):
            if ks < self.k:
                # Se añade al diccionario la clase y con el método get se obtiene
                # lo que había anteriormente en la nueva llave, si la llave es nueva
                # entonces se pone un valor predeterminado de 0
                # Ejemplo
                # Suponiendo que la llave 1 no existía
                # vote_cnt[1] = vote_cnt.get(1,0) + 1
                # Como no existía vote_cnt.get(1,0) entregará un valor de 0
                # y al sumarlo con 1 tenemos que la primera vez que se añade
                # una nueva llave, se cuenta 1 vez
                # La siguiente vez se vería así
                # vote_cnt[1] = vote_cnt.get(1,0) + 1
                # Con la misma llave vote_cnt.get(1,0) entregaría un valor de 1
                # por lo que la segunda vez que se cuente en esa llave se sumará
                # teniendo un valor de 2
                vote_cnt[int(class_row[1])] = vote_cnt.get(int(class_row[1]),0) + 1
            else:
                break
        
        # Una vez obtenido el conteo de la frecuencia de las clases
        # se busca el que tenga la mayor cantidad de "votos" o en este
        # caso que tenga una mayor frecuencia en el conteo:
        # Ejemplo:
        # clase | frecuencia
        #-----------
        #|1     | 2| <- Este sería elegido
        #-----------
        # 2     | 1
        vote_max = 0
        vote_class = -1
        for classes in vote_cnt:
            if vote_max < vote_cnt[classes]:
                vote_max = vote_cnt[classes]
                vote_class = classes
        
        # Se regresa el valor de la clase más votada
        return np.int32(vote_class)

def trainPredSeparation(df,sex,g1,g2):
    x_all = df[df['sex'] == sex][g1].to_numpy()
    y_all = df[df['sex'] == sex][g2].to_numpy()
    rnd_index = np.random.permutation(len(x_all))
    rnd_train = np.array(range(int(len(x_all)*0.7)))
    rnd_predc = np.array(range(int(len(x_all)*0.7),len(x_all)))
    x_train = x_all[rnd_index[rnd_train]]
    y_train = y_all[rnd_index[rnd_train]]
    x_predc = x_all[rnd_index[rnd_predc]]
    y_predc = y_all[rnd_index[rnd_predc]]
    
    return x_train,y_train,x_predc,y_predc

# Obtener arreglos de entrenamiento y predicción para regresión lineal
def trainPredSeparationLR(df, x, y):
    x_all = df[x].to_numpy()
    y_all = df[y].to_numpy()
    rnd_index = np.random.permutation(len(x_all))
    rnd_train = np.array(range(int(len(x_all)*0.7)))
    rnd_predc = np.array(range(int(len(x_all)*0.7),len(x_all)))
    x_train = x_all[rnd_index[rnd_train]]
    y_train = y_all[rnd_index[rnd_train]]
    x_predc = x_all[rnd_index[rnd_predc]]
    y_predc = y_all[rnd_index[rnd_predc]]
    
    return x_train, y_train, x_predc, y_predc

def trainPredSeparationLRNF(x,y):
    x_all = x
    y_all = y
    rnd_index = np.random.permutation(len(x_all))
    rnd_train = np.array(range(int(len(x_all)*0.7)))
    rnd_predc = np.array(range(int(len(x_all)*0.7),len(x_all)))
    x_train = x_all[rnd_index[rnd_train]]
    y_train = y_all[rnd_index[rnd_train]]
    x_predc = x_all[rnd_index[rnd_predc]]
    y_predc = y_all[rnd_index[rnd_predc]]
    
    return x_train,y_train,x_predc,y_predc

# Obtener diccionario de entrenamiento y clasificación para KNN
def trainPredSeparationKNN1D(x,c):
    rnd_index = np.random.permutation(len(x))
    rnd_train = np.array(range(int(len(x)*0.7)))
    rnd_predc = np.array(range(int(len(x)*0.7),len(x)))
    x_train = x[rnd_index[rnd_train]]
    c_train = c[rnd_index[rnd_train]]
    x_predc = x[rnd_index[rnd_predc]]
    c_predc = c[rnd_index[rnd_predc]]
    
    train_data = np.concatenate([
        x_train.reshape(-1,1),
        c_train.reshape(-1,1)
    ],axis=1)
    
    pred_data = np.concatenate([
        x_predc.reshape(-1,1),
        c_predc.reshape(-1,1)
    ],axis=1)
    
    return train_data,pred_data

# Obtener diccionario de entrenamiento y clasificación para KNN
def trainPredSeparationKNN2D(x,y,c):
    rnd_index = np.random.permutation(len(x))
    rnd_train = np.array(range(int(len(x)*0.7)))
    rnd_predc = np.array(range(int(len(x)*0.7),len(x)))
    x_train = x[rnd_index[rnd_train]]
    y_train = y[rnd_index[rnd_train]]
    c_train = c[rnd_index[rnd_train]]
    x_predc = x[rnd_index[rnd_predc]]
    y_predc = y[rnd_index[rnd_predc]]
    c_predc = c[rnd_index[rnd_predc]]
    
    train_data = np.concatenate([
        x_train.reshape(-1,1),
        y_train.reshape(-1,1),
        c_train.reshape(-1,1)
    ],axis=1)
    
    pred_data = np.concatenate([
        x_predc.reshape(-1,1),
        y_predc.reshape(-1,1),
        c_predc.reshape(-1,1)
    ],axis=1)
    
    return train_data,pred_data

def trainPredSeparationKNN3D(x,y,z,c):
    rnd_index = np.random.permutation(len(x))
    rnd_train = np.array(range(int(len(x)*0.7)))
    rnd_predc = np.array(range(int(len(x)*0.7),len(x)))
    x_train = x[rnd_index[rnd_train]]
    y_train = y[rnd_index[rnd_train]]
    z_train = z[rnd_index[rnd_train]]
    c_train = c[rnd_index[rnd_train]]
    x_predc = x[rnd_index[rnd_predc]]
    y_predc = y[rnd_index[rnd_predc]]
    z_predc = z[rnd_index[rnd_predc]]
    c_predc = c[rnd_index[rnd_predc]]
    
    train_data = np.concatenate([
        x_train.reshape(-1,1),
        y_train.reshape(-1,1),
        z_train.reshape(-1,1),
        c_train.reshape(-1,1)
    ],axis=1)
    
    pred_data = np.concatenate([
        x_predc.reshape(-1,1),
        y_predc.reshape(-1,1),
        z_predc.reshape(-1,1),
        c_predc.reshape(-1,1)
    ],axis=1)
    
    return train_data,pred_data

# Coeficiente de Pearson para medir el nivel de correlación entre datos x y
def pearsonCoef(x,y):
    n = len(x)
    
    nom = n*np.sum(x*y)-np.sum(x)*np.sum(y)
    den = np.sqrt((n*np.sum(x**2)-np.sum(x)**2)*(n*np.sum(y**2)-np.sum(y)**2))
    return nom/den