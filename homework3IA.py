### Importamos las librerias necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### CREAMOS LA CLASE KMEANS
class KMeans:
    def __init__(self, k, max_iter):
        self.ksgima = k
        self.longitud = max_iter
        
    