# Apuntes k-Means Clustering.
## En que casos es mejor k-Means?
Cuando los datos cambian continuamente y es necesario actualizar la información muy continuamente. O cuando se necesita clasificar en grupos o contextos.
No es tan importante que el algoritmo ya este echo sino lo importante es como lo utilizamos y como lo modificamos nosotros.

### Metas en kmeans
- Se tiene que calcular la cantidad de k, se elige normalmente el que es un cambio brutal de uno a otro.
- Otra forma es cuantos vecinos deberia de tener cada grupo, esto puede ser de la naturaleza de la problematica.
- Cual es tu sigma o cual es tu limite que se toma para decir hasta aqui se busca o este es el limite de busqueda.

En este proyecto para delimitar el numero de iteraciones y de k vamos a usar el valor de converggencia sigma.


### HOMEWORK 3
The objective of this classwork assignment is to implement the KMeans algorithm from scratch using Python.
KMeans is a popular unsupervised machine learning algorithm used for clustering data into distinct groups based on their similarity.
Following the example in the classroom drive (file ClaseKMeans.ipynb), implement the algorithm by doing the
next steps:
1. Create a class with name KMeans that aggregates all the different steps written in
the jupyter file.
2. The implementation will work for 1 dimension.
3. Use sigma as the similarity criterion for the assignment of the points to a cluster.
4. Define a convergence criterion to stop the execution of the algorithm (could be 0.005
or something representative of the data).
When you finish, upload the jupyter notebook into the google classroom with the name “Tarea3-
<name>.ipynb”, change the <name> tag to your name and the first letter of your last name (Tarea3-
AdrianM.ipynb).

## Apuntes 15/04/2024
Levenshtein Distance, es una distancia teoriaca que permite caluclar es que tan alejadas estan las palabras de otras desarolladas por 


Average precision
Minimium average precision