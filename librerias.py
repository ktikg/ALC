import numpy as np
#Ejercicio 1. Desarrollar una funcion esCuadrada(A) que devuelva verdadero
#si la matriz A es cuadrada y Falso en caso contrario.

A = np.array([[1,2,3],[4,5,6],[7,8,9]])
B = np.array([[10,20,30],[-1,-2,-3]])
C = np.array([[]])
D = np.array([[1]])
E = np.array([[1,2,3,4,5,6]])
F = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
G = np.array([[1],[2],[3],[4],[5]])
H = np.array([[1,2,3],[2,4,5],[3,5,8]])

def esCuadrada(A): #esta funcion nada mas toma en cuenta matrices validas? con igual len en cada columna?
    filas = (A.shape)[0]
    columnas = (A.shape)[1]
    return filas == columnas

assert esCuadrada(A)
assert not esCuadrada(B)
assert not esCuadrada(C) #preguntar
assert esCuadrada(D)
assert not esCuadrada(E)
assert not esCuadrada(F)
assert not esCuadrada(G)
print("Todos los tests pasaron")


#Ejercicio 2. Desarrollar una funcion triangSup(A) que devuelva la matriz U correspondiente a la matriz Triangular Superior de A sin su diagonal

def triangSup(A): #solo se le van a pasar matrices cuadradas?
    filas = (A.shape)[0]
    columnas = (A.shape)[1]
    U = np.zeros((filas,columnas))
    for i in range(filas):
        for j in range(columnas):
            if i < j:
                U[i][j] = A[i][j]
    return U

print(triangSup(A))
print(triangSup(B))
print(triangSup(E))
print(triangSup(F))
print(triangSup(G))

#Ejercicio 3. Desarrollar una funcion triangInf(A) que devuelva la matriz L correspondiente a la matriz Triangular Inferior de A sin su diagonal.

def triangInf(A):
    filas = (A.shape)[0]
    columnas = (A.shape)[1]
    U = np.zeros((filas,columnas))
    for i in range(filas):
        for j in range(columnas):
            if i > j:
                U[i][j] = A[i][j]
    return U


print(triangInf(A))
print(triangInf(B))
print(triangInf(E))
print(triangInf(F))
print(triangInf(G))

#Ejercicio 4. Desarrollar una funcion diagonal(A) que devuelva la matriz D correspondiente a la matriz diagonal de A.

def diagonal(A):
    filas = (A.shape)[0]
    columnas = (A.shape)[1]
    D = np.zeros((filas,columnas))
    for i in range(filas):
        for j in range(columnas):
            if i == j:
                D[i][j] = A[i][j]
    return D

print(diagonal(A))
print(diagonal(B))
print(diagonal(E))
print(diagonal(F))
print(diagonal(G))

#Ejercicio 5. Desarrollar una funcion traza(A) que calcule la traza de una matriz cualquiera A.

def traza(A):
    filas = (A.shape)[0]
    columnas = (A.shape)[1]
    traza = 0 
    for i in range(filas):
        for j in range(columnas):
            if i == j:
                traza = traza + A[i][j]
    return traza

assert (traza(A) == 15)
assert (traza(B) == 8)
assert (traza(C) == 0)
assert (traza(D) == 1)
assert (traza(E) == 1)
assert (traza(F) == 15)
assert (traza(G) == 1)
print("Todos los tests pasaron")

#Ejercicio 6. Desarrollar una funcion traspuesta(A) que devuelva la matriz traspuesta de A.

def traspuesta(A): #POR REFERENCIA O POR COPIA?
    filas = (A.shape)[0]
    columnas = (A.shape)[1]
    traspA = np.zeros((columnas,filas))
    for i in range(filas):
        for j in range(columnas):
            traspA[j][i] = A[i][j]
    return traspA

print(traspuesta(A))
print(traspuesta(B))
print(traspuesta(D))
print(traspuesta(E))
print(traspuesta(F))
print(traspuesta(G))

#Ejercicio 7. Desarrollar una funcion esSimetrica(A) que devuelve True si la matriz A es simetrica y False en caso contrario

def esSimetrica(A):
    res = esCuadrada(A)
    trasp = traspuesta(A)
    filas = A.shape[0]
    columnas = A.shape[1]
    i = 0
    j = 0
    while (res == True and i < filas):
        while (res == True and j < columnas):
            res = (A[i][j] == trasp[i][j])
            j += 1
        i += 1
        j = 0
    return res

assert not esSimetrica(A)
assert not esSimetrica(B)
assert esSimetrica(D)
assert esSimetrica(H)
print("Todos los tests pasaron")

#Ejercicio 8. Desarrollar una funcion calcularAx(A,x) que recibe una matriz A de tamano n × m y un vector x de largo m y devuelve un vector b de largo n resultado de la multiplicacion vectorial de la matriz y el vector.


def calcularAx(A,x):
    filas = (A.shape)[0]
    columnas = (A.shape)[1]
    b = np.zeros(filas, dtype = int)
    for i in range (filas):
        for j in range (columnas):
            b[i] += x[j]*A[i][j]
    return b

VectorA = np.array([1,2,3])
VectorB = np.array([20,105,-5])
VectorD = np.array([55])
VectorG = np.array([10])

print(calcularAx(A,VectorA))
print(calcularAx(B,VectorB))
print(calcularAx(D,VectorD))
print(calcularAx(G,VectorG))

#Ejercicio 9. Desarrollar una funcion intercambiarFilas(A, i, j), que intercambie las filas i y la j de la matriz A. El intercambio tiene que ser in-place

def intercambiarFilas(A,i,j):
    B = A.copy()
    columnas = (A.shape)[1]
    for c in range(columnas):
        A[i][c] = B[j][c]
        A[j][c] = B[i][c]

        
(intercambiarFilas(A,0,1))
(intercambiarFilas(B,0,1))
(intercambiarFilas(C,0,0))
(intercambiarFilas(D,0,0))
(intercambiarFilas(F,3,0))
print(A)
print(B)
print(C)
print(D)
print(F)

        
#Ejercicio 10. Desarrollar una funcion sumar fila multiplo(A, i, j, s) que a la fila i le sume la fila j multiplicada por un escalar s. Esta es una operacion elemental clave en la eliminacion gaussiana. La operacion debe ser in-place.

def multiplo(A,i,j,s):
    B = A.copy()
    columnas = (A.shape)[1]
    for c in range (columnas):
        A[i][c] += s*B[j][c]

multiplo(A,1,0,-4)
multiplo(B,0,1,10)
multiplo(F,2,1,5)
multiplo(G,3,1,-4)
print(A)
print(B)
print(F)
print(G)

#Ejercicio 11. Desarrollar una funcion esDiagonalmenteDominante(A) que devuelva True si una matriz cuadrada A es estrictamente diagonalmente dominante. Esto ocurre si para cada fila, el valor absoluto del elemento en la diagonal es mayor que la suma de los valores absolutos de los demas elementos en esa fila.

def esDiagonalmenteDominante(A):
    filas = (A.shape)[0]
    columnas = (A.shape)[1]
    elementoDiag = 0
    suma = 0
    res = True
    i = 0
    j = 0
    res = esCuadrada(A)
    while (res == True and i < (filas)):
        elementoDiag = A[i][i]
        while (res == True and j <(columnas)):
            suma += abs(A[i][j])
            j +=1
        suma -= abs(elementoDiag)
        if (abs(elementoDiag) < suma):
            res = False
        i +=1
    return res


I = np.array([[3,-2,1],[1,-3,2],[-1,2,4]])
J = np.array([[-4,2,1],[1,6,2],[1,-2,5]])
DD = np.array([[5,2,2],[1,-5,2],[10,-100,200]])
print (A)
assert not esDiagonalmenteDominante(A)
assert not esDiagonalmenteDominante(B)
assert esDiagonalmenteDominante(D)
assert esDiagonalmenteDominante(I)
assert esDiagonalmenteDominante(DD)
assert esDiagonalmenteDominante(J)
print("Todos los tests pasaron")
        
#Ejercicio 12. Desarrollar una funcion matrizCirculante(v) que genere una matriz circulante a partir de un vector. En una matriz circulante la primer fila es igual al vector v, y en cada fila se encuentra una permutacion cıclica de la fila anterior, moviendo los elementos un lugar hacia la derecha.


def matrizCirculante(v):
    largo = v.size
    res = np.array.zeros((largo,largo))
    i = 0


#Ejercicio 13. Desarrollar una funcion matrizVandermonde(v), donde v ∈ R n y se devuelve la matriz de Vandermonde V ∈ R n×n cuya fila i-esima corresponde con la potencias (i − 1)-esima de los elementos de v.
        
            
def matrizVandermonde(v):
    largo = v.size
    res = np.zeros((largo,largo))
    for i in range(largo):
        potencia = 0
        for j in range (largo):
            res[i][j] = v[i] ** potencia
            potencia += 1
    return res

print(matrizVandermonde(np.array([2,3,5])))
print(matrizVandermonde(np.array([1,2,3,4,5])))


#Ejercicio 14. Desarrollar una funcion numeroAureo(n) que estime el numero aureo ϕ como Fk+1/Fk, siendo Fk el k-esimo numero de la sucesion de Fibonacci. Para esto, formulen la sucesion de Fibonacci Fk+1 = Fk + Fk−1 de forma matricial, usando la semilla F0 = 0, F1 = 1. Grafique el valor aproximado de ϕ en funcion del numero de pasos de la sucesion considerado.

#PREGUNTAAAAAAAR


#Ejercicio 15. Desarrollar una funcion matrizFiboncacci(n), que genera una matriz A de n×n, y cada aij = Fi+j , siendo Fk el k-esimo numero de la sucesion de Fibonacci (considerando F0 = 0, F1 = 1).

def fibonacci(n):
    res = 0
    if (n==0) or (n==1):
        res = n 
    else:
        f0 = 0
        f1 = 1
        for i in range (n - 1):
            res = f0 + f1
            f0 = f1
            f1 = res
    return res

def matrizFibonacci(n):
    res = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            res[i][j] = fibonacci(i+j)
    return res

print(matrizFibonacci(3))
print(matrizFibonacci(5))
print(matrizFibonacci(1))

#Ejercicio 16. Desarrollar una funcion matrizHilbert(n), que genera una matriz de Hilbert H de n × n, y cada hij = 1/i+j+1 .

def matrizHilbert(n):
    res = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            res[i][j] = 1/(i+j+1)
    return res

print(matrizHilbert(1))
print(matrizHilbert(4))
print(matrizHilbert(7))


#Ejercicio 17. Usando las funciones previamente desarrolladas donde sea posible, escriba una rutina que calcule los valores entre -1 y 1 de los siguientes polinomios:

#Grafique el valor de los polinomios en el rango indicado, y calcule la cantidad de operaciones necesarias y el espacio en memoria para generar 100 puntos equiespaciados entre -1 y 1. ¿Como crecen estos valores con n? ¿Que modificarıa para hacer el calculo mas eficiente?

#Ejercicio 18. Modificar la funcion row echelon de manera que evalue en cada pivot si no hay otro elemento de la misma columan con m´odulo mayor (en valor absoluto). En caso afirmativo hacer el swap de las filas. Esta operatoria permite tener mayor estabilidad numerica.

#P


        
    