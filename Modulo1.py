import numpy as np
def error(x,y):
  return np.float64(abs(x-y))
 


def error_relativo(x,y):
  abs_x = np.float64(abs(x))
  if abs_x == 0:
    return 0.0
  return error(x, y) / abs_x
 

def matricesIguales(A,B):
  res: bool = True
  filas = A.shape[0]
  columnas = A.shape[1]
  tolerancia = 1e-9 
  i = 0
  j = 0
  if (A.shape != B.shape):     
      res = False
  else:
      while(res == True and i <filas ):
        while (res == True and j <columnas):
            if abs(A[i, j] - B[i, j]) > tolerancia:
                res = False 
            j+=1
        i+= 1
  return res
 

def sonIguales(x,y,atol=1e-08):
  return np.allclose(error(x,y),0,atol=atol)

assert(not sonIguales(1,1.1))
assert(sonIguales(1,1 + np.finfo('float64').eps))
assert(not sonIguales(1,1 + np.finfo('float32').eps))
assert(not sonIguales(np.float16(1),np.float16(1) + np.finfo('float32').eps))
assert(sonIguales(np.float16(1),np.float16(1) + np.finfo('float16').eps,atol=1e-3))

assert(np.allclose(error_relativo(1,1.1),0.1))
assert(np.allclose(error_relativo(2,1),0.5))
assert(np.allclose(error_relativo(-1,-1),0))
assert(np.allclose(error_relativo(1,-1),2))
assert(matricesIguales(np.diag([1,1]),np.eye(2)))
assert(matricesIguales(np.linalg.inv(np.array([[1,2],[3,4]]))@np.array([[1,2],[3,4]]),np.eye(2)))
assert(not matricesIguales(np.array([[1,2],[3,4]]).T,np.array([[1,2],[3,4]])))
print ("paso")