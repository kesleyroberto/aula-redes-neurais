
import numpy as np

entradas = np.array([1, 7, 5])
pesos_sin = np.array([0.8, 0.1, 0])

def soma(e, p):
    return e.dot(p)

def stepFunction (soma):
    if(soma>=1):
        return 1
    return 0

s = soma(entradas, pesos_sin)
r = stepFunction(s)

print("S: ", s)

print("R: ", r)