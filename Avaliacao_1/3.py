
import numpy as np

entradas = np.array([[0,0], [0,1], [1,0], [1,1]])
saidas = np.array([0,1,1,1])
pesos = np.array([0.0, 0.0])

print("Entradas: " +str(entradas))
print("Saidas: " +str(saidas))
print("Pesos: " +str(pesos))

aprendizado = 0.1

def stepFunction(soma):
    if(soma >=1):
        return 1
    return 0

def calculaSaida(registro):
    s = registro.dot(pesos) #recebe um par de entradas, multiplica pelos respectivos pesos e soma
    return stepFunction(s) #retorna a função de ativação a partir do calculo realizado

def treinar():
    erroTotal = 1
    while(erroTotal != 0):
        erroTotal = 0
        for i in range(len(saidas)):
            saidaCalculada = calculaSaida(np.asarray(entradas[i])) #verifica a saída para o peso atual
            erro = saidas[i] - saidaCalculada #calcula o erro para o peso atual
            print("Erro: " +str(erro))
            erroTotal += erro #soma o erro ?
            for j in range(len(pesos)): #atualizando os pesos
                pesos[j] = pesos[j] + (aprendizado*entradas[i][j]*erro)
            print("Peso Atualizado: " +str(pesos))
        print("Total do erro: " +str(erroTotal))

treinar()