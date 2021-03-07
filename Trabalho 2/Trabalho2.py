# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 15:42:59 2021

@author: Kesley
"""


from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#verificando o tamanho dos conjuntos
#60000 imagens para treino e 10000 para teste

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

#plotando dataset
for i in range(10):
    plt.subplot(2, 5, i + 1)
    imagem_pixels = x_train[i]
    imagem_label = y_train[i]

    plt.title(imagem_label)
    plt.imshow(imagem_pixels)

plt.show()

#A rede aqui implementada, terá uma camada de input com 28x28=784 neurônios (número de pixels), uma camada de 256 com um Dropout de 0.2, e a camada de 
#saída, contendo 10 neurônios (número de classes do dataset).

#A camada de input, que é do tipo Flatten, o que significa que a entrada é uma matriz de 28x28 e o seu output é um vetor de dimensão 728. 
#Que está #totalmenta conectada (fully connected) coma a próxima camada, que por sinal são densas (Dense), ou seja, os 728 neurônios estão 
#conectados com os 256, que por sua vez estão conetados com os 10 de saída.


modelo = Sequential()
modelo.add( Flatten(input_shape=(28, 28)) )
modelo.add( Dense(256, activation="sigmoid") )
modelo.add( Dropout(0.2) )
modelo.add( Dense(10, activation="sigmoid") )


#compilação
#passa a função de otimização e de perda

modelo.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


#treino
x_treino_preprocessado = x_train / 255.0
epochs = 100

historico = modelo.fit(
    x_treino_preprocessado, y_train,
    epochs = epochs,
    validation_split = 0.2
)

#historico

print(historico.history)

#acuracia x epocas
plt.figure(1)
plt.title("Acurácia")
plt.plot(range(1, epochs+1), historico.history["accuracy"], label="acurácia")
plt.plot(range(1, epochs+1 ), historico.history["val_accuracy"], label="acurácia em validação")
plt.xlabel("Época")
plt.ylabel("Porcentagem")
plt.legend()
plt.show()


#função de perda x epocas
plt.figure(2)
plt.title("Loss")
plt.plot(range(1, epochs+1), historico.history["loss"], label="Perda")
plt.plot(range(1, epochs+1 ), historico.history["val_loss"], label="Perda em validação")
plt.xlabel("Época")
plt.ylabel("Porcentagem")
plt.legend()
plt.show()


print("Média da acurácia: ", np.array(historico.history["accuracy"]).mean())
print("Desvio padrão da acurácia: ", np.array(historico.history["accuracy"]).std())
print("Média da perda: ", np.array(historico.history["loss"]).mean())
print("Desvio padrão da perda: ", np.array(historico.history["loss"]).std())


# index = 3000
# predicoes = modelo.predict( np.array([x_test[index] / 255.0]) )
# print("Predição:", np.argmax(predicoes[0]))
# print("Real:", y_test[index])