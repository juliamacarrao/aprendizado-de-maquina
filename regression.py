import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


boston_csv = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"

#Fazendo a leitura do arquivo csv
boston = pd.read_csv(boston_csv)

#Printando as primeiras linhas do csv
#print(boston.head())

#Criando os vetores de recursos e alvo

#Deletando a coluna target, ele por padrão deleta linhas, porém usando axis eu posso
#dizer para ele deletar a coluna
X = boston.drop('medv', axis=1,).values

#E aqui eu estou acessando apenas os valores da coluna do meu target
y = boston['medv'].values

#print(X, '\n')
#print(y)

#Prevendo o preço a partir de um único recurso
#Vamos pegar as linha da 5° coluna, para pegar linhas específicar é só colocar
#n°:n° e a mesma coisa com as colunas.
X_rooms = X[:, 5]

#printando todos os dados de X
#print(X)
#vendo qual é o tipo da variável
#print(type(X_rooms))
#printando todos os dados de y
#print(y)
#vendo o tipo da variável y
#print(type(y))

#O X_rooms estava como lista e agora os dados estão em colunas, a mesma coisa co
#o y
X_rooms = X_rooms.reshape(-1, 1)
y = y.reshape(-1, 1)

#print(X)
#print(y)

#Valor médio vs n de quarto
#Criando gráfico de dispersão
#plt.scatter(X_rooms, y)
#Definindo as legendas dos eixos x e y do gráfico
#plt.ylabel('Valor da casa /1000 ($)')
#plt.xlabel('Número de quartos')
#Mostrar o gráfico
#plt.show()

#Instanciando a classe do linear regression
reg = linear_model.LinearRegression()
#Colocando os dados no modelo
reg.fit(X_rooms, y)
#Criando meu espaço linear, que nada mais é do que a média entretodos aqueles pontos para
#criar uma linha que vai meio que generalizar esse valores, e o reshape serve
#para transformar a visualização da lista para uma coluna, o 1 faz esse conversão.
prediction_space = np.linspace(min(X_rooms), max(X_rooms)).reshape(-1, 1)

#Criando o gráfico de dispersão, sendo a bolinhas dar cor azul
plt.scatter(X_rooms, y, color='blue')
#Colocando a linha no meio do gráfico, basicamente, estamos definindo como x
#o espaço de previsão, que nada mais é do que uma coluna com todos os
#números que formam a linha e o y é a previsão de y baseado em x, sendo reg
#o método para criar a regressão linear 
plt.plot(prediction_space, reg.predict(prediction_space), color='black', linewidth=3)
plt.show()

