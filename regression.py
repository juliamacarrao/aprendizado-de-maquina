import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
plt.scatter(X_rooms, y)
#Definindo as legendas dos eixos x e y do gráfico
plt.ylabel('Valor da casa /1000 ($)')
plt.xlabel('Número de quartos')
#Mostrar o gráfico
plt.show()
