from sklearn import datasets
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

#Estilo de gráfico (plotagem) ggplot
plt.style.use('ggplot')

#Carregando iris, iris é um dos vários datasets prontos do scikit-learn
iris = datasets.load_iris()

#O dataset iris esta armazenado como um dicionário
#Nesse caso eu quero pegar somente as chaves do diconário
#print(iris.keys())
#print(iris['target'])

#atribuindo os dados da iris a variáveis
x = iris.data
y = iris.target

#Fazendo um dataframe no pandas, sendo x os dados (podem ser listas) e vou
#dizer que a classificação das colunas será o "feature_names" dentro do
#dataset iris
df = pd.DataFrame(x, columns=iris.feature_names)

#head serve para mostras as observações, ou seja as linhas da tabela
#inicialmente são 5 linhas mas você pode usar qualquer número como parâmetro
#print(df.head(10))

#exploração visual
#criar um gráfio de dispersão com o pandas, df 
pd.plotting.scatter_matrix(df[['sepal length (cm)', 'sepal width (cm)']], c=y, figsize=[8, 8], s=150, marker='.')
#plt.show()

#criando 6 pontos aleatórios no meu plano
#baseado nisso ele também vai criar 6 grupos com um conjunto de pontos
#utilizando distância euclidiana ele vai ver qual ponto pertence mais a qual grupo
#baseado nisso ele cria um ponto central para cada classe 'target'
#e nisso os pontos que estão mais perto daquele ponto central fazem parte
#daquela classe, é assim que as previsões são feitas.
knn = KNeighborsClassifier(n_neighbors=6)

#colocando meu pontos aleatórios no plano
#e a classe que ele quer descobrir
knn.fit(iris['data'], iris['target'])

#print(iris['data'].shape)
#print(iris['target'].shape)

#Dando dados aleatórios
X_new = [[6.0, 1.4, 5.1, 4.3]]

#Utilizando o método predict que baseado nos dados
#e no treinamento feito ele vai prever qual é a pétala
#que os dados apontam
prediction = knn.predict(X_new)

print ('Prediction {}'.format(prediction))