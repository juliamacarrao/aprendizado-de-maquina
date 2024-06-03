from sklearn import datasets
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#Carregando iris, iris é um dos vários datasets prontos do scikit-learn
iris = datasets.load_iris()

#atribuindo os dados da iris a variáveis
x = iris.data
y = iris.target

#Essa variável retorna 4 valores, como argumento, dou primeiro os dados que eu
#tenho e logo após as classes que eu quero descobir, test_size é a porcentagem
#dos dados de teste que eu vou utilizar, por exemplo, se eu tenho 30% para teste
#o resto que sobrou será para o treino, o random state é como uma espécie de seed
#onde o resultado de todos os alunos serão os mesmo, pode ser qualquer número
#e o stratify está classificado como y para dizer que ele terá todos os valores
#do target, ex: 3 0s, 4 1s e etc.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state=21, stratify=y)

#criando 8 pontos aleatórios no meu plano
#baseado nisso ele também vai criar 6 grupos com um conjunto de pontos
#utilizando distância euclidiana ele vai ver qual ponto pertence mais a qual grupo
#baseado nisso ele cria um ponto central para cada classe 'target'
#e nisso os pontos que estão mais perto daquele ponto central fazem parte
#daquela classe, é assim que as previsões são feitas.
knn = KNeighborsClassifier(n_neighbors=8)

#Treinando o modelo com meus dados reduzidos
#chamar o método assim faz com que o meu objeto knn
#armazene os valores do teste.
knn.fit(X_train, y_train)

#Fazendo a previsão com os dados que ele gerou pra teste
y_pred = knn.predict(X_test)

#print(f"Previsões do conjunto de teste insanooooo:\n {y_pred}")

#com o knn já com o fit, para eu ver o score eu chamo
#este método e uso como parâmetro os dados reais para ele comparar
#com os dados feitos no fit
print(knn.score(X_test, y_test))
