from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Carrega o dataset
url = "C:\Projects\\ai-exercises\iris\iris.csv"
nomes = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=nomes)

# Divide o dataset de avaliação
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_treinamento, X_validacao, Y_treinamento, Y_validacao = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

# Definição dos algoritmos que serão testatdos
modelos = []
modelos.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
modelos.append(('LDA', LinearDiscriminantAnalysis()))
modelos.append(('KNN', KNeighborsClassifier()))
modelos.append(('CART', DecisionTreeClassifier()))
modelos.append(('NB', GaussianNB()))
modelos.append(('SVM', SVC(gamma='auto')))

# Avaliar cada modelo por vez
resultados = []
nomes = []
for nome, model in modelos:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_treinamento, Y_treinamento, cv=kfold, scoring='accuracy')
	resultados.append(cv_results)
	nomes.append(nome)
	print('%s: %f (%f)' % (nome, cv_results.mean(), cv_results.std()))

# Compare Algorithms
pyplot.boxplot(resultados, labels=nomes)
pyplot.title('Comparação de algoritmos')
pyplot.show()