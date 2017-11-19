from sklearn import linear_model
from sklearn import datasets
from sklearn import svm
import pickle
from  sklearn.externals import joblib

reg = linear_model.LinearRegression()

reg.fit([[0, 0], [1,1], [2, 2]], [0, 1, 2])

print (reg.coef_)

digitos = datasets.load_digits()

print (digitos.data)
print (digitos.target)

#implementacao de uma Support Vector Machine
#Parametro gamma define o alcance da influencia de um unico exemplo de treino, quanto menor o valor maior o alcance de influencia
#Parametro C e o trade-off de uma classificacao errada contra a simplicidade da area de decisao, quanto maior o C, mais
#focado em classificar todos os exemplos corretamente o modelo estara
clf = svm.SVC(gamma=0.001, C=100)

#Treina o modelo
clf.fit(digitos.data[:-1], digitos.target[:-1])

#Prediz o valor do ultimo campo do vetor, deixado de fora do treinamento
print (clf.predict(digitos.data[-1:]))

m = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target

m.fit(X, y)

#Salva o modelo com pickle,
model_save = pickle.dumps(m)
clf2 = pickle.loads(model_save)
print (clf2.predict(X[0:1]))

#Salva o modelo usando o joblib, mais recomendado para Big Data
joblib.dump(m, "C:\\Users\\Gustavo\\PycharmProjects\\scikitTest\\file.pkl")
newModel = joblib.load("C:\\Users\\Gustavo\\PycharmProjects\\scikitTest\\file.pkl")

print (newModel)