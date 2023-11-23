# Importando as bibliotecas necessárias
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from flask import Flask, render_template, request
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Criando a aplicação Flask
app = Flask(__name__)

#vai pegar os datasets
def get_dataset(nome):
    data = None
    if nome == 'Iris':
        data = datasets.load_iris()
        X = data.data
        y = data.target
    elif nome == 'Breast Cancer':
        data = datasets.load_breast_cancer()
        X = data.data
        y = data.target
    else:
        # arrumando o Titanic
        url = 'https://learnenough.s3.amazonaws.com/titanic.csv'
        titanic = pd.read_csv(url)

        columns_to_drop = ['Name', 'PassengerId', 'Cabin', 'Embarked',
                           'SibSp', 'Parch', 'Ticket', 'Fare']

        for column in columns_to_drop:
            titanic = titanic.drop(column, axis=1)

        for column in ['Age', 'Sex', 'Pclass']:
            titanic = titanic[titanic[column].notna()]

        sex_int = {'male': 0, 'female': 1}
        titanic['Sex'] = titanic['Sex'].map(sex_int)

        X = titanic.drop('Survived', axis=1)
        y = titanic['Survived']
        data = titanic

    return X, y

#retorna os parametros de cada classificador, passados nos sliders do formulario
def classificador_interface(nome_clf):
    parametros = dict()
    if nome_clf == 'SVM':
        C = request.form.get('C', 1.0)
        parametros['C'] = float(C)
    elif nome_clf == 'KNN':
        K = request.form.get('K', 1)
        parametros['K'] = int(K)
    elif nome_clf == 'MLP':
        hidden_layer_sizes = request.form.get('hidden_layer_sizes', '100')
        parametros['hidden_layer_sizes'] = tuple(map(int, hidden_layer_sizes.split(',')))
    elif nome_clf == 'Decision Tree':
        max_depth = request.form.get('max_depth', '2')
        parametros['max_depth'] = int(max_depth)
    else:
        max_depth = request.form.get('max_depth', '2')
        n_estimators = request.form.get('n_estimators', '1')
        parametros['max_depth'] = int(max_depth)
        parametros['n_estimators'] = int(n_estimators)
    return parametros

#cria e configura os classificadores com base no nome
def get_classificador(nome_clf, params):
    clf = None
    if nome_clf == 'SVM':
        clf = SVC(C=params['C'])
    elif nome_clf == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif nome_clf == 'MLP':
        clf = MLPClassifier(hidden_layer_sizes=params['hidden_layer_sizes'])
    elif nome_clf == 'Decision Tree':
        clf = DecisionTreeClassifier(max_depth=params['max_depth'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                                     max_depth=params['max_depth'], random_state=1234)
    return clf


# Rota principal
@app.route('/')
def index():
    return render_template('index.html')

# essa rota é acessada quando o formulario é enviado via POST. Após o processamento do formulario
# ela treina o modelo, calcula as metricas e renderiza o "result.html"
@app.route('/result', methods=['POST'])
def result():
    # Obtenha os parâmetros do formulário
    nome_classificador = request.form['classificador']
    parametros = classificador_interface(nome_classificador)

    # Obtem os dados do dataset
    dataset_nome = request.form['dataset']
    X, y = get_dataset(dataset_nome)

    # Faz a criação do classificador
    clf = get_classificador(nome_classificador, parametros)

    # Treinamento
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    le = LabelEncoder()
    y_test_numeric = le.fit_transform(y_test)
    y_pred_numeric = le.transform(y_pred)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test_numeric, y_pred_numeric, average='weighted')
    recall = recall_score(y_test_numeric, y_pred_numeric, average='weighted')
    f1 = f1_score(y_test_numeric, y_pred_numeric, average='weighted')

    # printando os classificadores e as métricas
    print(f'Classificador = {nome_classificador}')
    print(f'Accuracy = ', acc)
    print(f'Precision = ', precision)
    print(f'Recall = ', recall)
    print(f'F1-Score = ', f1)

    # Fazendo a matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # Salvando a matriz de confusão
    temp_file_path = 'static/confusion_matrix.png'
    plt.savefig(temp_file_path)
    plt.close()

    # Retorna os resultados para a página result.html

    return render_template('result.html', parametros=parametros, acc=acc, precision=precision, recall=recall, f1=f1, dataset_size=X.shape[0])



if __name__ == '__main__':
    app.run(debug=True)