import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, f1_score, confusion_matrix
import os
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

# Inicialize as listas de resultados
scores = []
precisions = []
f1s = []

# Defina as classes
classes = ['Boa', 'CascaGrossa', 'Podre', 'Praga', 'Verde']

# Defina o número de atributos em cada arquivo
num_atributos = 256

# Inicialize as matrizes de dados X e os rótulos y
X = np.zeros((0, num_atributos))
y = np.zeros((0,))

# Percorra cada classe e leia os arquivos
for classe_idx, classe in enumerate(classes):
    print(classe)
    # Obtenha uma lista de todos os arquivos na pasta
    files = os.listdir(f'variosFold/Resultado3/{classe}')
    # Percorra cada arquivo na pasta
    for file in files:
        # Leia o arquivo
        data = np.loadtxt(f'variosFold/Resultado3/{classe}/{file}')
        # Adicione os dados à matriz X e o rótulo da classe à matriz y
        X = np.vstack((X, data))
        y = np.append(y, classe_idx)

# Defina o número de folds da validação cruzada
num_folds = 10

# Inicialize o objeto StratifiedKFold
skf = StratifiedKFold(n_splits=num_folds)

# Inicialize a lista de pontuações do modelo
train_scores = []
test_scores = []

# Percorra cada fold e treine/teste o modelo
for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    # Separe os dados de treinamento e teste
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Treine o modelo
    modelo = KNeighborsClassifier(n_neighbors=11)
    modelo.fit(X_train, y_train)

    # Armazene as pontuações do modelo
    train_scores.append(modelo.score(X_train, y_train))
    test_scores.append(modelo.score(X_test, y_test))

    # Teste o modelo
    y_pred = modelo.predict(X_test)

    cmnormalize = confusion_matrix(y_test, y_pred, normalize='true')
    print(cmnormalize)

    # Crie a matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    

    # Calcule a acurácia, a precisão e a F1-score
    score = modelo.score(X_test, y_test)
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    scores.append(score)
    precisions.append(precision)
    f1s.append(f1)

    print(f"Fold {fold_idx+1}: Acurácia = {score:.2f}, Precisão = {precision:.2f}, F1-score = {f1:.2f}")

# Calcule a média e o desvio padrão das métricas
mean_score = np.mean(scores)
std_score = np.std(scores)
mean_precision = np.mean(precisions)
std_precision = np.std(precisions)
mean_f1 = np.mean(f1s)
std_f1 = np.std(f1s)

# Imprima as médias e os desvios padrão das métricas
print(f"Média da acurácia = {mean_score:.2f} +/- {std_score:.2f}")
print(f"Média da precisão = {mean_precision:.2f} +/- {std_precision:.2f}")
print(f"Média do F1-score = {mean_f1:.2f} +/- {std_f1:.2f}")

train_sizes = np.linspace(0.1, 1.0, 10)
train_sizes, train_scores, test_scores = learning_curve(estimator=modelo, X=X, y=y, train_sizes=train_sizes, cv=skf, n_jobs=-1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure()
plt.title("Curva de Aprendizagem")
plt.xlabel("Tamanho do conjunto de treinamento")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
plt.xticks(np.arange(0, 1.1, 0.1))
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
train_scores_mean + train_scores_std, alpha=0.1,
color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
label="Pontuação de treinamento")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
label="Pontuação de validação cruzada")
plt.legend(loc="best")
plt.show()

# Plote a matriz de confusão
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=classes, yticklabels=classes,
       title='Matriz de Confusão',
       ylabel='Valor Real',
       xlabel='Valor Previsto')

# Adicione as anotações na matriz de confusão
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

# Adicione uma barra de cores à direita da matriz de confusão
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Número de amostras", rotation=-90, va="bottom")

# Mostra o gráfico
plt.show()