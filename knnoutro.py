import numpy as np
from sklearn.metrics import precision_score, f1_score
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, learning_curve

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

# Percorra cada fold e treine/teste o modelo
for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    # Separe os dados de treinamento e teste
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Treine o modelo
    modelo = KNeighborsClassifier(n_neighbors=11)
    modelo.fit(X_train, y_train)

     # Teste o modelo
    y_pred = modelo.predict(X_test)

     # Calcule a acurácia, a precisão e a F1-score
    score = modelo.score(X_test, y_test)
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    scores.append(score)
    precisions.append(precision)
    f1s.append(f1)

    print(f"Fold {fold_idx+1}: Acurácia = {score:.2f}, Precisão = {precision:.2f}, F1-score = {f1:.2f}")

    # # Teste o modelo
    # score = modelo.score(X_test, y_test)
    # print(f"Fold {fold_idx+1}: Acurácia = {score}")

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