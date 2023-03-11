import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, f1_score, confusion_matrix
import os
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

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
    files = os.listdir(f'variosFold/Resultado9/{classe}')
    # Percorra cada arquivo na pasta
    for file in files:
        # Leia o arquivo
        data = np.loadtxt(f'variosFold/Resultado9/{classe}/{file}')
        # Adicione os dados à matriz X e o rótulo da classe à matriz y
        X = np.vstack((X, data))
        y = np.append(y, classe_idx)

print(f'Tamanho da matriz X: {X.shape}')
print(f'Tamanho da matriz y: {y.shape}')

# Salve os dados em arquivos
# np.save('Banco/X9.npy', X)
# np.save('Banco/y9.npy', y)