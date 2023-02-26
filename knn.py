import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier

X = []
y = []
num_arquivos = 227  # substitua pelo número total de arquivos
for i in range(1, num_arquivos+1):
    if os.path.exists(f'Resultado/Boa/histBoa{i}.txt'):
        arquivo = f'Resultado/Boa/histBoa{i}.txt'
        resultado = np.loadtxt(arquivo)
        X.append(resultado)
        y.append("boa")  # substitua classe pela classe correspondente aos resultados do arquivo
    if os.path.exists(f'Resultado/CascaGrossa/histCascaGrossa{i}.txt'):
        arquivo = f'Resultado/CascaGrossa/histCascaGrossa{i}.txt'
        resultado = np.loadtxt(arquivo)
        X.append(resultado)
        y.append("cascagrossa")
X = np.array(X)
y = np.array(y)

modelo = KNeighborsClassifier(n_neighbors=3)
modelo.fit(X, y)