import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

X3 = np.load('Banco/X3.npy')
y3 = np.load('Banco/y3.npy')

X_train, X_test, y_train, y_test = train_test_split(
    X3, y3, test_size=0.3, random_state=42)

print(f'Número total de classes: {len(np.unique(y3))}')
# Escolhe o valor de n_neighbors
k_value = 3

# Treina o modelo com o valor escolhido de n_neighbors
knn = KNeighborsClassifier(n_neighbors=k_value)
knn.fit(X_train, y_train)

# Realiza as previsões no conjunto de teste
y_pred = knn.predict(X_test)

# Calcula a precisão para cada classe
precision_per_class = precision_score(y_test, y_pred, average=None)

# Imprime a precisão para cada classe
for class_label, precision in enumerate(precision_per_class):
    print(f'Precisão para a classe {class_label}: {precision:.4f}')

# Calcula a revocação para cada classe
recall_per_class = recall_score(y_test, y_pred, average=None)

# Imprime a revocação para cada classe
for class_label, recall in enumerate(recall_per_class):
    print(f'Revocação para a classe {class_label}: {recall:.4f}')

# Cria a matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Plota a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title(f'Matriz de Confusão - KNN (K={k_value})')
plt.show()
