import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

X9 = np.load('Banco/X3.npy')
y9 = np.load('Banco/y3.npy')

X9 = X9[y9 != 4]
y9 = y9[y9 != 4]

num_classes = len(np.unique(y9))
print(f'Número total de classes: {num_classes}')

X_train, X_test, y_train, y_test = train_test_split(
    X9, y9, test_size=0.3, random_state=42)

# Escolhe os parâmetros do SVM
kernel = 'rbf'
gamma = 1
C = 1000

# Treina o modelo SVM com os parâmetros especificados
svm_model = SVC(kernel=kernel, gamma=gamma, C=C)
svm_model.fit(X_train, y_train)

# Realiza as previsões no conjunto de teste
y_pred = svm_model.predict(X_test)

# Calcula a precisão e a revocação para cada classe
accuracy = accuracy_score(y_test, y_pred)
precision_per_class = precision_score(y_test, y_pred, average=None)
recall_per_class = recall_score(y_test, y_pred, average=None)

print(f'Acurácia: {accuracy:.4f}')
# Imprime a precisão e a revocação para cada classe
for class_label, (precision, recall) in enumerate(zip(precision_per_class, recall_per_class)):
    print(f'Classe {class_label}:')
    print(f'  Precisão: {precision:.4f}')
    print(f'  Revocação: {recall:.4f}')

# Cria a matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Plota a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title(f'Matriz de Confusão - SVM (Kernel={kernel}, Gamma={gamma}, C={C})')
plt.show()
