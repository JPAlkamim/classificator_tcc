import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
import seaborn as sns

X3 = np.load('Banco/X3.npy')
y3 = np.load('Banco/y3.npy')

# Verificar se há mais de uma classe no conjunto de dados

X3 = X3[y3 != 4]
y3 = y3[y3 != 4]

num_classes = len(np.unique(y3))
print(f"O conjunto de dados possui {num_classes} classes.")

if num_classes > 1:
    # Lista para armazenar os resultados
    class_results = []

    k_value = 3

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for class_label in np.unique(y3):
        print(f"\nClass {class_label} vs Todas as Outras Classes")

        # Criar uma cópia dos rótulos originais para não modificar o y3 original
        y_copy = y3.copy()

        # Definir a classe atual como 1 e todas as outras como 0
        y_copy_binary = np.where(y_copy == class_label, 1, 0)

        acc_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for train_index, test_index in skf.split(X3, y_copy_binary):
            X_train, X_test = X3[train_index], X3[test_index]
            y_train, y_test = y_copy_binary[train_index], y_copy_binary[test_index]

            knn = KNeighborsClassifier(n_neighbors=k_value)
            knn.fit(X_train, y_train)

            y_pred = knn.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            acc_scores.append(acc)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        # Armazenar os resultados para Classe vs Todas as Outras Classes
        class_result = {
            'Class': class_label,
            'Acurácia Média': np.mean(acc_scores),
            'Precisão Média': np.mean(precision_scores),
            'Revocação Média': np.mean(recall_scores),
            'F1-Score Médio': np.mean(f1_scores)
        }
        class_results.append(class_result)

        # Todas as Outras Classes vs Classe
        print(f"\nTodas as Outras Classes vs Class {class_label}")

        # Criar uma cópia dos rótulos originais para não modificar o y3 original
        y_copy = y3.copy()

        # Definir a classe atual como 0 e a classe 'class_label' como 1
        y_copy_binary = np.where(y_copy == class_label, 0, 1)

        acc_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for train_index, test_index in skf.split(X3, y_copy_binary):
            X_train, X_test = X3[train_index], X3[test_index]
            y_train, y_test = y_copy_binary[train_index], y_copy_binary[test_index]

            knn = KNeighborsClassifier(n_neighbors=k_value)
            knn.fit(X_train, y_train)

            y_pred = knn.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            acc_scores.append(acc)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        # Armazenar os resultados para Todas as Outras Classes vs Classe
        class_result = {
            'Class': class_label,
            'Acurácia Média': np.mean(acc_scores),
            'Precisão Média': np.mean(precision_scores),
            'Revocação Média': np.mean(recall_scores),
            'F1-Score Médio': np.mean(f1_scores)
        }
        class_results.append(class_result)

    # Imprimir os resultados
    print("\nResultados para Classe vs Todas as Outras Classes:")
    for result in class_results:
        print(f"\nResultados para a Classe {result['Class']} vs Todas as Outras Classes:")
        print(f"Acurácia Média: {result['Acurácia Média']:.4f}")
        print(f"Precisão Média: {result['Precisão Média']:.4f}")
        print(f"Revocação Média: {result['Revocação Média']:.4f}")
        print(f"F1-Score Médio: {result['F1-Score Médio']:.4f}")

else:
    print("O conjunto de dados possui menos de duas classes.")
