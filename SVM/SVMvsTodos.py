from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.svm import SVC

X = np.load('Banco/X3.npy')
y = np.load('Banco/y3.npy')

X = X[y != 4]
y = y[y != 4]

# Verificar se há mais de uma classe no conjunto de dados
num_classes = len(np.unique(y))

print(f"O conjunto de dados possui {num_classes} classes.")

if num_classes > 1:
    # Lista para armazenar os resultados
    class_results = []

    for class_label in np.unique(y):
        print(f"\nClass {class_label} vs All")

        # Criar uma cópia dos rótulos originais para não modificar o y original
        y_copy = y.copy()

        # Definir a classe atual como 1 e todas as outras como 0
        y_copy_binary = np.where(y_copy == class_label, 1, 0)

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        acc_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for train_index, test_index in skf.split(X, y_copy_binary):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y_copy_binary[train_index], y_copy_binary[test_index]

            svm = SVC(kernel='rbf', gamma=1, C=1000)

            svm.fit(X_train, y_train)

            y_pred = svm.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            acc_scores.append(acc)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        # Armazenar os resultados para a classe atual
        class_result = {
            'Class': class_label,
            'Acurácia Média': np.mean(acc_scores),
            'Precisão Média': np.mean(precision_scores),
            'Revocação Média': np.mean(recall_scores),
            'F1-Score Médio': np.mean(f1_scores)
        }

        class_results.append(class_result)

    # Imprimir os resultados para cada classe vs All
    print("\nResultados para cada classe vs All:")
    for result in class_results:
        print(f"\nResultados para a Classe {result['Class']} vs All:")
        print(f"Acurácia Média: {result['Acurácia Média']:.4f}")
        print(f"Precisão Média: {result['Precisão Média']:.4f}")
        print(f"Revocação Média: {result['Revocação Média']:.4f}")
        print(f"F1-Score Médio: {result['F1-Score Médio']:.4f}")

    # Agora, calcular métricas para All vs Classe
    all_vs_class_results = []

    for class_label in np.unique(y):
        print(f"\nAll vs Class {class_label}")

        # Criar uma cópia dos rótulos originais para não modificar o y original
        y_copy = y.copy()

        # Definir a classe atual como 0 e a classe 'class_label' como 1
        y_copy_binary = np.where(y_copy == class_label, 0, 1)

        acc_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for train_index, test_index in skf.split(X, y_copy_binary):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y_copy_binary[train_index], y_copy_binary[test_index]

            svm = SVC(kernel='rbf', gamma=1, C=1000)

            svm.fit(X_train, y_train)

            y_pred = svm.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            acc_scores.append(acc)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        # Armazenar os resultados para All vs a classe atual
        all_vs_class_result = {
            'Class': class_label,
            'Acurácia Média': np.mean(acc_scores),
            'Precisão Média': np.mean(precision_scores),
            'Revocação Média': np.mean(recall_scores),
            'F1-Score Médio': np.mean(f1_scores)
        }

        all_vs_class_results.append(all_vs_class_result)

    # Imprimir os resultados para All vs Classe
    print("\nResultados para All vs Classe:")
    for result in all_vs_class_results:
        print(f"\nResultados para All vs Classe {result['Class']}:")
        print(f"Acurácia Média: {result['Acurácia Média']:.4f}")
        print(f"Precisão Média: {result['Precisão Média']:.4f}")
        print(f"Revocação Média: {result['Revocação Média']:.4f}")
        print(f"F1-Score Médio: {result['F1-Score Médio']:.4f}")

else:
    print("O conjunto de dados possui menos de duas classes.")
