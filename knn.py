from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
import numpy as np

classes = ['Boa', 'CascaGrossa', 'Podre', 'Praga', 'Verde']

X = np.load('Banco/X9.npy')
y = np.load('Banco/y9.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)

scores = cross_val_score(knn, X_train, y_train, cv=10)

print('Mean cross-validation score: {:.5f}'.format(np.mean(scores)))

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print('Test set accuracy: {:.5f}'.format(knn.score(X_test, y_test)))

print(classification_report(y_test, y_pred))

# Acurácia individual de cada classe
for i, classe in enumerate(classes):
    class_accuracy = accuracy_score(y_test[y_test == i], y_pred[y_test == i])
    print(f'Acurácia da classe {classe}: {class_accuracy:.5f}')
