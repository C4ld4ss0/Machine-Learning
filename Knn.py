import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

iris = datasets.load_iris()

print(f"Quantidade de instâncias: {iris.data.shape[0]}")
print(f"Quantidade de atributos: {iris.data.shape[1]}")
print(f"Nomes dos atributos: {iris.feature_names}")
print(f"Classes existentes: {iris.target_names}")

print('-' * 40)

df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)

df_iris['especie'] = [iris.target_names[target] for target in iris.target]

print(df_iris.head())

X = iris.data
Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=42)

print(f"Tamanho do treino: {X_train.shape[0]} instâncias")
print(f"Tamanho do teste: {X_test.shape[0]} instâncias")
print("-" * 40)

scaler = StandardScaler()

X_train_norm = scaler.fit_transform(X_train)

X_test_norm = scaler.transform(X_test)

# --- COMENTÁRIO EXIGIDO PELA ATIVIDADE ---
# A normalização é crucial para algoritmos baseados em distância (como o KNN).
# Sem ela, atributos numéricos que possuem escalas muito grandes dominariam 
# o cálculo da distância, anulando a influência de atributos numéricos menores. 
# A padronização coloca todas as características em pé de igualdade.

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

knn.fit(X_train_norm, Y_train)

Y_pred = knn.predict(X_test_norm)

print("\n--- Resultados (Primeiras 10 instâncias) ---")
print(f"Classes Reais:    {Y_test[:10]}")
print(f"Classes Previstas: {Y_pred[:10]}")

print("\n--- Avaliação de Desempenho ---")
acuracia = accuracy_score(Y_test, Y_pred)
print(f"Acurácia: {acuracia:.4f}")

print("\nRelatório de Classificação (Precision, Recall, F1-score):")
print(classification_report(Y_test, Y_pred, target_names=iris.target_names))

print("Matriz de COnfusão está sendo gerada ...")

matriz_confusao = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=matriz_confusao, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de confusão - KNN (K = 5)")
plt.show()