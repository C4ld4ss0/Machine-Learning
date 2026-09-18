import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

wine = load_wine()

print("--- Informações Básicas do Wine Dataset ---")
print(f"Quantidade de amostras (linhas): {wine.data.shape[0]}")
print(f"Quantidade de características (colunas): {wine.data.shape[1]}")
print(f"Nomes das características: \n{wine.feature_names}")
print(f"\nClasses existentes (Tipos de Vinho): {wine.target_names}")
print("-" * 50)

df_wine = pd.DataFrame(data=wine.data, columns=wine.feature_names)
df_wine['classe_vinho'] = [wine.target_names[target] for target in wine.target]
print("\n--- Primeiras linhas do Dataset ---")
print(df_wine.head())

X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

modelo = GaussianNB()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

print("\n--- Avaliação de Desempenho ---")
print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")

print("\n--- Relatório de Classificação ---")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

print("--- Matriz de Confusão ---")
matriz = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=wine.target_names)
disp.plot(cmap=plt.cm.Greens)
plt.title("Matriz de confusão - Naive Bayes")

os.makedirs('imagens', exist_ok=True)
plt.savefig("imagens/Matriz_Confusao_NB.png")
print("Matriz salva em 'imagens/Matriz_Confusao_NB.png'!")
plt.clf()