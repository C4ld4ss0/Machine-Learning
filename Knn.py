import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
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

# A normalização é crucial para algoritmos baseados em distância.
# Atributos numéricos que possuem escalas muito grandes dominam o cálculo da distância, anulando a influência de atributos numéricos menores por isso a normalização. 
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

print("Matriz de Confusão está sendo gerada ...")

matriz_confusao = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=matriz_confusao, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de confusão - KNN (K = 5)")
plt.xlabel("Prevista")
plt.ylabel("Verdadeira")

os.makedirs('imagens',exist_ok=True)
plt.savefig("imagens/Matriz_de_Confusao.png")
print("A matriz está sendo salva em 'Imagens/Matriz_de_Confusao.png'!")
plt.clf()

nova_flor = np.array([[5.1, 3.5, 1.4, 0.2]])
nova_flor_norm = scaler.transform(nova_flor)
previsao_nova = knn.predict(nova_flor_norm)
nome_espec_prevista = iris.target_names[previsao_nova[0]]

print("\n--- Etapa 8: Classificação de Nova Flor ---")
print(f"Características da flor: {nova_flor[0]}")
print(f"Espécie prevista pelo KNN: {nome_espec_prevista}")

distancias, indices = knn.kneighbors(nova_flor_norm)

print("\n--- Etapa 9: Os 5 Vizinhos Mais Próximos ---")
for i in range(5):
    indice_treino = indices[0][i]
    distancia = distancias[0][i]
    classe_vizinho = Y_train[indice_treino]
    nome_classe_vizinho = iris.target_names[classe_vizinho]

    print(f"Vizinho {i+1}:")
    print(f"  - Posição no treino: {indice_treino}")
    print(f"  - Distância: {distancia:.4f}")
    print(f"  - Classe: {nome_classe_vizinho}")

valores_k = range (1, 21)
acuracias = []

for k in valores_k:
    knn_temp = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn_temp.fit(X_train_norm, Y_train)
    previsao_temp = knn_temp.predict(X_test_norm)
    acuracias.append(accuracy_score(Y_test, previsao_temp))

plt.figure(figsize=(10, 6))
plt.plot(valores_k, acuracias, marker='o', linestyle='-', color='#34AAF4', markerfacecolor='red')
plt.title('Variação da Acurácia em relação ao valor de K')
plt.xlabel('Valor de K')
plt.ylabel('Acurácia')
plt.xticks(valores_k)
plt.grid(axis='both', linestyle='--', alpha=0.5)

os.makedirs('imagens', exist_ok=True)
plt.savefig('imagens/Grafico_K_Acuracia.png')
print("\nGráfico de K vc Acurácia salvo em 'imagens/Grafico_K_Acuracia.png'!")
plt.clf()

indice_melor_k = np.argmax(acuracias)
melhor_k = valores_k[indice_melor_k]
maior_acuracia = acuracias[indice_melor_k]

print(f"\n--- Resultado da Etapa 10 ---")
print(f"O valor de K que apresentou a maior acurácia foi K={melhor_k} (Acurácia: {maior_acuracia:.4f})")

print("\n--- Etapa 11: Comparação de Distâncias (K=5) ---")

knn_euclidiana = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_euclidiana.fit(X_train_norm, Y_train)
acc_euclidiana = accuracy_score(Y_test, knn_euclidiana.predict(X_test_norm))

knn_manhattan = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
knn_manhattan.fit(X_train_norm, Y_train)
acc_manhattan = accuracy_score(Y_test, knn_manhattan.predict(X_test_norm))

print(f"Acurácia - Distância Euclidiana: {acc_euclidiana:.4f}")
print(f"Acurácia - Distância Manhattan:  {acc_manhattan:.4f}")