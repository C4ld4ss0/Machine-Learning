import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode

# Dados informados pela própria professora
dados = {
 'Horas_Estudo': [2, 3, 4, 5, 6, 7, 8, 9, 10, 6],
 'Frequencia': [65, 70, 72, 78, 80, 85, 88, 92, 95, 82],
 'Nota_Prova': [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 7.2],
 'Nota_Trabalho': [5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.2, 7.8]
}

df = pd.DataFrame(dados)

print("--- Médias ---")
print(df.mean())

print("\n--- Medianas ---")
print(df.median())

print("\n--- Modas ---")
print(df.mode())

print("\n--- Amplitude ---")
amplitude = df.max() - df.min()
print(amplitude)

print("\n--- Variância ---")
print(df.var().round(2))

print("\n--- Desvio Padrão ---")
print(df.std().round(2))

print("\n--- Matriz de Covariância ---")
print(df.cov().round(2))

print("\n --- Matriz de Correlação ---")
print(df.corr().round(2))

plt.figure(figsize=(8,6))
plt.scatter(df['Horas_Estudo'], df['Nota_Prova'], color='#34AAF4', edgecolor='black')
plt.title('Relação entre horas e Nota da Prova')
plt.xlabel('Horas de Estudo')
plt.ylabel('Nota da Prova')
plt.grid(axis='both', linestyle='--', alpha=0.5)

plt.savefig("imagens/Dispersao.png")
plt.clf()

tabela_final = df.agg(['mean', 'median', 'var', 'std', 'min', 'max']).T

print('\n--- Tabela Resumo ---')
print(tabela_final.round(2))