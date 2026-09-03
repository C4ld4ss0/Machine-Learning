import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = {
  'ID_Aluno': [1, 2, 3, 4, 5, 6], 
    'Curso': ['Engenharia', 'Direito', 'Engenharia', 'Computação', 'Direito', 'Computação'], 
    'Nivel_Satisfacao': ['Alto', 'Medio', 'Baixo', 'Alto', 'Medio', 'Alto'], 
    'Horas_Estudo': [12.5, 8.0, 4.5, 15.0, 10.0, 14.0], 
    'Nota_Final': [8.5, 6.0, 4.0, 9.5, 7.0, 8.0]
}
df = pd.DataFrame(data)

print("--- Dataset Original ---")
print(df)

new_data = pd.DataFrame({
    'ID_Aluno': [7, 8, 9],
    'Curso': ['Computação', 'Engenharia', 'Direito'],
    'Nivel_Satisfacao': ['Alto', 'Baixo', 'Medio'],
    'Horas_Estudo': [16.0, 5.0, 9.0],
    'Nota_Final': [9.8, 4.5, 6.5]
})
df = pd.concat([df, new_data], ignore_index = True)

print("\n --- Dataset após 3 novas instâncias ---")
print(df)

df_encoded = pd.get_dummies(df, columns=['Curso'], dtype=int)

print("\n ---Dataset com One-Hot Encoding ---")
print(df_encoded)

print("\n--- Resumo Estatístico (Horas_Estudo) ---")
resumo = df['Horas_Estudo'].describe()
variancia = df['Horas_Estudo'].var()
moda = df['Horas_Estudo'].mode()[0]
print(resumo)
print(f"Variância: {variancia:.2f}\nModa:{moda}")

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.histplot(df['Horas_Estudo'], kde=True, color='#34AAF4')
plt.title('Histograma - Horas de Estudo')

plt.subplot(1,2,2)
sns.boxplot(y=df['Horas_Estudo'], color='#34AAF4')
plt.title('Boxplot - Horas de Estudo')
plt.savefig('Analise_Univariada.png')
plt.clf()

simetrico = np.random.normal(loc=50, scale=5, size=1000)
enviesado = np.random.exponential(scale=2, size=1000)
print(f"\nSimétrico -> Média: {np.mean(simetrico):.2f} | Mediana: {np.median(simetrico):.2f}")
print(f"\nEnvesiado -> Média: {np.mean(enviesado):.2f} | Mediana: {np.median(enviesado):.2f}")

Q1 = df['Horas_Estudo'].quantile(0.25)
Q3 = df['Horas_Estudo'].quantile(0.75)
IQR= Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR
outliers = df[(df['Horas_Estudo'] < limite_inferior) | (df['Horas_Estudo'] > limite_superior)]
print("\n--- Outliers detectados pelo IQR ---")
print(outliers)

df['Frequencia'] = [80, 60, 40, 90, 75, 85, 95, 45, 70]
df['Idade'] = [20, 22, 19, 21, 23, 20, 24, 19, 21]

colunas_numericas = ['Horas_Estudo', 'Nota_Final', 'Frequencia', 'Idade']

sns.pairplot(df[colunas_numericas])
plt.savefig('Matriz_Dispersao.png')
plt.clf()

print("\n--- Matriz de Correlação ---")
print(df[colunas_numericas].corr(method='pearson').round(2))
