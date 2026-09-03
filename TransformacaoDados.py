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
df = pd.DataFrama(data)

print("--- Dataset Original ---")
print(df)

newData = pd.dataFrame({
  'ID_Aluno': [7, 8, 9],
    'Curso': ['Computação', 'Engenharia', 'Direito'],
    'Nivel_Satisfacao': ['Alto', 'Baixo', 'Medio'],
    'Horas_Estudo': [16.0, 5.0, 9.0],
    'Nota_Final': [9.8, 4.5, 6.5]
})
df = pd.concat ([df, newData], ignoreIndex = True)

print("\n --- Dataset após 3 novas instâncias ---")
print(df)

dfEncoded = pd.getDummies(df, columns=['curso'], dtype=int)

print("\n ---Dataset com One-Hot Encoding ---")
print(df_encoded)
