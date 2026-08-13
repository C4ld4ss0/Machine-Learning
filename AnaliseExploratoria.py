import pandas as pd
import numpy as np
import statistics

df = pd.read_csv("census.csv", sep=",")
idades = df['age']
mediaAritmetica = idades.mean()
mediaHarmonica = statistics.harmonic_mean(idades)
mediaGeometrica = statistics.geometric_mean(idades)
mediaQuadratica = np.sqrt((idades**2).mean())
mediana = idades.median()
moda = idades.mode()[0]

print ("--Resultados--")
print (f"A Média aritmética é: {mediaAritmetica:.2f}")
print (f"A Média Harmonica é: {mediaHarmonica:.2f}")
print (f"A Média Geométrica é: {mediaGeometrica:.2f}")
print (f"A média Quadrática é: {mediaQuadratica:.2f}")
print ("A Mediana é: ", mediana)
print ("A Moda é: ", moda)

# print(df.head())
# print(idades.head())