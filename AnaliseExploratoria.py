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
firstQuartil = idades.quantile(0.25)
secondQuartil = idades.quantile(0.50)
thirdQuartil = idades.quantile(0.75)
percentilFive = idades.quantile(0.05)
PercentilTen = idades.quantile(0.10)
PercentilNinety = idades.quantile(0.90)
variancia = idades.var()
desvioPadrao = idades.std()



print ("--Resultados--")
print (f"A Média aritmética é: {mediaAritmetica:.2f}")
print (f"A Média Harmonica é: {mediaHarmonica:.2f}")
print (f"A Média Geométrica é: {mediaGeometrica:.2f}")
print (f"A média Quadrática é: {mediaQuadratica:.2f}")
print ("A Mediana é: ", mediana)
print ("A Moda é: ", moda)
print ("O primeiro Quartil", firstQuartil)
print ("O segundo Quartil", secondQuartil)
print ("O terceriro Quartil", thirdQuartil)
print ("O primeiro Quartil", percentilFive)
print ("O primeiro Quartil", PercentilTen)
print ("O primeiro Quartil", PercentilNinety)
print(f"A Variância é: {variancia:.2f}")
print(f"O Desvio Padrão é: {desvioPadrao:.2f}")


print("\n" + "="*40)
print(" 📊 RELATÓRIO ESTATÍSTICO - IDADES")
print("="*40)

print("\n▶ MEDIDAS DE POSIÇÃO")
print(f"  a) Média Aritmética : {mediaAritmetica:.2f}")
print(f"  b) Média Harmônica  : {mediaHarmonica:.2f}")
print(f"  c) Média Geométrica : {mediaGeometrica:.2f}")
print(f"  d) Média Quadrática : {mediaQuadratica:.2f}")
print(f"  e) Mediana (Q2)     : {mediana:.2f}")
print(f"  f) Moda             : {moda}")

print("\n▶ QUARTIS E PERCENTIS")
print(f"  g) Quartis          : Q1 = {firstQuartil:.2f} | Q2 = {secondQuartil:.2f} | Q3 = {thirdQuartil:.2f}")
print(f"  h) Percentis        : P5 = {percentilFive:.2f} | P10 = {PercentilTen:.2f} | P90 = {PercentilNinety:.2f}")

print("\n▶ MEDIDAS DE DISPERSÃO")
print(f"  i) Variância        : {variancia:.2f}")
print(f"  j) Desvio Padrão    : {desvioPadrao:.2f}")

print("\n" + "="*40)
# print(df.head())
# print(idades.head())