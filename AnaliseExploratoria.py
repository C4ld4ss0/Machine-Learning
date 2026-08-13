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

# print(df.head())
# print(idades.head())