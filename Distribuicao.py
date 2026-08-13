
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

df = pd.read_csv("census.csv", sep=",")
idades = df['age']

idades = pd.DataFrame({


})



obliquidade = skew(idades)

curtose = kurtosis(idades)

print("Obliquidade é: ", obliquidade)
print("Curtose é: ", curtose)