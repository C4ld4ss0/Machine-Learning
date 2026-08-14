
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

df = pd.read_csv("census.csv", sep=",")
idades = df['age']
# Histograma
tabela_Idades = pd.DataFrame({'idade': idades})
plt.figure(figsize=(10,7))
plt.hist(idades, bins = 10, edgecolor = 'black')
plt.title('Distribuição de idade')
plt.xlabel('idade')
plt.ylabel('Frequência')
plt.tight_layout()
plt.show()

# Box plot
plt.boxplot(idades)
plt.title("Bloxplot das idades")
plt.ylabel("Idade em anos")


obliquidade = skew(idades)


curtose = kurtosis(idades)

print("Obliquidade é: ", obliquidade)
print("Curtose é: ", curtose)
plt.clf()
plt.show()
