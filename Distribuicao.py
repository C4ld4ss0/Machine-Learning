import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

#Leitor de CSV - categoria da idade
df = pd.read_csv("census.csv", sep=",") # leitor separa por vírgula
idades = df['age'] # Limita a só ler a idade - AGE im inglês

# Histograma
tabela_Idades = pd.DataFrame({'idade': idades}) # Faz o histograma
plt.figure(figsize=(8,6)) # Decide o tamanho do gráfico

#Formatação Histograma
plt.hist(idades, bins = 10, edgecolor = 'black') # variável que contém os dados, intervalo dos dados "de 10 em 10 anos", cor da borda
plt.title('Distribuição de idade') # título do Histograma
plt.xlabel('idade') # Nome do eixo X
plt.ylabel('Frequência') # nome do eixo Y
plt.tight_layout() # Diminui o farmato da imagem
plt.savefig("Histograma") # Salva o histograma como imagem
plt.clf() # Limpa a plt para a próxima imagem

# Box plot
plt.figure(figsize=(8,6))

# Estilizando os outliers
estilo_outliers = dict(markerfacecolor='red', marker='o', alpha=0.4, markeredgecolor = 'none')

# Formatação Box plot
plt.boxplot(idades, # Onde vai tirar os dados
            patch_artist=True, #Avisa que vai pintar
            tick_labels=['Idades do censo'], #Muda e põe um parâmetro
            boxprops=dict(facecolor='#87CEFA', color='#4682B4'), #Pinta a caixa de azul, borda escura
            medianprops=dict(color='red', linewidth=2), #Linha da mediana vermelha, tamanho da linha 2
            flierprops=estilo_outliers) # muda os Outliers conforme o formato que fizemos linha 2

# Nomeação dos parâmetros
plt.title("Distribuição das idades (Boxplot)", fontsize = 15) #Título, tamanho da fonte
plt.ylabel("Idade em anos", fontsize = 12) # 2ndo título vertical, tamanho da fonte

#Display - como é mostrado
plt.grid(axis='y', linestyle='--', alpha=0.5) #axis Y vai fazer com que durante a divisão do grid as linhas sejam '--' e com 0.5 de opacidade

#Salvamento e mostragem o Boxplot
plt.savefig("Box_Plot.png")
plt.clf() # Apaga as alterações para liberar e fazer o próximo

# Obliquidade
obliquidade = skew(idades)
obliResul = 'Não tem informação'

# Comparadores da Obliquidade
if obliquidade <= 0.3:
    obliResul = 'Assimetria Positiva'
elif obliquidade >= -0.3:
    obliResul = 'Assimetria Negativa'
else:
    obliResul = 'Simétrica'

# Curtose
curtose = kurtosis(idades)
curResul = "Não temos dados"

# Comparadores da Curtose
if curtose >= 0.1:
    curResul = 'Mais concentrada (Leptocúrtica)'
elif curtose <= -0.1:
    curResul = 'Mais achatada (Platicúrtica)'
else:
    curResul = 'Próxima da normal (Mesocúrtica)'

# Formatação para mostragem na tela
obliText = f"Obliquidade é = {obliquidade: .4f} | {obliResul}"
curText = f"Curtose é = {curtose: .4f} | {curResul}"
largPixel = 80 # Largura do texto

# Mostragem da Obliquidade
print(f"""
{'-' * largPixel}
{('\033[1m' + obliText + '\033[0m').center(largPixel + 8)}                   
{'-' * largPixel}
""")
 #\033[1m ativa o negrito, põe o texto, \033[0 desativa o negrito 
 # .center centraliza e (usa o tamanho estipulado para centralizar)

print(f"""
{'*' * largPixel}
{('\033[1m'+ curText + '\033[0m').center(largPixel + 8)}
{'*' * largPixel}
""")