import pandas as pd
import matplotlib.pyplot as plt
import nunpy as np
import scipy.stats import mode

# Dados informados pela própria professora
dados = {
 'Horas_Estudo': [2, 3, 4, 5, 6, 7, 8, 9, 10, 6],
 'Frequencia': [65, 70, 72, 78, 80, 85, 88, 92, 95, 82],
 'Nota_Prova': [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 7.2],
 'Nota_Trabalho': [5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.2, 7.8]
}

df = pd.DataFrame(dados)
print(dados.mean())


