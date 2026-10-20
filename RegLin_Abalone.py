import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
colunas = [
    'Sex', 'Length', 'Diameter', 'Height',
    'Whole_weight', 'Shucked_weight',
    'Viscera_weight', 'Shell_weight', 'Rings'
]
dados = pd.read_csv(url, names=colunas)

mapeamento_sexo = {'M': 0, 'F': 1, 'I': 2}
dados['Sex'] = dados['Sex'].map(mapeamento_sexo)

X = dados.drop('Rings', axis=1).values.astype(float)
y = dados['Rings'].values.astype(float)

X = X.astype(np.float64)
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_normalized = (X - X_mean) / X_std

X_b = np.c_[np.ones((X_normalized.shape[0], 1)), X_normalized]

def gradient_descent(X, y, learning_rate=0.001, epochs=10000):
    m = len(y)
    weights = np.zeros(X.shape[1])
    
    for epoch in range(epochs):
        y_pred = X.dot(weights)
        
        error = y_pred - y
        
        gradients = (2/m) * X.T.dot(error)
        
        weights -= learning_rate * gradients
        
    return weights

learning_rate = 0.001
best_weights = gradient_descent(X_b, y, learning_rate=learning_rate)

y_pred_final = X_b.dot(best_weights)

def calculate_metrics(y_true, y_pred):
    n = len(y_true)

    mse = np.sum((y_true - y_pred) ** 2) / n

    rmse = np.sqrt(mse)

    mae = np.sum(np.abs(y_true - y_pred)) / n

    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    
    return mse, rmse, mae, r2

mse, rmse, mae, r2 = calculate_metrics(y, y_pred_final)

print("=== Resultados da Avaliação ===")
print(f"MSE (Erro Médio Quadrático): {mse:.4f}")
print(f"RMSE (Raiz do Erro Médio Quadrático): {rmse:.4f}")
print(f"MAE (Erro Absoluto Médio): {mae:.4f}")
print(f"R2 (Coeficiente de Determinação): {r2:.4f}")

print("\n=== Teste: Gradiente Descendente com learning_rate = 0.0001 ===")
pesos_0001 = gradient_descent(X_b, y, learning_rate=0.0001, epochs=10000)
y_pred_0001 = X_b.dot(pesos_0001)

mse_2, rmse_2, mae_2, r2_2 = calculate_metrics(y, y_pred_0001)
print(f"MSE: {mse_2:.4f} | RMSE: {rmse_2:.4f} | MAE: {mae_2:.4f} | R2: {r2_2:.4f}")
print("Nota: Ao diminuir a taxa de aprendizado, o modelo dá passos menores. Se o número de épocas for o mesmo, o erro pode ficar maior pois ele não teve tempo de chegar no ponto mínimo.")

print("\n=== Resultados da Avaliação (Scikit-Learn) ===")

modelo_sklearn = LinearRegression()
modelo_sklearn.fit(X_normalized, y)

y_pred_sklearn = modelo_sklearn.predict(X_normalized)

mse_sk = mean_squared_error(y, y_pred_sklearn)
rmse_sk = np.sqrt(mse_sk)
mae_sk = mean_absolute_error(y, y_pred_sklearn)
r2_sk = r2_score(y, y_pred_sklearn)

print(f"MSE (Erro Médio Quadrático): {mse_sk:.4f}")
print(f"RMSE (Raiz do Erro Médio Quadrático): {rmse_sk:.4f}")
print(f"MAE (Erro Absoluto Médio): {mae_sk:.4f}")
print(f"R2 (Coeficiente de Determinação): {r2_sk:.4f}")
