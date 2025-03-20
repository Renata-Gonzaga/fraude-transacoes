import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Definir caminho do dataset
dataset_path = "creditcard.csv"

# Verificar se o arquivo existe antes de carregar
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"O arquivo {dataset_path} não foi encontrado. Certifique-se de que está no diretório correto.")

# Carregar o dataset
print(f"Carregando dataset de: {dataset_path}")
df = pd.read_csv(dataset_path)

# Verificar as primeiras linhas
display_rows = 5
print(df.head(display_rows))

# Dividir os dados entre features e target
X = df.drop(columns=['Class'])  # Todas as colunas menos a de fraude
y = df['Class']  # Coluna que indica fraude (1) ou não (0)

# Separar os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Fazer previsões
y_pred = modelo.predict(X_test)

# Avaliar o modelo
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))
