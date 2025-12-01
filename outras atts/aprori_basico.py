import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import numpy as np

# Dados da simulação
data = {
    'ProdutoID': [
        1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
        6, 6, 7, 7, 8, 8, 9, 9, 10, 10
    ],
    'Produto': [
        'Laptop', 'Mouse',
        'Laptop', 'Teclado',
        'Monitor', 'Mouse',
        'Teclado', 'Mouse',
        'Notebook', 'Notebook',
        'Teclado', 'Teclado',
        'Mouse', 'Notebook',
        'Mouse', 'Teclado',
        'Mouse', 'Monitor',
        'Teclado', 'Mouse'
    ]
}

df = pd.DataFrame(data)

# Transformar em matriz binária (cesta)
cesta = df.pivot_table(
    index='ProdutoID',
    columns='Produto',
    aggfunc=lambda x: 1,
    fill_value=0
)

# Apriori com suporte reduzido
frequencia_item = apriori(cesta, min_support=0.03, use_colnames=True)

# Regras com confidence mínimo acima de 0.7
regras = association_rules(frequencia_item, metric="confidence", min_threshold=0.7)

# Exportar para CSV
regras.to_csv("regras_mercado.csv", index=False)
print("Arquivo CSV exportado: regras_mercado.csv")

# Mostrar regras filtradas
if not regras.empty:
    print(regras[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
else:
    print("Nenhuma regra encontrada com confidence > 0.7.")
