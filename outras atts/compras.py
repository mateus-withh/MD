import pandas as pd

dados_compras = {
    'cliente': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'D', 'D', 'D', 'E', 'E', 'E', 'F', 'F', 'F'],
    'produto': ['Leite', 'Pão', 'Manteiga', 'Queijo', 'Leite', 'Presunto', 'Leite', 'Pão', 'Café', 'Queijo', 'Presunto', 'Vinho', 'Leite', 'Pão', 'Manteiga', 'Café', 'Açúcar', 'Biscoito']
}

df_compras = pd.DataFrame(dados_compras)
df_compras.to_csv('compras.csv', index=False)
print("Arquivo compras.csv criado com sucesso!")
print(df_compras)