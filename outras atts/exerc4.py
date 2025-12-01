import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('comportamento_de_compra.csv')
print("Primeiras 10 linhas do dataset:")
print(df.head(10))
print("\n" + "="*50 + "\n")

df_clean = df.drop('IDTransacao', axis=1)

df_clean = df_clean.rename(columns={
    'IDCliente': 'cliente',
    'NomeProduto': 'produto',
    'Quantidade': 'quantidade',
    'Preco': 'preco',
    'Data': 'data'
})

print("DataFrame após limpeza e renomeação:")
print(df_clean.head())
print("\n" + "="*50 + "\n")

df_clean['data'] = pd.to_datetime(df_clean['data'])
print("Info após conversão de data:")
print(df_clean.info())
print("\n" + "="*50 + "\n")

df_clean['valor_total'] = df_clean['preco'] * df_clean['quantidade']
total_vendas = df_clean['valor_total'].sum()
print(f"Total de vendas: R$ {total_vendas:,.2f}")

produto_mais_vendido = df_clean.groupby('produto')['quantidade'].sum().idxmax()
quantidade_mais_vendido = df_clean.groupby('produto')['quantidade'].sum().max()
print(f"Produto mais vendido: {produto_mais_vendido} ({quantidade_mais_vendido} unidades)")

media_compras_cliente = df_clean.groupby('cliente')['valor_total'].sum().mean()
print(f"Média de compras por cliente: R$ {media_compras_cliente:,.2f}")
print("\n" + "="*50 + "\n")

top_5_produtos = df_clean.groupby('produto')['quantidade'].sum().sort_values(ascending=False).head(5)

plt.figure(figsize=(10, 6))
top_5_produtos.plot(kind='bar', color='skyblue')
plt.title('Top 5 Produtos Mais Vendidos (por quantidade)')
plt.xlabel('Produto')
plt.ylabel('Quantidade Vendida')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Top 5 produtos mais vendidos:")
for i, (produto, quantidade) in enumerate(top_5_produtos.items(), 1):
    print(f"{i}. {produto}: {quantidade} unidades")

print("\n" + "="*50 + "\n")

df_final = df_clean.copy()
df_final.to_csv('vendas_tratadas.csv', index=False)
print("DataFrame salvo como 'vendas_tratadas.csv'")

print("\nInformações do DataFrame final:")
print(f"Número de linhas: {len(df_final)}")
print(f"Número de colunas: {len(df_final.columns)}")
print(f"Colunas: {list(df_final.columns)}")
print(f"Período das vendas: {df_final['data'].min().strftime('%d/%m/%Y')} a {df_final['data'].max().strftime('%d/%m/%Y')}")