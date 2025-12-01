import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('vendas_tratadas.csv')

print("Colunas disponíveis no arquivo:")
print(df.columns.tolist())
print("\nPrimeiras 5 linhas:")
print(df.head())
print("\n" + "="*60)

print("1. RESUMO ESTATÍSTICO:")
print(df.describe())
print("\n" + "="*60)

print("\nINSIGHTS DO RESUMO ESTATÍSTICO:")
print(f"1. A quantidade média por transação é de {df['quantidade'].mean():.1f} unidades")
print(f"2. Há grande variação nos preços, com máximo de R$ {df['preco'].max():.2f} e mínimo de R$ {df['preco'].min():.2f}")
print(f"3. O valor médio por transação é de R$ {df['valor_total'].mean():.2f}")
print("\n" + "="*60)

print("\n2. GRÁFICOS:")

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.hist(df['preco'], bins=15, color='lightblue', edgecolor='black')
plt.title('Histograma do Preço dos Produtos')
plt.xlabel('Preço (R$)')
plt.ylabel('Frequência')

plt.subplot(2, 2, 2)
plt.boxplot(df['quantidade'])
plt.title('Boxplot das Quantidades')
plt.ylabel('Quantidade')

plt.subplot(2, 2, 3)
produtos_vendidos = df.groupby('produto')['quantidade'].sum().sort_values(ascending=False)
top_produtos = produtos_vendidos.head(10)

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.bar(range(len(top_produtos)), top_produtos.values, color='lightblue')
ax1.set_xlabel('Produtos')
ax1.set_ylabel('Quantidade Vendida', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
plt.xticks(range(len(top_produtos)), top_produtos.index, rotation=45)

plt.title('Top 10 Produtos Mais Vendidos')
plt.tight_layout()
plt.show()

print("\n3. MEDIDAS ESTATÍSTICAS DO PREÇO:")
media = df['preco'].mean()
mediana = df['preco'].median()
moda = df['preco'].mode()[0]
desvio_padrao = df['preco'].std()

print(f"Média: R$ {media:.2f}")
print(f"Mediana: R$ {mediana:.2f}")
print(f"Moda: R$ {moda:.2f}")
print(f"Desvio Padrão: R$ {desvio_padrao:.2f}")

print("\n4. ANÁLISE DA DISPERSÃO:")
coeficiente_variacao = (desvio_padrao / media) * 100
print(f"Coeficiente de Variação: {coeficiente_variacao:.1f}%")

if coeficiente_variacao > 30:
    print("DISPERSÃO: ALTA - Os preços variam significativamente entre produtos")
else:
    print("DISPERSÃO: BAIXA - Os preços são relativamente homogêneos")

print("\n5. DIFERENÇA ENTRE MÉDIA E MEDIANA:")
diferenca = abs(media - mediana)
print(f"Diferença: R$ {diferenca:.2f}")

if diferenca > 2:
    print("INDICA: Presença de outliers - alguns produtos têm preços muito diferentes da maioria")
    print("A média é puxada por valores extremos (produtos muito caros ou muito baratos)")
else:
    print("INDICA: Distribuição equilibrada - pouca influência de outliers")

print("\nTOP 5 PRODUTOS MAIS VENDIDOS:")
top_5 = produtos_vendidos.head()
for i, (produto, quantidade) in enumerate(top_5.items(), 1):
    print(f"{i}. {produto}: {quantidade} unidades")