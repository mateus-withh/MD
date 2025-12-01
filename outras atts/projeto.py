# PROJETO MINI MERCADO INTELIGENTE
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

print("🐛 MINI MERCADO INTELIGENTE - VALE DO RIBEIRA")
print("=" * 60)

# 1. ETL - EXTRACT, TRANSFORM, LOAD
# =================================

print("\n1. ETL - CARREGANDO E TRATANDO OS DADOS")

# Criando dataset simulado baseado no Vale do Ribeira
np.random.seed(42)
categorias_produtos = {
    'Hortifruti': ['Banana', 'Laranja', 'Mandioca', 'Tomate', 'Alface'],
    'Laticínios': ['Queijo', 'Leite', 'Manteiga', 'Iogurte', 'Requeijão'],
    'Padaria': ['Pão', 'Biscoito', 'Bolacha', 'Rosca', 'Bolo'],
    'Bebidas': ['Café', 'Refrigerante', 'Suco', 'Cerveja', 'Vinho'],
    'Açougue': ['Carne', 'Frango', 'Linguiça', 'Peixe', 'Ovos']
}

produtos_vale = []
for categoria, prods in categorias_produtos.items():
    produtos_vale.extend(prods)

# Gerando transações simuladas
transacoes = []
for i in range(1, 501):  # 500 transações
    cliente = f"C{str(i).zfill(3)}"
    num_produtos = np.random.randint(2, 8)  # 2 a 7 produtos por transação
    
    # Produtos baseados em combinações típicas do Vale do Ribeira
    if np.random.random() < 0.3:  # 30% compram produtos típicos
        produtos = np.random.choice(['Banana', 'Mandioca', 'Laranja', 'Peixe'], 
                                  size=min(3, num_produtos), replace=False)
        produtos = list(produtos)
    else:
        produtos = list(np.random.choice(produtos_vale, size=num_produtos, replace=False))
    
    for produto in produtos:
        quantidade = np.random.randint(1, 4)
        preco = np.random.uniform(1.5, 25.0)
        transacoes.append({
            'IDTransacao': i,
            'IDCliente': cliente,
            'Produto': produto,
            'Quantidade': quantidade,
            'Preco': round(preco, 2),
            'Data': f"2024-{np.random.randint(1,13):02d}-{np.random.randint(1,28):02d}"
        })

df = pd.DataFrame(transacoes)
df['ValorTotal'] = df['Quantidade'] * df['Preco']

print(f"Dataset criado: {len(df)} registros, {df['IDCliente'].nunique()} clientes, {df['Produto'].nunique()} produtos")
print("\nPrimeiras 5 transações:")
print(df.head())

# 2. ANÁLISE DESCRITIVA
# ====================

print("\n" + "=" * 60)
print("2. ANÁLISE DESCRITIVA")

# Estatísticas básicas
print("\nESTATÍSTICAS GERAIS:")
print(f"Período: {df['Data'].min()} a {df['Data'].max()}")
print(f"Faturamento total: R$ {df['ValorTotal'].sum():,.2f}")
print(f"Ticket médio por transação: R$ {df.groupby('IDTransacao')['ValorTotal'].sum().mean():.2f}")
print(f"Média de produtos por transação: {df.groupby('IDTransacao')['Produto'].count().mean():.1f}")

# Top produtos
top_produtos_qtd = df.groupby('Produto')['Quantidade'].sum().sort_values(ascending=False).head(10)
top_produtos_valor = df.groupby('Produto')['ValorTotal'].sum().sort_values(ascending=False).head(10)

print("\nTOP 10 PRODUTOS POR QUANTIDADE:")
for i, (produto, qtd) in enumerate(top_produtos_qtd.items(), 1):
    print(f"{i}. {produto}: {qtd} unidades")

print("\nTOP 10 PRODUTOS POR FATURAMENTO:")
for i, (produto, valor) in enumerate(top_produtos_valor.items(), 1):
    print(f"{i}. {produto}: R$ {valor:,.2f}")

# Visualizações
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
top_produtos_qtd.head(8).plot(kind='bar', color='green', alpha=0.7)
plt.title('Top 8 Produtos Mais Vendidos (Quantidade)')
plt.ylabel('Quantidade')
plt.xticks(rotation=45)

plt.subplot(2, 2, 2)
top_produtos_valor.head(8).plot(kind='bar', color='blue', alpha=0.7)
plt.title('Top 8 Produtos por Faturamento')
plt.ylabel('Valor (R$)')
plt.xticks(rotation=45)

plt.subplot(2, 2, 3)
df.groupby('IDTransacao')['ValorTotal'].sum().hist(bins=20, color='orange', alpha=0.7)
plt.title('Distribuição do Valor das Transações')
plt.xlabel('Valor (R$)')
plt.ylabel('Frequência')

plt.subplot(2, 2, 4)
df.groupby('IDTransacao')['Produto'].count().hist(bins=15, color='purple', alpha=0.7)
plt.title('Distribuição de Produtos por Transação')
plt.xlabel('Número de Produtos')
plt.ylabel('Frequência')

plt.tight_layout()
plt.show()

# 3. MINERAÇÃO - REGRAS DE ASSOCIAÇÃO (APRIORI)
# ============================================

print("\n" + "=" * 60)
print("3. MINERAÇÃO - REGRAS DE ASSOCIAÇÃO")

# Criando matriz binária para Apriori
transacoes_binarias = df.groupby(['IDTransacao', 'Produto'])['Quantidade'].sum().unstack(fill_value=0)
transacoes_binarias = (transacoes_binarias > 0).astype(int)

print(f"Matriz de transações: {transacoes_binarias.shape}")

# Aplicando algoritmo Apriori
frequent_itemsets = apriori(transacoes_binarias, min_support=0.03, use_colnames=True)
regras = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

print(f"\nRegras encontradas: {len(regras)}")

if len(regras) > 0:
    # Top regras por lift
    top_regras = regras.sort_values('lift', ascending=False).head(10)
    
    print("\nTOP 10 REGRAS DE ASSOCIAÇÃO (por Lift):")
    for i, (idx, regra) in enumerate(top_regras.iterrows(), 1):
        antecedente = list(regra['antecedents'])[0] if len(regra['antecedents']) == 1 else list(regra['antecedents'])
        consequente = list(regra['consequents'])[0] if len(regra['consequents']) == 1 else list(regra['consequents'])
        print(f"{i}. {antecedente} → {consequente}")
        print(f"   Suporte: {regra['support']:.3f} | Confiança: {regra['confidence']:.3f} | Lift: {regra['lift']:.3f}")

# 4. SIMILARIDADE - ÍNDICE DE JACCARD
# ===================================

print("\n" + "=" * 60)
print("4. SIMILARIDADE ENTRE CLIENTES")

def indice_jaccard(set1, set2):
    if not set1 or not set2:
        return 0
    intersecao = len(set1.intersection(set2))
    uniao = len(set1.union(set2))
    return intersecao / uniao if uniao > 0 else 0

# Agrupando produtos por cliente
compras_por_cliente = df.groupby('IDCliente')['Produto'].apply(set).to_dict()

# Calculando similaridades
clientes_ativos = list(compras_por_cliente.keys())[:20]  # Analisando apenas 20 clientes para exemplo
similaridades = []

for cliente1, cliente2 in combinations(clientes_ativos, 2):
    similaridade = indice_jaccard(
        compras_por_cliente[cliente1], 
        compras_por_cliente[cliente2]
    )
    similaridades.append({
        'cliente1': cliente1,
        'cliente2': cliente2,
        'similaridade': similaridade
    })

# Top similaridades
similaridades_df = pd.DataFrame(similaridades)
top_similaridades = similaridades_df.sort_values('similaridade', ascending=False).head(10)

print("\nTOP 10 PARES DE CLIENTES MAIS SIMILARES:")
for i, (idx, linha) in enumerate(top_similaridades.iterrows(), 1):
    print(f"{i}. {linha['cliente1']} & {linha['cliente2']}: {linha['similaridade']:.3f}")

# 5. SISTEMA DE RECOMENDAÇÃO
# =========================

print("\n" + "=" * 60)
print("5. SISTEMA DE RECOMENDAÇÃO")

def recomendar_produtos(cliente_alvo, compras_por_cliente, top_n=5):
    if cliente_alvo not in compras_por_cliente:
        return []
    
    recomendacoes = {}
    produtos_cliente = compras_por_cliente[cliente_alvo]
    
    for outro_cliente, produtos in compras_por_cliente.items():
        if outro_cliente != cliente_alvo:
            similaridade = indice_jaccard(produtos_cliente, produtos)
            
            if similaridade > 0.3:  # Considera apenas clientes com similaridade > 30%
                produtos_novos = produtos - produtos_cliente
                for produto in produtos_novos:
                    if produto not in recomendacoes:
                        recomendacoes[produto] = {'frequencia': 0, 'similaridade_total': 0}
                    recomendacoes[produto]['frequencia'] += 1
                    recomendacoes[produto]['similaridade_total'] += similaridade
    
    # Calculando score
    for produto, dados in recomendacoes.items():
        dados['score'] = dados['frequencia'] * dados['similaridade_total']
    
    return sorted(recomendacoes.items(), key=lambda x: x[1]['score'], reverse=True)[:top_n]

# Exemplo de recomendação
cliente_exemplo = clientes_ativos[0]
recomendacoes = recomendar_produtos(cliente_exemplo, compras_por_cliente)

print(f"\nRECOMENDAÇÕES PARA CLIENTE {cliente_exemplo}:")
print(f"Produtos atuais: {', '.join(compras_por_cliente[cliente_exemplo])}")

if recomendacoes:
    print("\nProdutos recomendados:")
    for produto, dados in recomendacoes:
        print(f"- {produto} (score: {dados['score']:.3f})")
else:
    print("Nenhuma recomendação encontrada.")

# 6. VISUALIZAÇÃO FINAL - PAINEL RESUMO
# =====================================

print("\n" + "=" * 60)
print("6. PAINEL FINAL - RESUMO DO MINI MERCADO INTELIGENTE")

plt.figure(figsize=(16, 12))

# Gráfico 1: Top produtos
plt.subplot(2, 3, 1)
top_produtos_qtd.head(5).plot(kind='bar', color='lightgreen')
plt.title('TOP 5 PRODUTOS MAIS VENDIDOS', fontweight='bold')
plt.ylabel('Quantidade')
plt.xticks(rotation=45)

# Gráfico 2: Regras de associação (se existirem)
plt.subplot(2, 3, 2)
if len(regras) > 0:
    top_5_regras = regras.nlargest(5, 'lift')
    regras_labels = [f"{list(r.antecedents)[0]}\n→ {list(r.consequents)[0]}" 
                    for _, r in top_5_regras.iterrows()]
    plt.bar(range(len(top_5_regras)), top_5_regras['lift'], color='lightblue')
    plt.title('TOP 5 REGRAS (por LIFT)', fontweight='bold')
    plt.ylabel('Lift')
    plt.xticks(range(len(top_5_regras)), regras_labels, rotation=45, ha='right')
else:
    plt.text(0.5, 0.5, 'Nenhuma regra\nencontrada', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('REGRAS DE ASSOCIAÇÃO', fontweight='bold')

# Gráfico 3: Similaridade entre clientes
plt.subplot(2, 3, 3)
top_10_sim = similaridades_df.nlargest(10, 'similaridade')
pares = [f"{row.cliente1}\n{row.cliente2}" for _, row in top_10_sim.iterrows()]
plt.bar(range(len(top_10_sim)), top_10_sim['similaridade'], color='lightcoral')
plt.title('TOP 10 CLIENTES SIMILARES', fontweight='bold')
plt.ylabel('Similaridade Jaccard')
plt.xticks(range(len(top_10_sim)), pares, rotation=45, ha='right', fontsize=8)

# Gráfico 4: Distribuição de valores
plt.subplot(2, 3, 4)
df['ValorTotal'].hist(bins=30, color='gold', alpha=0.7)
plt.title('DISTRIBUIÇÃO DOS VALORES', fontweight='bold')
plt.xlabel('Valor (R$)')
plt.ylabel('Frequência')

# Gráfico 5: Produtos por categoria
plt.subplot(2, 3, 5)
categorias_count = {}
for produto in df['Produto'].unique():
    for categoria, prods in categorias_produtos.items():
        if produto in prods:
            categorias_count[categoria] = categorias_count.get(categoria, 0) + 1
            break

plt.pie(categorias_count.values(), labels=categorias_count.keys(), autopct='%1.1f%%', 
        colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#ff99cc'])
plt.title('DISTRIBUIÇÃO POR CATEGORIA', fontweight='bold')

# Gráfico 6: Recomendações
plt.subplot(2, 3, 6)
if recomendacoes:
    produtos_rec = [rec[0] for rec in recomendacoes]
    scores_rec = [rec[1]['score'] for rec in recomendacoes]
    plt.barh(produtos_rec, scores_rec, color='lightseagreen')
    plt.title(f'RECOMENDAÇÕES - CLIENTE {cliente_exemplo}', fontweight='bold')
    plt.xlabel('Score de Recomendação')
else:
    plt.text(0.5, 0.5, 'Nenhuma\nrecomendação', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('RECOMENDAÇÕES', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("🎯 INSIGHTS E RECOMENDAÇÕES PARA O VALE DO RIBEIRA:")

print("\n📊 INSIGHTS ESTRATÉGICOS:")
print("1. PRODUTOS ESTRELA: Identificar os top 5 para promoções especiais")
print("2. COMBINAÇÕES: Usar regras de associação para disposição de produtos")
print("3. CLIENTES SIMILARES: Criar segmentos para marketing direcionado")
print("4. RECOMENDAÇÕES: Implementar sistema de sugestões no caixa")

print("\n💡 AÇÕES RECOMENDADAS:")
print("• Promoções cruzadas baseadas nas regras de associação")
print("• Layout de loja que favoreça combinações frequentes") 
print("• Programas de fidelidade segmentados por perfil de compra")
print("• Recomendações personalizadas no app e site")

print(f"\n✅ PROJETO CONCLUÍDO: {len(df)} transações analisadas")