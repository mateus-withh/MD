import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv('comportamento_de_compra.csv')

print("Primeiras linhas do dataset:")
print(df.head())
print("\n" + "="*60)

print("1. CRIANDO MATRIZ BINÁRIA DE TRANSAÇÕES")

transacoes = df.groupby(['IDTransacao', 'NomeProduto'])['Quantidade'].sum().unstack(fill_value=0)
transacoes_binarias = (transacoes > 0).astype(int)

print(f"Dimensões da matriz: {transacoes_binarias.shape}")
print(f"Número de transações: {len(transacoes_binarias)}")
print(f"Número de produtos: {len(transacoes_binarias.columns)}")

print("\nFrequência dos produtos mais comuns:")
freq_produtos = transacoes_binarias.sum().sort_values(ascending=False)
print(freq_produtos.head(10))
print("\n" + "="*60)

print("2. APLICANDO ALGORITMO APRIORI")
print("Parâmetros: min_support=0.02, metric='confidence', min_threshold=0.5")

frequent_itemsets = apriori(transacoes_binarias, min_support=0.02, use_colnames=True)
print(f"\nNúmero de itemsets frequentes encontrados: {len(frequent_itemsets)}")

if len(frequent_itemsets) > 0:
    regras = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    print(f"Número de regras encontradas: {len(regras)}")
    
    if len(regras) > 0:
        print("\n3. TOP 10 REGRAS COM MAIOR LIFT")
        regras_ordenadas = regras.sort_values('lift', ascending=False)
        top_10_lift = regras_ordenadas.head(10)

        for i, (idx, regra) in enumerate(top_10_lift.iterrows(), 1):
            print(f"\n{i}. LIFT: {regra['lift']:.3f}")
            print(f"   REGRA: {list(regra['antecedents'])} → {list(regra['consequents'])}")
            print(f"   Suporte: {regra['support']:.3f} | Confiança: {regra['confidence']:.3f}")

        print("\n" + "="*60)
        print("4. INTERPRETAÇÃO DE 2 REGRAS")

        if len(regras) >= 2:
            regra1 = regras_ordenadas.iloc[0]
            regra2 = regras_ordenadas.iloc[1]
            
            print(f"\nREGRAS 1:")
            print(f"Antecedente: {list(regra1['antecedents'])}")
            print(f"Consequente: {list(regra1['consequents'])}")
            print(f"Suporte: {regra1['support']:.3f} - Frequência da ocorrência conjunta")
            print(f"Confiança: {regra1['confidence']:.3f} - Probabilidade do consequente dado o antecedente")
            print(f"Lift: {regra1['lift']:.3f}")
            
            if regra1['lift'] > 1:
                print("ASSOCIAÇÃO: FORTE - Os produtos estão positivamente correlacionados")
            else:
                print("ASSOCIAÇÃO: FRACA - Os produtos não estão correlacionados")
            
            print(f"\nREGRAS 2:")
            print(f"Antecedente: {list(regra2['antecedents'])}")
            print(f"Consequente: {list(regra2['consequents'])}")
            print(f"Suporte: {regra2['support']:.3f} - Frequência da ocorrência conjunta")
            print(f"Confiança: {regra2['confidence']:.3f} - Probabilidade do consequente dado o antecedente")
            print(f"Lift: {regra2['lift']:.3f}")
            
            if regra2['lift'] > 1:
                print("ASSOCIAÇÃO: FORTE - Os produtos estão positivamente correlacionados")
            else:
                print("ASSOCIAÇÃO: FRACA - Os produtos não estão correlacionados")

        print("\n" + "="*60)
        print("5. TESTANDO COM SUPORTE = 0.01")

        frequent_itemsets_001 = apriori(transacoes_binarias, min_support=0.01, use_colnames=True)
        print(f"\nNúmero de itemsets frequentes com suporte 0.01: {len(frequent_itemsets_001)}")

        regras_001 = association_rules(frequent_itemsets_001, metric="confidence", min_threshold=0.5)
        print(f"Número de regras com suporte 0.01: {len(regras_001)}")

        print(f"\nCOMPARAÇÃO:")
        print(f"Suporte 0.02: {len(regras)} regras encontradas")
        print(f"Suporte 0.01: {len(regras_001)} regras encontradas")
        print(f"Diferença: {len(regras_001) - len(regras)} regras adicionais")

        if len(regras_001) > len(regras):
            print("\nREDUZIR O SUPORTE PERMITE ENCONTRAR:")
            print("- Regras envolvendo produtos menos frequentes")
            print("- Associações mais específicas entre nichos de produtos")
            print("- Padrões que ocorrem com menor frequência mas são fortes")

    else:
        print("Nenhuma regra encontrada com os parâmetros atuais.")
        print("Tentando com confiança mínima mais baixa...")
        regras = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
        print(f"Regras encontradas com confiança 0.3: {len(regras)}")
        
else:
    print("Nenhum itemset frequente encontrado com suporte 0.02.")
    print("Tentando com suporte mais baixo...")
    frequent_itemsets = apriori(transacoes_binarias, min_support=0.01, use_colnames=True)
    print(f"Itemsets encontrados com suporte 0.01: {len(frequent_itemsets)}")

print("\n" + "="*60)
print("RESUMO DAS MÉTRICAS:")
print("Suporte: Frequência de ocorrência do itemset no dataset")
print("Confiança: Probabilidade do consequente dado o antecedente") 
print("Lift: Mede a força da associação (>1 = positiva, <1 = negativa)")