import pandas as pd
import argparse
from itertools import combinations

def carregar_dados():
    df = pd.read_csv('compras.csv')
    compras_por_cliente = df.groupby('cliente')['produto'].apply(set).to_dict()
    return compras_por_cliente

def indice_jaccard(set1, set2):
    if not set1 or not set2:
        return 0
    intersecao = len(set1.intersection(set2))
    uniao = len(set1.union(set2))
    return intersecao / uniao if uniao > 0 else 0

def calcular_similaridades(compras_por_cliente):
    clientes = list(compras_por_cliente.keys())
    similaridades = []
    
    for cliente1, cliente2 in combinations(clientes, 2):
        similaridade = indice_jaccard(
            compras_por_cliente[cliente1], 
            compras_por_cliente[cliente2]
        )
        similaridades.append({
            'cliente1': cliente1,
            'cliente2': cliente2,
            'similaridade': similaridade,
            'produtos_cliente1': compras_por_cliente[cliente1],
            'produtos_cliente2': compras_por_cliente[cliente2]
        })
    
    return sorted(similaridades, key=lambda x: x['similaridade'], reverse=True)

def recomendar_produtos(cliente_alvo, compras_por_cliente, top_n=3):
    recomendacoes = {}
    
    for outro_cliente, produtos in compras_por_cliente.items():
        if outro_cliente != cliente_alvo:
            similaridade = indice_jaccard(
                compras_por_cliente[cliente_alvo],
                produtos
            )
            
            produtos_novos = produtos - compras_por_cliente[cliente_alvo]
            for produto in produtos_novos:
                if produto not in recomendacoes:
                    recomendacoes[produto] = {'frequencia': 0, 'similaridade_total': 0}
                recomendacoes[produto]['frequencia'] += 1
                recomendacoes[produto]['similaridade_total'] += similaridade
    
    for produto, dados in recomendacoes.items():
        dados['score'] = dados['frequencia'] * dados['similaridade_total']
    
    return sorted(recomendacoes.items(), key=lambda x: x[1]['score'], reverse=True)[:top_n]

def main():
    parser = argparse.ArgumentParser(description='Sistema de Recomendação por Similaridade de Jaccard')
    parser.add_argument('--detalhe', nargs=2, help='Calcular similaridade entre dois clientes específicos')
    
    args = parser.parse_args()
    
    compras_por_cliente = carregar_dados()
    
    print("PRODUTOS POR CLIENTE:")
    for cliente, produtos in compras_por_cliente.items():
        print(f"Cliente {cliente}: {', '.join(produtos)}")
    
    print("\n" + "="*50)
    print("3 PARES DE CLIENTES MAIS SIMILARES:")
    
    similaridades = calcular_similaridades(compras_por_cliente)
    
    for i, sim in enumerate(similaridades[:3], 1):
        print(f"\n{i}. {sim['cliente1']} e {sim['cliente2']}")
        print(f"   Similaridade Jaccard: {sim['similaridade']:.3f}")
        print(f"   Produtos em comum: {', '.join(sim['produtos_cliente1'].intersection(sim['produtos_cliente2']))}")
    
    if args.detalhe:
        cliente1, cliente2 = args.detalhe
        if cliente1 in compras_por_cliente and cliente2 in compras_por_cliente:
            print(f"\n" + "="*50)
            print(f"DETALHE: {cliente1} vs {cliente2}")
            similaridade = indice_jaccard(
                compras_por_cliente[cliente1],
                compras_por_cliente[cliente2]
            )
            print(f"Similaridade Jaccard: {similaridade:.3f}")
            print(f"Produtos {cliente1}: {', '.join(compras_por_cliente[cliente1])}")
            print(f"Produtos {cliente2}: {', '.join(compras_por_cliente[cliente2])}")
            print(f"Produtos em comum: {', '.join(compras_por_cliente[cliente1].intersection(compras_por_cliente[cliente2]))}")
            
            print(f"\nRECOMENDAÇÕES PARA {cliente1}:")
            recomendacoes = recomendar_produtos(cliente1, compras_por_cliente)
            for produto, dados in recomendacoes:
                print(f"- {produto} (score: {dados['score']:.3f})")
        else:
            print("Cliente(s) não encontrado(s)!")

if __name__ == "__main__":
    main()