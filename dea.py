"""
DEA (Data Envelopment Analysis) - Análise Envoltória de Dados
Implementação do modelo CCR usando PuLP para medir eficiência relativa dos estados
"""

import pandas as pd
import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, value


def calcular_dea_ccr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a eficiência relativa de cada estado usando modelo DEA modificado
    com ÊNFASE NO RESULTADO (baixa taxa de homicídios).
    
    Modelo com pesos fixos (Assurance Region DEA):
    - 75% do peso para o RESULTADO (segurança = baixa taxa de homicídios)
    - 25% do peso para o CUSTO (economia = baixo gasto per capita)
    
    Isso garante que estados com BAIXA TAXA de homicídios sejam priorizados,
    mesmo que gastem mais que outros estados.
    
    A eficiência varia de 0 a 1 (0% a 100%), onde:
    - 1.0 (100%) = Estado mais eficiente (benchmark)
    - < 1.0 = Estado menos eficiente
    
    Args:
        df: DataFrame com colunas 'sigla', 'estado', 'gasto_per_capita', 'taxa_mortes_100k'
    
    Returns:
        DataFrame com colunas adicionais: 'eficiencia_dea', 'eficiencia_percentual', 'benchmark'
    """
    
    # Prepara os dados
    df_dea = df.copy()
    
    # =========================================================================
    # COMPONENTE 1: RESULTADO (75% do peso)
    # Quanto MENOR a taxa de homicídios, MAIOR a pontuação
    # =========================================================================
    taxa_min = df_dea['taxa_mortes_100k'].min()
    df_dea['score_resultado'] = taxa_min / df_dea['taxa_mortes_100k']
    # SP (taxa 7.5) -> 7.5/7.5 = 1.0
    # MG (taxa 14.2) -> 7.5/14.2 = 0.53
    # BA (taxa 43.9) -> 7.5/43.9 = 0.17
    
    # =========================================================================
    # COMPONENTE 2: CUSTO (25% do peso)
    # Quanto MENOR o gasto per capita (para mesmo resultado), MAIOR a pontuação
    # =========================================================================
    gasto_min = df_dea['gasto_per_capita'].min()
    df_dea['score_custo'] = gasto_min / df_dea['gasto_per_capita']
    # SP (gasto 325) -> 325/325 = 1.0
    # MG (gasto 884) -> 325/884 = 0.37
    # BA (gasto 391) -> 325/391 = 0.83
    
    # =========================================================================
    # EFICIÊNCIA FINAL: 75% resultado + 25% custo
    # =========================================================================
    PESO_RESULTADO = 0.75  # Taxa de homicídios (quanto menor, melhor)
    PESO_CUSTO = 0.25      # Gasto per capita (quanto menor, melhor)
    
    df_dea['eficiencia_dea'] = (
        PESO_RESULTADO * df_dea['score_resultado'] + 
        PESO_CUSTO * df_dea['score_custo']
    )
    
    # Normaliza para que o máximo seja 100%
    max_ef = df_dea['eficiencia_dea'].max()
    df_dea['eficiencia_dea'] = df_dea['eficiencia_dea'] / max_ef
    
    df_dea['eficiencia_percentual'] = (df_dea['eficiencia_dea'] * 100).round(1)
    df_dea['benchmark'] = df_dea['eficiencia_dea'] >= 0.999
    
    # Remove colunas auxiliares
    df_dea = df_dea.drop(columns=['score_resultado', 'score_custo'])
    
    # Ordena por eficiência (maior = melhor)
    df_dea = df_dea.sort_values('eficiencia_dea', ascending=False)
    
    return df_dea


def calcular_dea_bcc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a eficiência DEA-BCC para cada estado.
    
    Modelo DEA-BCC (Banker-Charnes-Cooper) - permite retornos variáveis de escala.
    Mais flexível que o CCR, considera que estados pequenos/grandes podem ter
    eficiências diferentes devido à escala.
    
    Args:
        df: DataFrame com colunas 'sigla', 'estado', 'gasto_per_capita', 'taxa_mortes_100k'
    
    Returns:
        DataFrame com eficiência BCC
    """
    
    df_dea = df.copy()
    
    inputs = df_dea['gasto_per_capita'].values
    outputs = (1000 / df_dea['taxa_mortes_100k']).values
    
    n_dmus = len(df_dea)
    siglas = df_dea['sigla'].values
    
    inputs_norm = inputs / inputs.mean()
    outputs_norm = outputs / outputs.mean()
    
    eficiencias = []
    
    for k in range(n_dmus):
        prob = LpProblem(f"DEA_BCC_{siglas[k]}", LpMaximize)
        
        v = LpVariable("v", lowBound=0.0001)
        u = LpVariable("u", lowBound=0.0001)
        u0 = LpVariable("u0")  # Variável livre para retornos de escala
        
        prob += u * outputs_norm[k] + u0, "Eficiencia"
        prob += v * inputs_norm[k] == 1, "Normalizacao"
        
        for j in range(n_dmus):
            prob += (
                u * outputs_norm[j] + u0 - v * inputs_norm[j] <= 0,
                f"Eficiencia_DMU_{siglas[j]}"
            )
        
        prob.solve()
        
        if LpStatus[prob.status] == "Optimal":
            eficiencia = value(prob.objective)
            eficiencias.append(min(eficiencia, 1.0))
        else:
            eficiencias.append(0.0)
    
    df_dea['eficiencia_bcc'] = eficiencias
    df_dea['eficiencia_bcc_percentual'] = (df_dea['eficiencia_bcc'] * 100).round(1)
    
    return df_dea


def identificar_benchmarks(df_dea: pd.DataFrame) -> list:
    """
    Identifica os estados que estão na fronteira de eficiência (benchmarks).
    
    Args:
        df_dea: DataFrame com coluna 'eficiencia_dea'
    
    Returns:
        Lista de siglas dos estados benchmark
    """
    benchmarks = df_dea[df_dea['eficiencia_dea'] >= 0.999]['sigla'].tolist()
    return benchmarks


def calcular_metas(df_dea: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula as metas de melhoria para estados ineficientes.
    
    Para cada estado ineficiente, calcula:
    - Redução necessária no input (gasto) mantendo mesmo output
    - Ou aumento necessário no output (segurança) mantendo mesmo input
    
    Args:
        df_dea: DataFrame com eficiência DEA calculada
    
    Returns:
        DataFrame com metas de melhoria
    """
    
    df_metas = df_dea.copy()
    
    # Meta de gasto = gasto_atual * eficiencia (redução proporcional)
    df_metas['meta_gasto'] = (df_metas['gasto_per_capita'] * df_metas['eficiencia_dea']).round(0)
    
    # Redução necessária no gasto
    df_metas['reducao_gasto'] = df_metas['gasto_per_capita'] - df_metas['meta_gasto']
    df_metas['reducao_gasto_pct'] = ((1 - df_metas['eficiencia_dea']) * 100).round(1)
    
    # Meta de taxa = taxa_atual * eficiencia (ou seja, taxa menor)
    df_metas['meta_taxa'] = (df_metas['taxa_mortes_100k'] * df_metas['eficiencia_dea']).round(1)
    
    return df_metas


def resumo_dea(df_dea: pd.DataFrame) -> dict:
    """
    Gera um resumo estatístico da análise DEA.
    
    Args:
        df_dea: DataFrame com eficiência DEA calculada
    
    Returns:
        Dicionário com estatísticas resumidas
    """
    
    return {
        'eficiencia_media': df_dea['eficiencia_dea'].mean(),
        'eficiencia_mediana': df_dea['eficiencia_dea'].median(),
        'eficiencia_min': df_dea['eficiencia_dea'].min(),
        'eficiencia_max': df_dea['eficiencia_dea'].max(),
        'n_eficientes': (df_dea['eficiencia_dea'] >= 0.999).sum(),
        'n_ineficientes': (df_dea['eficiencia_dea'] < 0.999).sum(),
        'benchmarks': identificar_benchmarks(df_dea),
        'estado_mais_eficiente': df_dea.iloc[0]['estado'] if len(df_dea) > 0 else None,
        'estado_menos_eficiente': df_dea.iloc[-1]['estado'] if len(df_dea) > 0 else None,
    }
