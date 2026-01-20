# =============================================================================
# MÓDULO DE SIMULAÇÃO MONTE CARLO E ANÁLISE DE CENÁRIOS
# =============================================================================
# Este módulo implementa simulação estocástica para análise de incerteza
# nos parâmetros do modelo de otimização.
#
# MOTIVAÇÃO:
# Os parâmetros do modelo (elasticidade, taxa de crime) têm incerteza.
# Monte Carlo permite quantificar como essa incerteza se propaga para
# a solução ótima.
#
# METODOLOGIA:
# 1. Define distribuições de probabilidade para parâmetros incertos
# 2. Amostra N conjuntos de parâmetros
# 3. Resolve o modelo para cada conjunto
# 4. Analisa distribuição dos resultados
#
# REFERÊNCIAS:
# - Rubinstein, R. Y. "Simulation and the Monte Carlo Method"
# - Metropolis, N. et al. (1953). "Equation of State Calculations"
# =============================================================================

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ProcessPoolExecutor
import warnings

from otimizacao import otimizar_alocacao, ResultadoOtimizacao


@dataclass
class ResultadoMonteCarlo:
    """Armazena resultados da simulação Monte Carlo."""
    n_simulacoes: int
    n_sucesso: int
    media_reducao: float
    desvio_padrao_reducao: float
    intervalo_confianca_95: Tuple[float, float]
    percentis: Dict[int, float]
    distribuicao_reducao: List[float]
    distribuicao_custo: List[float]


def simular_parametros(
    df_dados: pd.DataFrame,
    incerteza_elasticidade: float = 0.20,
    incerteza_taxa: float = 0.10,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Gera uma amostra de parâmetros com incerteza.
    
    Aplica perturbação estocástica aos parâmetros do modelo,
    seguindo distribuição normal truncada.
    
    Args:
        df_dados: DataFrame original
        incerteza_elasticidade: Coeficiente de variação da elasticidade (0.20 = 20%)
        incerteza_taxa: Coeficiente de variação da taxa de crime
        seed: Semente para reprodutibilidade
    
    Returns:
        DataFrame com parâmetros perturbados
    """
    if seed is not None:
        np.random.seed(seed)
    
    df = df_dados.copy()
    
    # Perturba elasticidade (distribuição normal, truncada em [0.01, 0.30])
    for idx in df.index:
        if pd.notna(df.loc[idx, 'elasticidade']):
            elast_base = df.loc[idx, 'elasticidade']
            elast_std = elast_base * incerteza_elasticidade
            
            nova_elast = np.random.normal(elast_base, elast_std)
            nova_elast = np.clip(nova_elast, 0.01, 0.30)
            
            df.loc[idx, 'elasticidade'] = nova_elast
    
    # Perturba taxa de mortes (menor incerteza, dado que é observado)
    for idx in df.index:
        if pd.notna(df.loc[idx, 'mortes_violentas']):
            mortes_base = df.loc[idx, 'mortes_violentas']
            mortes_std = mortes_base * incerteza_taxa
            
            novas_mortes = np.random.normal(mortes_base, mortes_std)
            novas_mortes = max(1, int(novas_mortes))
            
            df.loc[idx, 'mortes_violentas'] = novas_mortes
    
    return df


def executar_monte_carlo(
    df_dados: pd.DataFrame,
    orcamento: float,
    n_simulacoes: int = 1000,
    incerteza_elasticidade: float = 0.20,
    incerteza_taxa: float = 0.10,
    seed: Optional[int] = 42,
    verbose: bool = True
) -> ResultadoMonteCarlo:
    """
    Executa simulação Monte Carlo para análise de incerteza.
    
    Para cada simulação:
    1. Perturba parâmetros estocasticamente
    2. Resolve o problema de otimização
    3. Armazena resultado
    
    Args:
        df_dados: DataFrame original
        orcamento: Orçamento disponível
        n_simulacoes: Número de simulações
        incerteza_elasticidade: CV da elasticidade
        incerteza_taxa: CV da taxa de crime
        seed: Semente base para reprodutibilidade
        verbose: Exibir progresso
    
    Returns:
        ResultadoMonteCarlo com estatísticas
    """
    if seed is not None:
        np.random.seed(seed)
    
    reducoes = []
    custos = []
    n_sucesso = 0
    
    if verbose:
        print(f"🎲 Executando {n_simulacoes} simulações Monte Carlo...")
    
    for i in range(n_simulacoes):
        # Gera parâmetros perturbados
        df_sim = simular_parametros(
            df_dados, 
            incerteza_elasticidade,
            incerteza_taxa,
            seed=seed + i if seed else None
        )
        
        # Resolve otimização
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resultado = otimizar_alocacao(df_sim, orcamento, verbose=False)
        
        if resultado.status == 'Optimal':
            reducoes.append(resultado.reducao_crimes)
            custo = resultado.orcamento_usado / resultado.reducao_crimes if resultado.reducao_crimes > 0 else np.nan
            custos.append(custo)
            n_sucesso += 1
        
        if verbose and (i + 1) % 100 == 0:
            print(f"  Progresso: {i + 1}/{n_simulacoes} ({(i+1)/n_simulacoes*100:.0f}%)")
    
    # Calcula estatísticas
    reducoes = np.array(reducoes)
    custos = np.array([c for c in custos if not np.isnan(c)])
    
    media = np.mean(reducoes)
    std = np.std(reducoes)
    
    # Intervalo de confiança 95%
    ic_inferior = np.percentile(reducoes, 2.5)
    ic_superior = np.percentile(reducoes, 97.5)
    
    # Percentis
    percentis = {
        5: np.percentile(reducoes, 5),
        25: np.percentile(reducoes, 25),
        50: np.percentile(reducoes, 50),
        75: np.percentile(reducoes, 75),
        95: np.percentile(reducoes, 95)
    }
    
    return ResultadoMonteCarlo(
        n_simulacoes=n_simulacoes,
        n_sucesso=n_sucesso,
        media_reducao=round(media, 1),
        desvio_padrao_reducao=round(std, 1),
        intervalo_confianca_95=(round(ic_inferior, 1), round(ic_superior, 1)),
        percentis={k: round(v, 1) for k, v in percentis.items()},
        distribuicao_reducao=reducoes.tolist(),
        distribuicao_custo=custos.tolist()
    )


def gerar_cenarios_elasticidade(
    df_dados: pd.DataFrame,
    fator_otimista: float = 1.25,
    fator_pessimista: float = 0.75
) -> Dict[str, pd.DataFrame]:
    """
    Gera cenários de elasticidade: otimista, base e pessimista.
    
    - Otimista: Elasticidade 25% maior (políticas mais efetivas)
    - Base: Elasticidade original
    - Pessimista: Elasticidade 25% menor (políticas menos efetivas)
    
    Args:
        df_dados: DataFrame original
        fator_otimista: Multiplicador para cenário otimista
        fator_pessimista: Multiplicador para cenário pessimista
    
    Returns:
        Dicionário com DataFrames para cada cenário
    """
    df_otimista = df_dados.copy()
    df_pessimista = df_dados.copy()
    
    df_otimista['elasticidade'] = df_otimista['elasticidade'] * fator_otimista
    df_pessimista['elasticidade'] = df_pessimista['elasticidade'] * fator_pessimista
    
    # Limita elasticidade a valores razoáveis
    df_otimista['elasticidade'] = df_otimista['elasticidade'].clip(upper=0.25)
    df_pessimista['elasticidade'] = df_pessimista['elasticidade'].clip(lower=0.03)
    
    return {
        'pessimista': df_pessimista,
        'base': df_dados.copy(),
        'otimista': df_otimista
    }


def comparar_cenarios(
    cenarios: Dict[str, pd.DataFrame],
    orcamento: float
) -> pd.DataFrame:
    """
    Compara resultados de otimização entre cenários.
    
    Args:
        cenarios: Dicionário {nome: DataFrame}
        orcamento: Orçamento disponível
    
    Returns:
        DataFrame comparativo
    """
    resultados = []
    
    for nome, df in cenarios.items():
        resultado = otimizar_alocacao(df, orcamento)
        
        if resultado.status == 'Optimal':
            resultados.append({
                'cenario': nome.capitalize(),
                'reducao_crimes': resultado.reducao_crimes,
                'reducao_pct': resultado.reducao_percentual,
                'orcamento_usado': resultado.orcamento_usado,
                'custo_por_vida': round(
                    resultado.orcamento_usado / resultado.reducao_crimes, 2
                ) if resultado.reducao_crimes > 0 else np.nan
            })
    
    return pd.DataFrame(resultados)


def gerar_grafico_monte_carlo(resultado: ResultadoMonteCarlo) -> go.Figure:
    """
    Gera histograma da distribuição de resultados do Monte Carlo.
    
    Args:
        resultado: Resultado da simulação Monte Carlo
    
    Returns:
        Figura Plotly
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'Distribuição: Vidas Salvas',
            'Distribuição: Custo por Vida'
        )
    )
    
    # Histograma de vidas salvas
    fig.add_trace(
        go.Histogram(
            x=resultado.distribuicao_reducao,
            nbinsx=30,
            name='Vidas Salvas',
            marker_color='#3498db',
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Linha vertical para média
    fig.add_vline(
        x=resultado.media_reducao, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Média: {resultado.media_reducao:.0f}",
        row=1, col=1
    )
    
    # IC 95%
    ic = resultado.intervalo_confianca_95
    fig.add_vrect(
        x0=ic[0], x1=ic[1],
        fillcolor="green", opacity=0.1,
        layer="below", line_width=0,
        row=1, col=1
    )
    
    # Histograma de custo
    fig.add_trace(
        go.Histogram(
            x=resultado.distribuicao_custo,
            nbinsx=30,
            name='Custo/Vida',
            marker_color='#e74c3c',
            opacity=0.7
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=f'Simulação Monte Carlo ({resultado.n_simulacoes} simulações)',
        showlegend=False,
        height=400
    )
    
    fig.update_xaxes(title_text="Vidas Salvas", row=1, col=1)
    fig.update_xaxes(title_text="R$ milhões / vida", row=1, col=2)
    fig.update_yaxes(title_text="Frequência", row=1, col=1)
    fig.update_yaxes(title_text="Frequência", row=1, col=2)
    
    return fig


def gerar_grafico_cenarios(df_cenarios: pd.DataFrame) -> go.Figure:
    """
    Gera gráfico comparativo de cenários.
    
    Args:
        df_cenarios: DataFrame com comparação de cenários
    
    Returns:
        Figura Plotly
    """
    cores = {
        'Pessimista': '#e74c3c',
        'Base': '#3498db',
        'Otimista': '#2ecc71'
    }
    
    fig = go.Figure()
    
    for _, row in df_cenarios.iterrows():
        fig.add_trace(go.Bar(
            x=[row['cenario']],
            y=[row['reducao_crimes']],
            name=row['cenario'],
            marker_color=cores.get(row['cenario'], '#95a5a6'),
            text=[f"{row['reducao_crimes']:,.0f}"],
            textposition='outside'
        ))
    
    fig.update_layout(
        title='Comparação de Cenários: Vidas Salvas',
        xaxis_title='Cenário',
        yaxis_title='Vidas Salvas',
        showlegend=False,
        height=400
    )
    
    return fig


# =============================================================================
# TESTE DO MÓDULO
# =============================================================================
if __name__ == "__main__":
    from dados import carregar_dados_consolidados
    
    print("=" * 70)
    print("SIMULAÇÃO MONTE CARLO E ANÁLISE DE CENÁRIOS")
    print("=" * 70)
    
    # Carrega dados
    df = carregar_dados_consolidados()
    orcamento = 5000  # R$ 5 bilhões
    
    # 1. Análise de Cenários (Otimista/Base/Pessimista)
    print("\n" + "=" * 70)
    print("1. ANÁLISE DE CENÁRIOS")
    print("=" * 70)
    
    cenarios = gerar_cenarios_elasticidade(df)
    df_cenarios = comparar_cenarios(cenarios, orcamento)
    
    print("\nComparação de cenários com orçamento de R$ 5 bilhões:")
    print(df_cenarios.to_string(index=False))
    
    # 2. Simulação Monte Carlo
    print("\n" + "=" * 70)
    print("2. SIMULAÇÃO MONTE CARLO")
    print("=" * 70)
    
    resultado_mc = executar_monte_carlo(
        df, 
        orcamento,
        n_simulacoes=500,  # Reduzido para teste rápido
        incerteza_elasticidade=0.20,
        incerteza_taxa=0.10,
        verbose=True
    )
    
    print(f"\n📊 Resultados ({resultado_mc.n_sucesso}/{resultado_mc.n_simulacoes} simulações bem-sucedidas):")
    print(f"  Média de vidas salvas: {resultado_mc.media_reducao:,.0f}")
    print(f"  Desvio padrão: {resultado_mc.desvio_padrao_reducao:,.0f}")
    print(f"  IC 95%: [{resultado_mc.intervalo_confianca_95[0]:,.0f}, {resultado_mc.intervalo_confianca_95[1]:,.0f}]")
    print(f"\n  Percentis:")
    for p, v in resultado_mc.percentis.items():
        print(f"    P{p}: {v:,.0f} vidas")
    
    # 3. Interpretação
    print("\n" + "=" * 70)
    print("3. INTERPRETAÇÃO")
    print("=" * 70)
    
    print(f"""
    Com um investimento de R$ {orcamento/1000:.0f} bilhões e considerando
    a incerteza nos parâmetros (elasticidade ±20%, taxa ±10%):
    
    • Esperamos salvar aproximadamente {resultado_mc.media_reducao:,.0f} vidas
    • Com 95% de confiança, esse número estará entre 
      {resultado_mc.intervalo_confianca_95[0]:,.0f} e {resultado_mc.intervalo_confianca_95[1]:,.0f} vidas
    • No cenário pessimista (elasticidade -25%), salvaríamos {df_cenarios[df_cenarios['cenario'] == 'Pessimista']['reducao_crimes'].values[0]:,.0f} vidas
    • No cenário otimista (elasticidade +25%), salvaríamos {df_cenarios[df_cenarios['cenario'] == 'Otimista']['reducao_crimes'].values[0]:,.0f} vidas
    """)
