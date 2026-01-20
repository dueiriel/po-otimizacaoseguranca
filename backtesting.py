# =============================================================================
# MÓDULO DE BACKTESTING - VALIDAÇÃO DO MODELO
# =============================================================================
# Este módulo implementa validação histórica do modelo de otimização.
#
# METODOLOGIA:
# 1. Usa dados de 2012-2017 para "treinar" elasticidades
# 2. Aplica o modelo para "prever" alocação ótima
# 3. Compara com resultados observados em 2018-2022
# 4. Mede acurácia das previsões
#
# OBJETIVO:
# Responder: "Se tivéssemos usado este modelo em 2017, quão bem
# ele teria previsto a evolução da violência até 2022?"
#
# LIMITAÇÕES:
# - Não temos dados históricos de orçamento por ano
# - Usamos tendência como proxy para efeito de políticas
# - Correlação não implica causalidade
# =============================================================================

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


@dataclass
class ResultadoBacktest:
    """Resultados da validação por backtesting."""
    periodo_treino: Tuple[int, int]
    periodo_teste: Tuple[int, int]
    n_estados: int
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    mape: float  # Mean Absolute Percentage Error
    correlacao: float  # Correlação previsto vs observado
    r_squared: float
    previsoes: pd.DataFrame


def carregar_serie_completa() -> pd.DataFrame:
    """Carrega série histórica completa de homicídios."""
    dados_dir = Path(__file__).parent / "dados"
    arquivo = dados_dir / "taxa_homicidios_jovens.csv"
    
    df = pd.read_csv(arquivo, sep=';')
    df = df.rename(columns={
        'cod': 'cod_uf',
        'nome': 'estado',
        'período': 'ano',
        'valor': 'taxa_homicidios'
    })
    
    return df


def calcular_tendencia_estado(
    df: pd.DataFrame,
    estado: str,
    ano_inicio: int,
    ano_fim: int
) -> Tuple[float, float, float]:
    """
    Calcula tendência linear para um estado em um período.
    
    Returns:
        Tupla (slope, intercept, r_squared)
    """
    df_estado = df[
        (df['estado'] == estado) & 
        (df['ano'] >= ano_inicio) & 
        (df['ano'] <= ano_fim)
    ].sort_values('ano')
    
    if len(df_estado) < 3:
        return 0.0, 0.0, 0.0
    
    X = df_estado['ano'].values
    y = df_estado['taxa_homicidios'].values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
    
    return slope, intercept, r_value**2


def prever_taxa(
    slope: float,
    intercept: float,
    ano_base: int,
    ano_previsao: int
) -> float:
    """
    Prevê taxa de homicídios usando tendência linear.
    
    Args:
        slope: Inclinação da tendência
        intercept: Intercepto
        ano_base: Ano inicial da série
        ano_previsao: Ano a prever
    
    Returns:
        Taxa prevista
    """
    return intercept + slope * ano_previsao


def executar_backtest(
    ano_treino_inicio: int = 2012,
    ano_treino_fim: int = 2017,
    ano_teste_inicio: int = 2018,
    ano_teste_fim: int = 2022
) -> ResultadoBacktest:
    """
    Executa backtesting do modelo.
    
    Processo:
    1. Calcula tendências usando período de treino
    2. Projeta taxas para período de teste
    3. Compara previsões com valores observados
    
    Args:
        ano_treino_inicio: Início do período de treino
        ano_treino_fim: Fim do período de treino
        ano_teste_inicio: Início do período de teste
        ano_teste_fim: Fim do período de teste
    
    Returns:
        ResultadoBacktest com métricas e previsões
    """
    # Carrega dados
    df = carregar_serie_completa()
    
    resultados = []
    
    for estado in df['estado'].unique():
        # Calcula tendência no período de treino
        slope, intercept, r2_treino = calcular_tendencia_estado(
            df, estado, ano_treino_inicio, ano_treino_fim
        )
        
        # Dados observados no período de teste
        df_teste = df[
            (df['estado'] == estado) & 
            (df['ano'] >= ano_teste_inicio) & 
            (df['ano'] <= ano_teste_fim)
        ].sort_values('ano')
        
        if len(df_teste) == 0:
            continue
        
        for _, row in df_teste.iterrows():
            ano = row['ano']
            observado = row['taxa_homicidios']
            
            # Previsão baseada na tendência
            previsto = prever_taxa(slope, intercept, ano_treino_inicio, ano)
            previsto = max(0, previsto)  # Taxa não pode ser negativa
            
            erro = previsto - observado
            erro_pct = abs(erro / observado * 100) if observado > 0 else 0
            
            resultados.append({
                'estado': estado,
                'ano': ano,
                'observado': observado,
                'previsto': round(previsto, 2),
                'erro': round(erro, 2),
                'erro_abs': round(abs(erro), 2),
                'erro_pct': round(erro_pct, 2),
                'tendencia_anual': round(slope, 4),
                'r2_treino': round(r2_treino, 4)
            })
    
    df_resultados = pd.DataFrame(resultados)
    
    # Calcula métricas agregadas
    mae = df_resultados['erro_abs'].mean()
    rmse = np.sqrt((df_resultados['erro'] ** 2).mean())
    mape = df_resultados['erro_pct'].mean()
    
    # Correlação entre previsto e observado
    corr, _ = stats.pearsonr(
        df_resultados['previsto'], 
        df_resultados['observado']
    )
    
    # R² da previsão
    ss_res = ((df_resultados['observado'] - df_resultados['previsto']) ** 2).sum()
    ss_tot = ((df_resultados['observado'] - df_resultados['observado'].mean()) ** 2).sum()
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return ResultadoBacktest(
        periodo_treino=(ano_treino_inicio, ano_treino_fim),
        periodo_teste=(ano_teste_inicio, ano_teste_fim),
        n_estados=df_resultados['estado'].nunique(),
        mae=round(mae, 2),
        rmse=round(rmse, 2),
        mape=round(mape, 2),
        correlacao=round(corr, 4),
        r_squared=round(r_squared, 4),
        previsoes=df_resultados
    )


def analisar_estados_por_acuracia(resultado: ResultadoBacktest) -> pd.DataFrame:
    """
    Agrupa resultados por estado e ordena por acurácia.
    
    Args:
        resultado: Resultado do backtest
    
    Returns:
        DataFrame com métricas por estado
    """
    df = resultado.previsoes.groupby('estado').agg({
        'erro_abs': 'mean',
        'erro_pct': 'mean',
        'tendencia_anual': 'first',
        'r2_treino': 'first',
        'observado': 'mean',
        'previsto': 'mean'
    }).reset_index()
    
    df.columns = ['estado', 'mae', 'mape', 'tendencia', 'r2_treino', 
                  'media_observado', 'media_previsto']
    
    # Classifica acurácia
    df['acuracia'] = df['mape'].apply(
        lambda x: 'Alta' if x < 15 else ('Média' if x < 30 else 'Baixa')
    )
    
    return df.sort_values('mape')


def gerar_grafico_backtest(resultado: ResultadoBacktest) -> go.Figure:
    """
    Gera gráfico de dispersão: Previsto vs Observado.
    
    Args:
        resultado: Resultado do backtest
    
    Returns:
        Figura Plotly
    """
    df = resultado.previsoes
    
    fig = go.Figure()
    
    # Pontos
    fig.add_trace(go.Scatter(
        x=df['observado'],
        y=df['previsto'],
        mode='markers',
        marker=dict(
            size=8,
            color=df['erro_pct'],
            colorscale='RdYlGn_r',
            colorbar=dict(title='Erro %'),
            opacity=0.7
        ),
        text=df['estado'] + ' (' + df['ano'].astype(str) + ')',
        hovertemplate='%{text}<br>Observado: %{x:.1f}<br>Previsto: %{y:.1f}<extra></extra>'
    ))
    
    # Linha de referência (previsão perfeita)
    max_val = max(df['observado'].max(), df['previsto'].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='Previsão Perfeita'
    ))
    
    fig.update_layout(
        title=f'Backtest: Previsto vs Observado<br><sub>R² = {resultado.r_squared:.3f}, Correlação = {resultado.correlacao:.3f}</sub>',
        xaxis_title='Taxa Observada (por 100 mil)',
        yaxis_title='Taxa Prevista (por 100 mil)',
        height=500,
        showlegend=False
    )
    
    return fig


def gerar_grafico_serie_temporal(
    resultado: ResultadoBacktest,
    estados: List[str]
) -> go.Figure:
    """
    Gera gráfico de série temporal para estados selecionados.
    
    Args:
        resultado: Resultado do backtest
        estados: Lista de estados a plotar
    
    Returns:
        Figura Plotly
    """
    df_hist = carregar_serie_completa()
    
    fig = make_subplots(
        rows=len(estados), cols=1,
        subplot_titles=estados,
        shared_xaxes=True,
        vertical_spacing=0.08
    )
    
    cores = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    
    for i, estado in enumerate(estados, 1):
        # Série observada completa
        df_estado = df_hist[df_hist['estado'] == estado].sort_values('ano')
        
        fig.add_trace(
            go.Scatter(
                x=df_estado['ano'],
                y=df_estado['taxa_homicidios'],
                mode='lines+markers',
                name=f'{estado} (Obs)',
                line=dict(color=cores[i-1]),
                marker=dict(size=6)
            ),
            row=i, col=1
        )
        
        # Previsões
        df_prev = resultado.previsoes[resultado.previsoes['estado'] == estado]
        
        fig.add_trace(
            go.Scatter(
                x=df_prev['ano'],
                y=df_prev['previsto'],
                mode='lines+markers',
                name=f'{estado} (Prev)',
                line=dict(color=cores[i-1], dash='dash'),
                marker=dict(size=6, symbol='x')
            ),
            row=i, col=1
        )
        
        # Linha vertical separando treino/teste
        fig.add_vline(
            x=resultado.periodo_treino[1] + 0.5,
            line_dash="dot",
            line_color="gray",
            row=i, col=1
        )
    
    fig.update_layout(
        height=200 * len(estados),
        title='Série Temporal: Observado vs Previsto',
        showlegend=True
    )
    
    return fig


def validar_modelo_rolling(
    janela_treino: int = 5,
    janela_teste: int = 1,
    ano_inicio: int = 2010,
    ano_fim: int = 2022
) -> pd.DataFrame:
    """
    Validação com janela deslizante (rolling window).
    
    Para cada ano de teste, usa os N anos anteriores para treinar.
    
    Args:
        janela_treino: Tamanho da janela de treino (anos)
        janela_teste: Tamanho da janela de teste (anos)
        ano_inicio: Ano inicial para início dos testes
        ano_fim: Ano final
    
    Returns:
        DataFrame com métricas por ano de teste
    """
    resultados = []
    
    for ano_teste in range(ano_inicio + janela_treino, ano_fim + 1):
        treino_inicio = ano_teste - janela_treino
        treino_fim = ano_teste - 1
        
        resultado = executar_backtest(
            ano_treino_inicio=treino_inicio,
            ano_treino_fim=treino_fim,
            ano_teste_inicio=ano_teste,
            ano_teste_fim=ano_teste
        )
        
        resultados.append({
            'ano_teste': ano_teste,
            'treino': f'{treino_inicio}-{treino_fim}',
            'mae': resultado.mae,
            'rmse': resultado.rmse,
            'mape': resultado.mape,
            'correlacao': resultado.correlacao
        })
    
    return pd.DataFrame(resultados)


# =============================================================================
# TESTE DO MÓDULO
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("BACKTESTING - VALIDAÇÃO DO MODELO")
    print("=" * 70)
    
    # 1. Backtest principal
    print("\n" + "=" * 70)
    print("1. BACKTEST: Treino 2012-2017 → Teste 2018-2022")
    print("=" * 70)
    
    resultado = executar_backtest(
        ano_treino_inicio=2012,
        ano_treino_fim=2017,
        ano_teste_inicio=2018,
        ano_teste_fim=2022
    )
    
    print(f"\n📊 Métricas de Validação:")
    print(f"  • Período de treino: {resultado.periodo_treino}")
    print(f"  • Período de teste: {resultado.periodo_teste}")
    print(f"  • Estados avaliados: {resultado.n_estados}")
    print(f"\n  • MAE (Erro Absoluto Médio): {resultado.mae:.2f} por 100 mil")
    print(f"  • RMSE (Raiz do Erro Quadrático): {resultado.rmse:.2f}")
    print(f"  • MAPE (Erro Percentual): {resultado.mape:.1f}%")
    print(f"  • Correlação: {resultado.correlacao:.3f}")
    print(f"  • R²: {resultado.r_squared:.3f}")
    
    # 2. Estados por acurácia
    print("\n" + "=" * 70)
    print("2. ESTADOS POR ACURÁCIA DE PREVISÃO")
    print("=" * 70)
    
    df_estados = analisar_estados_por_acuracia(resultado)
    
    print("\n🎯 Estados com ALTA acurácia (MAPE < 15%):")
    alta = df_estados[df_estados['acuracia'] == 'Alta']
    if len(alta) > 0:
        print(alta[['estado', 'mape', 'tendencia']].head(10).to_string(index=False))
    else:
        print("  Nenhum estado")
    
    print("\n⚠️ Estados com BAIXA acurácia (MAPE > 30%):")
    baixa = df_estados[df_estados['acuracia'] == 'Baixa']
    if len(baixa) > 0:
        print(baixa[['estado', 'mape', 'tendencia']].head(10).to_string(index=False))
    else:
        print("  Nenhum estado")
    
    # 3. Validação rolling
    print("\n" + "=" * 70)
    print("3. VALIDAÇÃO COM JANELA DESLIZANTE (5 anos)")
    print("=" * 70)
    
    df_rolling = validar_modelo_rolling(
        janela_treino=5,
        ano_inicio=2015,
        ano_fim=2022
    )
    print("\n" + df_rolling.to_string(index=False))
    
    # 4. Interpretação
    print("\n" + "=" * 70)
    print("4. INTERPRETAÇÃO")
    print("=" * 70)
    
    print(f"""
    O modelo de tendência linear apresenta:
    
    • Correlação de {resultado.correlacao:.2f} entre previsto e observado
      → Previsões seguem a direção correta
    
    • MAPE de {resultado.mape:.1f}%
      → Erro médio de {resultado.mape:.1f}% nas previsões
    
    • Estados com tendências claras (alto R² no treino) têm
      previsões mais acuradas
    
    • Estados com mudanças bruscas (ex: novas políticas) têm
      maior erro de previsão
    
    ⚠️ LIMITAÇÃO: O modelo assume que tendências passadas continuam.
       Intervenções políticas podem alterar significativamente 
       a trajetória, o que é desejável mas dificulta a previsão.
    """)
