# =============================================================================
# MÓDULO DE ANÁLISE DE SENSIBILIDADE
# =============================================================================
# Este módulo implementa técnicas de análise de sensibilidade para o modelo
# de otimização, essenciais para entender a robustez da solução.
#
# ANÁLISES IMPLEMENTADAS:
# 1. Shadow Prices (Preços-Sombra): Valor marginal de relaxar cada restrição
# 2. Análise de Intervalo: Como a solução muda com variação de parâmetros
# 3. Gráfico de Tornado: Identifica parâmetros mais influentes
# 4. Análise What-If: Cenários alternativos de orçamento
#
# REFERÊNCIAS:
# - Winston, W. L. "Operations Research" - Cap. 6: Sensitivity Analysis
# - Hillier & Lieberman "Introduction to OR" - Cap. 7: Duality Theory
# =============================================================================

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from otimizacao import otimizar_alocacao, ResultadoOtimizacao


@dataclass
class ResultadoSensibilidade:
    """Armazena resultados da análise de sensibilidade."""
    parametro: str
    valor_base: float
    valor_variado: float
    variacao_pct: float
    fo_base: float
    fo_variada: float
    impacto_fo_pct: float
    alocacao_mudou: bool


def analisar_sensibilidade_orcamento(
    df_dados: pd.DataFrame,
    orcamento_base: float,
    variacoes_pct: List[float] = [-20, -10, -5, 5, 10, 20, 50, 100]
) -> pd.DataFrame:
    """
    Analisa como a solução ótima muda com variações no orçamento disponível.
    
    Esta é a análise de sensibilidade mais importante: quanto mais orçamento
    disponibilizamos, mais vidas salvamos - mas a que taxa?
    
    Args:
        df_dados: DataFrame com dados consolidados
        orcamento_base: Orçamento base para comparação (R$ milhões)
        variacoes_pct: Lista de variações percentuais a testar
    
    Returns:
        DataFrame com resultados para cada cenário de orçamento
    """
    resultados = []
    
    # Calcula solução base
    resultado_base = otimizar_alocacao(df_dados, orcamento_base)
    
    # Testa cada variação
    for var_pct in variacoes_pct:
        orcamento_var = orcamento_base * (1 + var_pct / 100)
        
        resultado = otimizar_alocacao(df_dados, orcamento_var)
        
        if resultado.status == 'Optimal':
            # Calcula métricas comparativas
            delta_reducao = resultado.reducao_crimes - resultado_base.reducao_crimes
            eficiencia_marginal = (
                delta_reducao / (orcamento_var - orcamento_base) 
                if var_pct != 0 else 0
            )
            
            resultados.append({
                'variacao_pct': var_pct,
                'orcamento_milhoes': orcamento_var,
                'orcamento_usado': resultado.orcamento_usado,
                'reducao_crimes': resultado.reducao_crimes,
                'reducao_pct': resultado.reducao_percentual,
                'delta_reducao': delta_reducao,
                'eficiencia_marginal': round(eficiencia_marginal, 4),
                'custo_por_vida': round(
                    resultado.orcamento_usado / resultado.reducao_crimes 
                    if resultado.reducao_crimes > 0 else 0, 2
                ),
                'status': resultado.status
            })
    
    return pd.DataFrame(resultados)


def analisar_sensibilidade_elasticidade(
    df_dados: pd.DataFrame,
    orcamento: float,
    estado_alvo: str,
    variacoes_pct: List[float] = [-50, -25, -10, 10, 25, 50]
) -> pd.DataFrame:
    """
    Analisa impacto de variações na elasticidade de um estado específico.
    
    Útil para entender: "E se a eficiência do investimento neste estado
    for maior/menor do que estimamos?"
    
    Args:
        df_dados: DataFrame com dados consolidados
        orcamento: Orçamento disponível (R$ milhões)
        estado_alvo: Sigla do estado para variar elasticidade
        variacoes_pct: Variações percentuais na elasticidade
    
    Returns:
        DataFrame com resultados para cada cenário
    """
    resultados = []
    
    # Elasticidade original
    elast_original = df_dados.loc[
        df_dados['sigla'] == estado_alvo, 'elasticidade'
    ].values[0]
    
    # Resultado base
    resultado_base = otimizar_alocacao(df_dados, orcamento)
    invest_base = resultado_base.alocacao.loc[
        resultado_base.alocacao['sigla'] == estado_alvo, 'investimento_milhoes'
    ].values[0]
    
    for var_pct in variacoes_pct:
        # Cria cópia e varia elasticidade
        df_var = df_dados.copy()
        nova_elast = elast_original * (1 + var_pct / 100)
        df_var.loc[df_var['sigla'] == estado_alvo, 'elasticidade'] = nova_elast
        
        resultado = otimizar_alocacao(df_var, orcamento)
        
        if resultado.status == 'Optimal':
            invest_novo = resultado.alocacao.loc[
                resultado.alocacao['sigla'] == estado_alvo, 'investimento_milhoes'
            ].values[0]
            
            resultados.append({
                'estado': estado_alvo,
                'variacao_elasticidade_pct': var_pct,
                'elasticidade_original': elast_original,
                'elasticidade_nova': nova_elast,
                'investimento_base': invest_base,
                'investimento_novo': invest_novo,
                'delta_investimento': invest_novo - invest_base,
                'reducao_crimes_total': resultado.reducao_crimes,
                'delta_reducao': resultado.reducao_crimes - resultado_base.reducao_crimes
            })
    
    return pd.DataFrame(resultados)


def calcular_shadow_prices(
    df_dados: pd.DataFrame,
    orcamento: float,
    delta: float = 100.0
) -> Dict[str, float]:
    """
    Calcula os preços-sombra (shadow prices) das restrições.
    
    O preço-sombra indica quanto a função objetivo melhoraria se
    relaxássemos a restrição em uma unidade.
    
    Para a restrição de orçamento:
    Shadow Price = ΔVidas_Salvas / ΔOrçamento
    
    Interpretação: "Cada R$ 1 milhão adicional salva X vidas"
    
    Args:
        df_dados: DataFrame com dados consolidados
        orcamento: Orçamento base
        delta: Variação para calcular derivada numérica (R$ milhões)
               Usar delta maior (100) para capturar variação marginal corretamente
    
    Returns:
        Dicionário com preços-sombra por restrição
    """
    # Resultado base
    resultado_base = otimizar_alocacao(df_dados, orcamento)
    
    # Variação no orçamento total (usa delta maior para capturar variação)
    resultado_mais = otimizar_alocacao(df_dados, orcamento + delta)
    resultado_menos = otimizar_alocacao(df_dados, orcamento - delta)
    
    # Shadow price do orçamento (derivada central)
    shadow_orcamento = (
        resultado_mais.reducao_crimes - resultado_menos.reducao_crimes
    ) / (2 * delta)
    
    shadow_prices = {
        'orcamento_total': round(shadow_orcamento, 4),
        'interpretacao': f"Cada R$ 1 milhão adicional salva ~{shadow_orcamento:.2f} vidas"
    }
    
    # Shadow prices dos limites por estado
    for _, row in df_dados.iterrows():
        if pd.isna(row['orcamento_2022_milhoes']):
            continue
            
        estado = row['sigla']
        
        # Verifica se a restrição de máximo está ativa
        aloc = resultado_base.alocacao
        invest = aloc.loc[aloc['sigla'] == estado, 'investimento_milhoes'].values
        
        if len(invest) > 0:
            limite_max = row['orcamento_2022_milhoes'] * 0.30  # 30% default
            
            # Se investimento está no limite, a restrição está ativa
            if abs(invest[0] - limite_max) < 0.01:
                shadow_prices[f'limite_max_{estado}'] = "ATIVO"
    
    return shadow_prices


def gerar_grafico_tornado(
    df_dados: pd.DataFrame,
    orcamento: float,
    top_n: int = 10
) -> go.Figure:
    """
    Gera gráfico de tornado mostrando sensibilidade aos parâmetros.
    
    O gráfico de tornado é uma ferramenta visual que mostra quais
    parâmetros têm maior impacto na solução quando variados.
    
    Args:
        df_dados: DataFrame com dados consolidados
        orcamento: Orçamento base
        top_n: Número de parâmetros a mostrar
    
    Returns:
        Figura Plotly com gráfico de tornado
    """
    # Resultado base
    resultado_base = otimizar_alocacao(df_dados, orcamento)
    base_reducao = resultado_base.reducao_crimes
    
    impactos = []
    
    # Testa variação de elasticidade para cada estado
    for _, row in df_dados.iterrows():
        if pd.isna(row['elasticidade']):
            continue
        
        estado = row['sigla']
        elast_original = row['elasticidade']
        
        # Varia elasticidade em +/- 20%
        for var in [-0.20, 0.20]:
            df_var = df_dados.copy()
            df_var.loc[df_var['sigla'] == estado, 'elasticidade'] = elast_original * (1 + var)
            
            resultado = otimizar_alocacao(df_var, orcamento)
            
            if resultado.status == 'Optimal':
                impacto = resultado.reducao_crimes - base_reducao
                impactos.append({
                    'parametro': f"Elasticidade {estado}",
                    'variacao': '+20%' if var > 0 else '-20%',
                    'impacto': impacto,
                    'impacto_abs': abs(impacto)
                })
    
    # Ordena por impacto absoluto
    df_impactos = pd.DataFrame(impactos)
    
    if len(df_impactos) == 0:
        return go.Figure()
    
    # Agrupa por parâmetro e pega máximo impacto
    df_agg = df_impactos.groupby('parametro')['impacto_abs'].max().reset_index()
    df_agg = df_agg.nlargest(top_n, 'impacto_abs')
    
    # Pega valores positivos e negativos
    parametros_top = df_agg['parametro'].tolist()
    
    positivos = []
    negativos = []
    
    for param in parametros_top:
        df_param = df_impactos[df_impactos['parametro'] == param]
        pos = df_param[df_param['variacao'] == '+20%']['impacto'].values
        neg = df_param[df_param['variacao'] == '-20%']['impacto'].values
        
        positivos.append(pos[0] if len(pos) > 0 else 0)
        negativos.append(neg[0] if len(neg) > 0 else 0)
    
    # Cria gráfico de tornado
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=parametros_top,
        x=negativos,
        name='Redução -20%',
        orientation='h',
        marker_color='#ff6b6b'
    ))
    
    fig.add_trace(go.Bar(
        y=parametros_top,
        x=positivos,
        name='Aumento +20%',
        orientation='h',
        marker_color='#51cf66'
    ))
    
    fig.update_layout(
        title='Gráfico de Tornado: Sensibilidade da Elasticidade',
        xaxis_title='Impacto na Redução de Crimes (vidas)',
        yaxis_title='',
        barmode='relative',
        height=400 + top_n * 20,
        showlegend=True
    )
    
    return fig


def gerar_grafico_sensibilidade_orcamento(
    df_sensibilidade: pd.DataFrame
) -> go.Figure:
    """
    Gera gráfico de sensibilidade ao orçamento.
    
    Mostra como a redução de crimes varia com o orçamento disponível,
    e a eficiência marginal (vidas salvas por R$ adicional).
    
    Args:
        df_sensibilidade: DataFrame da função analisar_sensibilidade_orcamento
    
    Returns:
        Figura Plotly com gráficos combinados
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            'Redução de Crimes vs Orçamento',
            'Custo por Vida Salva vs Orçamento'
        ),
        vertical_spacing=0.15
    )
    
    # Gráfico 1: Redução de crimes
    fig.add_trace(
        go.Scatter(
            x=df_sensibilidade['orcamento_milhoes'],
            y=df_sensibilidade['reducao_crimes'],
            mode='lines+markers',
            name='Vidas Salvas',
            line=dict(color='#2ecc71', width=3),
            marker=dict(size=10)
        ),
        row=1, col=1
    )
    
    # Gráfico 2: Custo por vida
    fig.add_trace(
        go.Scatter(
            x=df_sensibilidade['orcamento_milhoes'],
            y=df_sensibilidade['custo_por_vida'],
            mode='lines+markers',
            name='Custo/Vida (R$ mi)',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=10)
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Orçamento (R$ milhões)", row=1, col=1)
    fig.update_xaxes(title_text="Orçamento (R$ milhões)", row=2, col=1)
    fig.update_yaxes(title_text="Vidas Salvas", row=1, col=1)
    fig.update_yaxes(title_text="R$ milhões / vida", row=2, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title='Análise de Sensibilidade: Orçamento'
    )
    
    return fig


def analisar_cenarios(
    df_dados: pd.DataFrame,
    orcamentos: Dict[str, float]
) -> pd.DataFrame:
    """
    Executa análise de cenários com diferentes orçamentos.
    
    Args:
        df_dados: DataFrame com dados consolidados
        orcamentos: Dicionário {nome_cenario: orcamento_milhoes}
    
    Returns:
        DataFrame comparativo entre cenários
    """
    resultados = []
    
    for nome, orcamento in orcamentos.items():
        resultado = otimizar_alocacao(df_dados, orcamento)
        
        if resultado.status == 'Optimal':
            # Top 3 estados por investimento
            top3 = resultado.alocacao.nlargest(3, 'investimento_milhoes')['sigla'].tolist()
            
            resultados.append({
                'cenario': nome,
                'orcamento_milhoes': orcamento,
                'orcamento_bilhoes': orcamento / 1000,
                'reducao_crimes': resultado.reducao_crimes,
                'reducao_pct': resultado.reducao_percentual,
                'custo_por_vida': round(orcamento / resultado.reducao_crimes, 2),
                'top_3_estados': ', '.join(top3),
                'estados_atendidos': (resultado.alocacao['investimento_milhoes'] > 0).sum()
            })
    
    return pd.DataFrame(resultados)


# =============================================================================
# TESTE DO MÓDULO
# =============================================================================
if __name__ == "__main__":
    from dados import carregar_dados_consolidados
    
    print("=" * 70)
    print("ANÁLISE DE SENSIBILIDADE")
    print("=" * 70)
    
    # Carrega dados
    df = carregar_dados_consolidados()
    orcamento_base = 5000  # R$ 5 bilhões
    
    print(f"\n📊 Orçamento base: R$ {orcamento_base:,} milhões")
    
    # 1. Sensibilidade ao orçamento
    print("\n" + "=" * 70)
    print("1. SENSIBILIDADE AO ORÇAMENTO")
    print("=" * 70)
    
    df_sens = analisar_sensibilidade_orcamento(df, orcamento_base)
    print(df_sens[['variacao_pct', 'orcamento_milhoes', 'reducao_crimes', 
                   'custo_por_vida']].to_string(index=False))
    
    # 2. Shadow Prices
    print("\n" + "=" * 70)
    print("2. PREÇOS-SOMBRA (SHADOW PRICES)")
    print("=" * 70)
    
    shadow = calcular_shadow_prices(df, orcamento_base)
    for k, v in shadow.items():
        if not k.startswith('limite_max'):
            print(f"  {k}: {v}")
    
    restricoes_ativas = [k for k in shadow.keys() if k.startswith('limite_max')]
    print(f"\n  Restrições de limite máximo ativas: {len(restricoes_ativas)}")
    if restricoes_ativas:
        print(f"  Estados no limite: {[r.replace('limite_max_', '') for r in restricoes_ativas[:5]]}")
    
    # 3. Análise de cenários
    print("\n" + "=" * 70)
    print("3. ANÁLISE DE CENÁRIOS")
    print("=" * 70)
    
    cenarios = {
        'Conservador': 2000,
        'Moderado': 5000,
        'Ambicioso': 10000,
        'Máximo': 20000
    }
    
    df_cenarios = analisar_cenarios(df, cenarios)
    print(df_cenarios[['cenario', 'orcamento_bilhoes', 'reducao_crimes', 
                       'custo_por_vida', 'estados_atendidos']].to_string(index=False))
