# =============================================================================
# MÓDULO DE OTIMIZAÇÃO MULTI-PERÍODO
# =============================================================================
# Este módulo estende o modelo de otimização para múltiplos períodos (anos).
#
# MOTIVAÇÃO:
# O modelo original otimiza apenas um período. Na prática, o planejamento
# de segurança pública é feito para múltiplos anos, e o investimento de
# um ano afeta a taxa de crime nos anos seguintes (efeito acumulado).
#
# FORMULAÇÃO:
# Variáveis: x[i,t] = investimento no estado i no período t
#
# Função Objetivo:
# Min Σ_t Σ_i [ C[i,t] × (1 - ε_i × Σ_{s≤t} x[i,s] / O_i) ]
#
# Restrições:
# 1. Orçamento por período: Σ_i x[i,t] ≤ B_t
# 2. Orçamento total: Σ_t Σ_i x[i,t] ≤ B_total
# 3. Limites por estado/período: L_i ≤ x[i,t] ≤ U_i
#
# NOTA: Este é um modelo simplificado. Um modelo completo consideraria
# dinâmica de crime, defasagens, depreciação de capital, etc.
# =============================================================================

import pandas as pd
import numpy as np
from pulp import (
    LpProblem, LpMinimize, LpVariable, lpSum,
    LpStatus, value, PULP_CBC_CMD
)
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ResultadoMultiPeriodo:
    """Resultados da otimização multi-período."""
    status: str
    n_periodos: int
    orcamento_total_usado: float
    reducao_total_crimes: float
    alocacao_por_periodo: Dict[int, pd.DataFrame]
    reducao_por_periodo: Dict[int, float]
    trajetoria_crimes: pd.DataFrame


def otimizar_multi_periodo(
    df_dados: pd.DataFrame,
    orcamentos_por_periodo: List[float],
    fator_acumulacao: float = 0.7,
    depreciacao_anual: float = 0.1,
    investimento_min_pct: float = 0.0,
    investimento_max_pct: float = 30.0,
    verbose: bool = False
) -> ResultadoMultiPeriodo:
    """
    Resolve o problema de otimização para múltiplos períodos.
    
    Características do modelo:
    - Investimentos se acumulam ao longo do tempo (com depreciação)
    - Cada período tem seu próprio orçamento
    - O efeito do investimento persiste (mas diminui) em períodos futuros
    
    Args:
        df_dados: DataFrame com dados consolidados
        orcamentos_por_periodo: Lista de orçamentos para cada período [B_1, B_2, ...]
        fator_acumulacao: Quanto do investimento de períodos anteriores ainda afeta
        depreciacao_anual: Taxa de depreciação do "estoque" de segurança
        investimento_min_pct: % mínimo de investimento por estado
        investimento_max_pct: % máximo de investimento por estado
        verbose: Exibir detalhes do solver
    
    Returns:
        ResultadoMultiPeriodo com alocações e trajetórias
    """
    # Prepara dados
    df = df_dados.dropna(subset=['orcamento_2022_milhoes', 'elasticidade', 'mortes_violentas']).copy()
    
    estados = df['sigla'].tolist()
    n_periodos = len(orcamentos_por_periodo)
    periodos = list(range(1, n_periodos + 1))
    
    # Parâmetros
    mortes = dict(zip(df['sigla'], df['mortes_violentas']))
    orcamento_atual = dict(zip(df['sigla'], df['orcamento_2022_milhoes']))
    elasticidade = dict(zip(df['sigla'], df['elasticidade']))
    
    # Limites
    inv_min = {e: orcamento_atual[e] * investimento_min_pct / 100 for e in estados}
    inv_max = {e: orcamento_atual[e] * investimento_max_pct / 100 for e in estados}
    
    # ==========================================================================
    # MODELO DE PROGRAMAÇÃO LINEAR MULTI-PERÍODO
    # ==========================================================================
    
    modelo = LpProblem("Alocacao_Multi_Periodo", LpMinimize)
    
    # Variáveis de decisão: x[estado, periodo]
    x = {}
    for e in estados:
        for t in periodos:
            x[e, t] = LpVariable(
                name=f"invest_{e}_t{t}",
                lowBound=inv_min[e],
                upBound=inv_max[e],
                cat='Continuous'
            )
    
    # Variáveis auxiliares: estoque acumulado de investimento por estado/período
    # estoque[e,t] = x[e,t] + (1-depreciacao) * estoque[e,t-1]
    estoque = {}
    for e in estados:
        for t in periodos:
            if t == 1:
                estoque[e, t] = x[e, t]
            else:
                estoque[e, t] = x[e, t] + (1 - depreciacao_anual) * estoque[e, t-1]
    
    # --------------------------------------------------------------------------
    # FUNÇÃO OBJETIVO
    # --------------------------------------------------------------------------
    # Minimizar crimes totais ao longo de todos os períodos
    # Crimes[e,t] = Mortes_base[e] × (1 - elasticidade[e] × estoque[e,t] / orcamento[e])
    #
    # Para linearizar, minimizamos:
    # Σ_t Σ_e [ -mortes[e] × elasticidade[e] × estoque[e,t] / orcamento[e] ]
    # --------------------------------------------------------------------------
    
    # Peso para períodos futuros (desconto temporal - opcional)
    desconto = {t: 1.0 for t in periodos}  # Sem desconto por padrão
    
    modelo += lpSum([
        desconto[t] * (-mortes[e] * elasticidade[e] * estoque[e, t] / orcamento_atual[e])
        for e in estados
        for t in periodos
    ]), "Funcao_Objetivo"
    
    # --------------------------------------------------------------------------
    # RESTRIÇÕES
    # --------------------------------------------------------------------------
    
    # Restrição 1: Orçamento por período
    for t in periodos:
        modelo += (
            lpSum([x[e, t] for e in estados]) <= orcamentos_por_periodo[t-1],
            f"Orcamento_Periodo_{t}"
        )
    
    # Resolve
    solver = PULP_CBC_CMD(msg=verbose)
    modelo.solve(solver)
    
    status = LpStatus[modelo.status]
    
    if status != 'Optimal':
        return ResultadoMultiPeriodo(
            status=status,
            n_periodos=n_periodos,
            orcamento_total_usado=0,
            reducao_total_crimes=0,
            alocacao_por_periodo={},
            reducao_por_periodo={},
            trajetoria_crimes=pd.DataFrame()
        )
    
    # ==========================================================================
    # EXTRAÇÃO DOS RESULTADOS
    # ==========================================================================
    
    alocacao_por_periodo = {}
    reducao_por_periodo = {}
    trajetoria_lista = []
    orcamento_total = 0
    reducao_total = 0
    
    for t in periodos:
        alocacao_lista = []
        reducao_periodo = 0
        
        for e in estados:
            investimento = value(x[e, t])
            
            # Calcula estoque acumulado até t
            estoque_acum = 0
            for s in range(1, t + 1):
                invest_s = value(x[e, s])
                # Aplica depreciação para investimentos de períodos anteriores
                anos_passados = t - s
                fator_dep = (1 - depreciacao_anual) ** anos_passados
                estoque_acum += invest_s * fator_dep
            
            # Redução de crimes baseada no estoque acumulado
            crimes_base = mortes[e]
            reducao = crimes_base * elasticidade[e] * estoque_acum / orcamento_atual[e]
            crimes_apos = max(0, crimes_base - reducao)
            
            alocacao_lista.append({
                'sigla': e,
                'periodo': t,
                'investimento': round(investimento, 2),
                'estoque_acumulado': round(estoque_acum, 2),
                'crimes_base': crimes_base,
                'crimes_apos': round(crimes_apos, 0),
                'reducao': round(reducao, 0)
            })
            
            reducao_periodo += reducao
            orcamento_total += investimento
        
        df_periodo = pd.DataFrame(alocacao_lista)
        
        # Merge com dados do estado
        df_periodo = pd.merge(
            df_periodo,
            df[['sigla', 'estado', 'regiao']],
            on='sigla'
        )
        
        alocacao_por_periodo[t] = df_periodo
        reducao_por_periodo[t] = round(reducao_periodo, 0)
        reducao_total += reducao_periodo
        
        # Trajetória agregada
        trajetoria_lista.append({
            'periodo': t,
            'orcamento_periodo': orcamentos_por_periodo[t-1],
            'investimento_total': df_periodo['investimento'].sum(),
            'crimes_base': df_periodo['crimes_base'].sum(),
            'crimes_apos': df_periodo['crimes_apos'].sum(),
            'reducao_acumulada': reducao_periodo
        })
    
    df_trajetoria = pd.DataFrame(trajetoria_lista)
    
    return ResultadoMultiPeriodo(
        status=status,
        n_periodos=n_periodos,
        orcamento_total_usado=round(orcamento_total, 2),
        reducao_total_crimes=round(reducao_total, 0),
        alocacao_por_periodo=alocacao_por_periodo,
        reducao_por_periodo=reducao_por_periodo,
        trajetoria_crimes=df_trajetoria
    )


def comparar_estrategias(
    df_dados: pd.DataFrame,
    orcamento_total: float,
    n_periodos: int = 5
) -> pd.DataFrame:
    """
    Compara diferentes estratégias de distribuição temporal do orçamento.
    
    Estratégias:
    1. Uniforme: mesmo valor todo período
    2. Frontloaded: mais no início, menos depois
    3. Backloaded: menos no início, mais depois
    4. Crescente: aumenta gradualmente
    
    Args:
        df_dados: DataFrame com dados
        orcamento_total: Orçamento total para todos os períodos
        n_periodos: Número de períodos
    
    Returns:
        DataFrame comparativo
    """
    orcamento_medio = orcamento_total / n_periodos
    
    estrategias = {
        'Uniforme': [orcamento_medio] * n_periodos,
        'Frontloaded': [orcamento_medio * (1 + 0.5 * (n_periodos - t) / n_periodos) 
                        for t in range(n_periodos)],
        'Backloaded': [orcamento_medio * (1 + 0.5 * t / n_periodos) 
                       for t in range(n_periodos)],
        'Crescente_Linear': [orcamento_medio * (0.5 + t / n_periodos) 
                             for t in range(n_periodos)]
    }
    
    # Normaliza para somar ao orçamento total
    for nome in estrategias:
        soma = sum(estrategias[nome])
        estrategias[nome] = [x * orcamento_total / soma for x in estrategias[nome]]
    
    resultados = []
    
    for nome, orcamentos in estrategias.items():
        resultado = otimizar_multi_periodo(df_dados, orcamentos)
        
        if resultado.status == 'Optimal':
            resultados.append({
                'estrategia': nome,
                'orcamento_total': orcamento_total,
                'n_periodos': n_periodos,
                'reducao_total': resultado.reducao_total_crimes,
                'reducao_primeiro_periodo': resultado.reducao_por_periodo[1],
                'reducao_ultimo_periodo': resultado.reducao_por_periodo[n_periodos],
                'distribuicao': [round(x, 0) for x in orcamentos]
            })
    
    return pd.DataFrame(resultados)


# =============================================================================
# TESTE DO MÓDULO
# =============================================================================
if __name__ == "__main__":
    from dados import carregar_dados_consolidados
    
    print("=" * 70)
    print("OTIMIZAÇÃO MULTI-PERÍODO")
    print("=" * 70)
    
    # Carrega dados
    df = carregar_dados_consolidados()
    
    # 1. Otimização com 5 períodos (anos)
    print("\n" + "=" * 70)
    print("1. OTIMIZAÇÃO PARA 5 PERÍODOS")
    print("   Orçamento: R$ 2 bi/ano = R$ 10 bi total")
    print("=" * 70)
    
    orcamentos = [2000, 2000, 2000, 2000, 2000]  # R$ 2 bi por ano
    
    resultado = otimizar_multi_periodo(df, orcamentos)
    
    print(f"\n📊 Resultados:")
    print(f"  Status: {resultado.status}")
    print(f"  Orçamento total usado: R$ {resultado.orcamento_total_usado:,.0f} milhões")
    print(f"  Redução total de crimes: {resultado.reducao_total_crimes:,.0f}")
    
    print(f"\n📈 Redução por período:")
    for t, red in resultado.reducao_por_periodo.items():
        print(f"  Período {t}: {red:,.0f} crimes evitados")
    
    print(f"\n📊 Trajetória:")
    print(resultado.trajetoria_crimes[
        ['periodo', 'investimento_total', 'crimes_apos', 'reducao_acumulada']
    ].to_string(index=False))
    
    # 2. Comparação de estratégias
    print("\n" + "=" * 70)
    print("2. COMPARAÇÃO DE ESTRATÉGIAS DE DISTRIBUIÇÃO TEMPORAL")
    print("=" * 70)
    
    df_estrategias = comparar_estrategias(df, orcamento_total=10000, n_periodos=5)
    
    print("\n" + df_estrategias[
        ['estrategia', 'reducao_total', 'reducao_primeiro_periodo', 'reducao_ultimo_periodo']
    ].to_string(index=False))
    
    # 3. Efeito da depreciação
    print("\n" + "=" * 70)
    print("3. EFEITO DA TAXA DE DEPRECIAÇÃO")
    print("=" * 70)
    
    for dep in [0.0, 0.1, 0.2, 0.3]:
        resultado = otimizar_multi_periodo(
            df, orcamentos,
            depreciacao_anual=dep
        )
        print(f"  Depreciação {dep*100:.0f}%: {resultado.reducao_total_crimes:,.0f} crimes evitados")
    
    print("\n" + "=" * 70)
    print("INTERPRETAÇÃO")
    print("=" * 70)
    print("""
    • O modelo multi-período captura o efeito acumulado do investimento
    • Investimentos de anos anteriores continuam gerando benefícios
    • Com depreciação 0%, o efeito é permanente
    • Com depreciação 10%/ano, o efeito diminui gradualmente
    • Estratégia "frontloaded" (investir mais cedo) é ligeiramente
      superior devido ao efeito de acumulação
    """)
