# =============================================================================
# MÓDULO DE OTIMIZAÇÃO - PROGRAMAÇÃO LINEAR PARA ALOCAÇÃO DE RECURSOS
# =============================================================================
# Este módulo implementa o modelo de Programação Linear usando PuLP.
#
# PROBLEMA DE OTIMIZAÇÃO:
# Dado um orçamento suplementar disponível, determinar quanto investir
# em cada estado para minimizar o número total de crimes esperados.
#
# FORMULAÇÃO MATEMÁTICA:
#
# Variáveis de Decisão:
#   x_i = investimento adicional no estado i (em R$ milhões)
#
# Função Objetivo (Minimizar):
#   Min Σ (Crimes_i × (1 - Elasticidade_i × x_i / Orçamento_i))
#
# Restrições:
#   (1) Σ x_i ≤ Orçamento_Total_Disponível  (limite de orçamento)
#   (2) x_i ≥ Investimento_Mínimo_i          (piso por estado)
#   (3) x_i ≤ Investimento_Máximo_i          (teto por estado)
#   (4) x_i ≥ 0                              (não-negatividade)
#
# MÉTODO DE SOLUÇÃO:
#   Simplex (via solver CBC do PuLP)
#
# REFERÊNCIAS:
# - Winston, W. L. "Operations Research: Applications and Algorithms"
# - Hillier, F. S.; Lieberman, G. J. "Introduction to Operations Research"
# =============================================================================

import pandas as pd
import numpy as np
from pulp import (
    LpProblem, LpMinimize, LpVariable, lpSum, 
    LpStatus, value, PULP_CBC_CMD
)
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class ResultadoOtimizacao:
    """
    Estrutura para armazenar resultados da otimização.
    
    Attributes:
        status: Status da solução ('Optimal', 'Infeasible', etc.)
        orcamento_usado: Total de orçamento alocado (R$ milhões)
        reducao_crimes: Redução esperada no número de crimes
        reducao_percentual: Redução percentual da taxa de crimes
        alocacao: DataFrame com alocação por estado
        fo_valor: Valor da função objetivo
    """
    status: str
    orcamento_usado: float
    reducao_crimes: float
    reducao_percentual: float
    alocacao: pd.DataFrame
    fo_valor: float


def otimizar_alocacao(
    df_dados: pd.DataFrame,
    orcamento_disponivel: float,
    investimento_minimo_pct: float = 0.0,
    investimento_maximo_pct: float = 50.0,
    verbose: bool = False
) -> ResultadoOtimizacao:
    """
    Resolve o problema de otimização de alocação de recursos.
    
    Este é o ponto central do modelo de Pesquisa Operacional.
    Usa o método Simplex (via PuLP/CBC) para encontrar a alocação ótima.
    
    Args:
        df_dados: DataFrame com dados consolidados dos estados
                  (deve conter: sigla, mortes_violentas, orcamento_2022_milhoes, elasticidade)
        orcamento_disponivel: Orçamento total disponível para distribuição (R$ milhões)
        investimento_minimo_pct: % mínimo do orçamento atual como piso de investimento
        investimento_maximo_pct: % máximo do orçamento atual como teto de investimento
        verbose: Se True, exibe detalhes do solver
    
    Returns:
        ResultadoOtimizacao com status, alocação e métricas
    """
    
    # ==========================================================================
    # PREPARAÇÃO DOS DADOS
    # ==========================================================================
    
    # Filtra estados com dados completos (remove NaN)
    df = df_dados.dropna(subset=['orcamento_2022_milhoes', 'elasticidade', 'mortes_violentas']).copy()
    
    # Lista de estados (índices do problema)
    estados = df['sigla'].tolist()
    n_estados = len(estados)
    
    # Extrai parâmetros do modelo
    mortes = dict(zip(df['sigla'], df['mortes_violentas']))
    orcamento_atual = dict(zip(df['sigla'], df['orcamento_2022_milhoes']))
    elasticidade = dict(zip(df['sigla'], df['elasticidade']))
    
    # Calcula limites de investimento por estado
    # Mínimo: garantir algum investimento proporcional
    # Máximo: evitar concentração excessiva em poucos estados
    inv_min = {e: orcamento_atual[e] * investimento_minimo_pct / 100 for e in estados}
    inv_max = {e: orcamento_atual[e] * investimento_maximo_pct / 100 for e in estados}
    
    # ==========================================================================
    # CRIAÇÃO DO MODELO DE PROGRAMAÇÃO LINEAR
    # ==========================================================================
    
    # Cria o problema de minimização
    modelo = LpProblem("Alocacao_Seguranca_Publica", LpMinimize)
    
    # --------------------------------------------------------------------------
    # VARIÁVEIS DE DECISÃO
    # --------------------------------------------------------------------------
    # x[i] = investimento adicional no estado i (em R$ milhões)
    # 
    # Cada variável tem limite inferior (inv_min) e superior (inv_max)
    # para garantir uma distribuição equilibrada dos recursos.
    # --------------------------------------------------------------------------
    
    x = {
        e: LpVariable(
            name=f"invest_{e}",
            lowBound=inv_min[e],
            upBound=inv_max[e],
            cat='Continuous'  # Variável contínua (não inteira)
        )
        for e in estados
    }
    
    # --------------------------------------------------------------------------
    # FUNÇÃO OBJETIVO
    # --------------------------------------------------------------------------
    # Minimizar o número esperado de crimes após o investimento.
    #
    # A redução de crimes é modelada como:
    #   Crimes_Após = Crimes_Antes × (1 - Elasticidade × Δ_Orçamento / Orçamento_Atual)
    #
    # Onde:
    #   - Elasticidade: sensibilidade do crime ao investimento (0.05 a 0.15)
    #   - Δ_Orçamento: investimento adicional (variável x[i])
    #   - Orçamento_Atual: orçamento existente do estado
    #
    # Simplificando para Programação Linear:
    #   Crimes_Após = Crimes_Antes - Crimes_Antes × Elasticidade × x[i] / Orçamento_Atual
    #
    # Como queremos minimizar, podemos usar a parte variável:
    #   Min Σ [ -Crimes[i] × Elasticidade[i] × x[i] / Orçamento[i] ]
    #
    # Ou equivalentemente, maximizar a redução:
    #   Max Σ [ Crimes[i] × Elasticidade[i] × x[i] / Orçamento[i] ]
    #
    # Para manter como minimização, usamos o negativo:
    # --------------------------------------------------------------------------
    
    modelo += lpSum([
        -mortes[e] * elasticidade[e] * x[e] / orcamento_atual[e]
        for e in estados
    ]), "Funcao_Objetivo_Minimizar_Crimes"
    
    # --------------------------------------------------------------------------
    # RESTRIÇÕES
    # --------------------------------------------------------------------------
    
    # Restrição 1: O total investido não pode exceder o orçamento disponível
    modelo += (
        lpSum([x[e] for e in estados]) <= orcamento_disponivel,
        "Restricao_Orcamento_Total"
    )
    
    # Nota: As restrições de limite mínimo e máximo por estado já estão
    # incorporadas nos limites das variáveis (lowBound e upBound).
    
    # ==========================================================================
    # RESOLUÇÃO DO PROBLEMA
    # ==========================================================================
    
    # Configura o solver CBC (COIN-OR Branch and Cut)
    # É um solver open-source eficiente para Programação Linear
    solver = PULP_CBC_CMD(msg=verbose)
    
    # Resolve o problema
    modelo.solve(solver)
    
    # ==========================================================================
    # EXTRAÇÃO DOS RESULTADOS
    # ==========================================================================
    
    status = LpStatus[modelo.status]
    
    # Se não encontrou solução ótima, retorna com status de erro
    if status != 'Optimal':
        return ResultadoOtimizacao(
            status=status,
            orcamento_usado=0.0,
            reducao_crimes=0.0,
            reducao_percentual=0.0,
            alocacao=pd.DataFrame(),
            fo_valor=0.0
        )
    
    # Extrai valores das variáveis de decisão
    alocacao_lista = []
    for e in estados:
        investimento = value(x[e])
        crimes_antes = mortes[e]
        
        # Calcula redução de crimes com o investimento
        reducao = crimes_antes * elasticidade[e] * investimento / orcamento_atual[e]
        crimes_depois = crimes_antes - reducao
        
        alocacao_lista.append({
            'sigla': e,
            'investimento_milhoes': round(investimento, 2),
            'mortes_antes': int(crimes_antes),
            'mortes_depois': int(round(crimes_depois)),
            'reducao_mortes': int(round(reducao)),
            'reducao_percentual': round(reducao / crimes_antes * 100, 2) if crimes_antes > 0 else 0
        })
    
    df_alocacao = pd.DataFrame(alocacao_lista)
    
    # Merge com dados originais para informações adicionais
    df_alocacao = pd.merge(
        df_alocacao,
        df[['sigla', 'estado', 'regiao', 'populacao', 'orcamento_2022_milhoes', 'elasticidade']],
        on='sigla',
        how='left'
    )
    
    # Calcula métricas agregadas
    orcamento_usado = df_alocacao['investimento_milhoes'].sum()
    reducao_total = df_alocacao['reducao_mortes'].sum()
    mortes_antes_total = df_alocacao['mortes_antes'].sum()
    reducao_pct_total = reducao_total / mortes_antes_total * 100 if mortes_antes_total > 0 else 0
    
    return ResultadoOtimizacao(
        status=status,
        orcamento_usado=round(orcamento_usado, 2),
        reducao_crimes=reducao_total,
        reducao_percentual=round(reducao_pct_total, 2),
        alocacao=df_alocacao,
        fo_valor=round(value(modelo.objective), 4)
    )


def gerar_formulacao_latex() -> Dict[str, str]:
    """
    Gera as equações do modelo formatadas em LaTeX para exibição educacional.
    
    Útil para a seção "Explicação do Modelo" na interface Streamlit.
    
    Returns:
        Dicionário com chaves: 'objetivo', 'restricoes', 'variaveis'
    """
    
    formulacao = {
        'variaveis': r"""
        \textbf{Variáveis de Decisão:}
        $$x_i = \text{Investimento adicional no estado } i \text{ (em R\$ milhões)}$$
        """,
        
        'objetivo': r"""
        \textbf{Função Objetivo (Minimizar crimes esperados):}
        $$\min Z = \sum_{i=1}^{n} \left( C_i \times \left(1 - \varepsilon_i \times \frac{x_i}{O_i}\right) \right)$$
        
        Onde:
        - $C_i$ = número de crimes no estado $i$
        - $\varepsilon_i$ = elasticidade crime-investimento do estado $i$
        - $O_i$ = orçamento atual de segurança do estado $i$
        - $x_i$ = investimento adicional (variável de decisão)
        """,
        
        'restricoes': r"""
        \textbf{Restrições:}
        
        1. **Limite de orçamento total:**
        $$\sum_{i=1}^{n} x_i \leq B$$
        
        2. **Investimento mínimo por estado:**
        $$x_i \geq L_i \quad \forall i$$
        
        3. **Investimento máximo por estado:**
        $$x_i \leq U_i \quad \forall i$$
        
        4. **Não-negatividade:**
        $$x_i \geq 0 \quad \forall i$$
        
        Onde:
        - $B$ = orçamento total disponível
        - $L_i$ = limite inferior (piso) para estado $i$
        - $U_i$ = limite superior (teto) para estado $i$
        """
    }
    
    return formulacao


def explicar_elasticidade() -> str:
    """
    Retorna texto explicativo sobre o conceito de elasticidade no modelo.
    
    Útil para a seção educacional da interface.
    """
    return """
    ### 📊 O que é Elasticidade Crime-Investimento?
    
    A **elasticidade** é um conceito da economia que mede a sensibilidade 
    de uma variável em relação a outra. No nosso modelo:
    
    > **Elasticidade = Quanto a taxa de crime reduz quando aumentamos o investimento em 1%**
    
    #### Exemplo Prático:
    Se um estado tem elasticidade de **0.10**, isso significa que:
    - Um aumento de **10%** no orçamento de segurança
    - Resulta em redução de **1%** na taxa de crimes
    
    #### Por que a elasticidade varia entre estados?
    
    1. **Eficiência da gestão**: Estados com melhor gestão conseguem 
       converter investimento em resultados de forma mais eficiente.
    
    2. **Rendimentos decrescentes**: Estados que já investem muito 
       têm menor margem para ganhos adicionais (a elasticidade diminui).
    
    3. **Características locais**: Fatores como urbanização, desigualdade 
       e infraestrutura afetam como o investimento se traduz em resultados.
    
    #### Valores típicos na literatura:
    - **0.05 a 0.08**: Baixa elasticidade (estados com alto investimento)
    - **0.08 a 0.12**: Elasticidade média
    - **0.12 a 0.15**: Alta elasticidade (maior potencial de retorno)
    """


# =============================================================================
# TESTE DO MÓDULO
# =============================================================================
if __name__ == "__main__":
    from dados import carregar_dados_consolidados
    
    print("=" * 70)
    print("TESTE: MODELO DE OTIMIZAÇÃO")
    print("=" * 70)
    
    # Carrega dados
    df = carregar_dados_consolidados()
    print(f"\n✓ Dados carregados: {len(df)} estados")
    
    # Parâmetros do teste
    orcamento_teste = 5000  # R$ 5 bilhões
    
    print(f"\n📊 Orçamento disponível para alocação: R$ {orcamento_teste:,.0f} milhões")
    print(f"   (equivalente a R$ {orcamento_teste/1000:.1f} bilhões)")
    
    # Executa otimização
    print("\n🔄 Executando otimização...")
    resultado = otimizar_alocacao(
        df_dados=df,
        orcamento_disponivel=orcamento_teste,
        investimento_minimo_pct=0,
        investimento_maximo_pct=30,
        verbose=False
    )
    
    print(f"\n✓ Status da solução: {resultado.status}")
    print(f"✓ Orçamento utilizado: R$ {resultado.orcamento_usado:,.2f} milhões")
    print(f"✓ Redução esperada de mortes: {resultado.reducao_crimes:,.0f}")
    print(f"✓ Redução percentual: {resultado.reducao_percentual:.2f}%")
    
    print("\n" + "=" * 70)
    print("TOP 10 ESTADOS COM MAIOR INVESTIMENTO ALOCADO")
    print("=" * 70)
    
    top10 = resultado.alocacao.nlargest(10, 'investimento_milhoes')[
        ['sigla', 'estado', 'investimento_milhoes', 'reducao_mortes', 'reducao_percentual']
    ]
    print(top10.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("ALOCAÇÃO POR REGIÃO")
    print("=" * 70)
    por_regiao = resultado.alocacao.groupby('regiao').agg({
        'investimento_milhoes': 'sum',
        'reducao_mortes': 'sum'
    }).round(2)
    print(por_regiao.to_string())
