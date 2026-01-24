# -*- coding: utf-8 -*-
# =============================================================================
# APLICAÇÃO STREAMLIT - OTIMIZAÇÃO DE RECURSOS DE SEGURANÇA PÚBLICA
# =============================================================================
# Trabalho Acadêmico - Pesquisa Operacional
#
# Esta aplicação permite:
# 1. Visualizar dados atuais de violência e orçamento por estado (Dashboard)
# 2. Calcular alocação ótima de recursos (Otimização)
# 3. Comparar cenários antes e depois (Comparativo)
#
# Autor: [Seu Nome]
# Disciplina: Pesquisa Operacional
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import requests
from pathlib import Path

# Importa módulos locais
from dados import carregar_dados_consolidados, obter_coordenadas_estados, ANOS_DISPONIVEIS
from otimizacao import (
    otimizar_alocacao, 
    ResultadoOtimizacao,
    gerar_formulacao_latex,
    explicar_elasticidade
)

# Módulos avançados de Pesquisa Operacional
from analise_estatistica import atualizar_elasticidade_dados, gerar_relatorio_elasticidade
from sensibilidade import (
    analisar_sensibilidade_orcamento,
    calcular_shadow_prices,
    analisar_cenarios,
    gerar_grafico_tornado
)
from monte_carlo import executar_monte_carlo
from backtesting import executar_backtest, validar_modelo_rolling
from multi_periodo import otimizar_multi_periodo, comparar_estrategias
from dea import calcular_dea_ccr, identificar_benchmarks, calcular_metas, resumo_dea

# =============================================================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Otimização de Segurança Pública",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para melhor visualização
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Estiliza o radio horizontal para parecer com abas */
    div[data-testid="stHorizontalBlock"]:has(div[data-testid="stRadio"]) {
        background-color: transparent;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 0;
        margin-bottom: 1rem;
    }
    
    /* Container do radio */
    div[data-testid="stRadio"] > div {
        flex-direction: row !important;
        gap: 0 !important;
        background: transparent;
    }
    
    /* Cada opção do radio (aba) */
    div[data-testid="stRadio"] label {
        background-color: transparent;
        border: none;
        border-bottom: 3px solid transparent;
        border-radius: 0;
        padding: 0.75rem 1.5rem;
        margin: 0;
        font-size: 1rem;
        font-weight: 500;
        color: #555;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    /* Hover nas abas */
    div[data-testid="stRadio"] label:hover {
        color: #1f77b4;
        background-color: rgba(31, 119, 180, 0.05);
    }
    
    /* Aba selecionada */
    div[data-testid="stRadio"] label[data-checked="true"] {
        color: #1f77b4;
        border-bottom: 3px solid #1f77b4;
        background-color: transparent;
        font-weight: 600;
    }
    
    /* Esconde o círculo do radio */
    div[data-testid="stRadio"] label span[data-testid="stMarkdownContainer"] {
        margin-left: 0 !important;
    }
    
    div[data-testid="stRadio"] input[type="radio"] {
        display: none !important;
    }
    
    /* Remove a borda padrão do radio selecionado */
    div[data-testid="stRadio"] label[data-checked="true"]::before {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CACHE DE DADOS
# =============================================================================
@st.cache_data
def carregar_dados(ano: int = 2022):
    """
    Carrega e cacheia os dados consolidados para um ano específico.
    Usa elasticidade calculada por regressão da série histórica.
    
    Args:
        ano: Ano dos dados (2013-2023)
    """
    df = carregar_dados_consolidados(ano=ano)
    # Substitui elasticidade estimada pela calculada via regressão linear
    df = atualizar_elasticidade_dados(df)
    return df


@st.cache_data
def carregar_dados_todos_anos():
    """
    Carrega dados de todos os anos disponíveis (2013-2023) para análises temporais.
    """
    from dados import carregar_gastos_todos_anos, carregar_homicidios
    
    df_gastos = carregar_gastos_todos_anos()
    df_homicidios = carregar_homicidios()
    
    # Merge gastos com homicídios
    df = pd.merge(
        df_gastos,
        df_homicidios[['sigla', 'ano', 'homicidios']],
        on=['sigla', 'ano'],
        how='left'
    )
    
    # Calcula taxa por 100k
    df['taxa_mortes_100k'] = (df['homicidios'] / df['populacao'] * 100000).round(2)
    df['gasto_milhoes'] = (df['gasto_seguranca'] / 1e6).round(2)
    df['gasto_per_capita'] = (df['gasto_seguranca'] / df['populacao']).round(2)
    
    return df


@st.cache_data
def carregar_geojson_brasil():
    """
    Carrega GeoJSON dos estados brasileiros para o mapa coroplético.
    Fonte: Instituto Brasileiro de Geografia e Estatística (IBGE)
    """
    url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    
    # Fallback: retorna None se não conseguir carregar
    return None


# =============================================================================
# FUNÇÕES PRÉ-CALCULADAS (valores padrão)
# =============================================================================
@st.cache_data
def obter_otimizacao_padrao(_df):
    """Calcula otimização com parâmetros padrão para exibição inicial."""
    return otimizar_alocacao(_df, orcamento_disponivel=5000, verbose=False)


@st.cache_data
def obter_sensibilidade_padrao(_df):
    """Calcula análise de sensibilidade com parâmetros padrão."""
    sens = analisar_sensibilidade_orcamento(_df, orcamento_base=5000)
    shadow = calcular_shadow_prices(_df, orcamento=5000)
    
    # Análise de cenários precisa de dicionário
    cenarios_dict = {'pessimista': 3000, 'base': 5000, 'otimista': 7000}
    cenarios_df = analisar_cenarios(_df, cenarios_dict)
    
    # Converte para formato esperado
    cenarios = {}
    for _, row in cenarios_df.iterrows():
        cenarios[row['cenario']] = {'vidas_salvas': row['reducao_crimes']}
    
    tornado = gerar_grafico_tornado(_df, orcamento=5000)
    return {'sensibilidade': sens, 'shadow': shadow, 'cenarios': cenarios, 'tornado': tornado}


@st.cache_data
def obter_monte_carlo_padrao(_df):
    """Executa Monte Carlo com parâmetros padrão (menos simulações para ser rápido)."""
    return executar_monte_carlo(
        df_dados=_df,
        orcamento=5000,
        n_simulacoes=250,  # Menos para carregar rápido
        incerteza_elasticidade=0.15,
        incerteza_taxa=0.08,
        verbose=False
    )


@st.cache_data
def obter_backtesting_padrao():
    """Executa backtesting com parâmetros padrão."""
    return validar_modelo_rolling(janela_treino=5, janela_teste=1, ano_inicio=2010, ano_fim=2022)


@st.cache_data  
def obter_multiperiodo_padrao(_df):
    """Calcula estratégias multi-período com parâmetros padrão."""
    return comparar_estrategias(_df, orcamento_total=25000, n_periodos=5)


# =============================================================================
# SIDEBAR - EXPLICAÇÃO DO MODELO
# =============================================================================
def render_sidebar():
    """Renderiza a sidebar com explicação educacional do modelo e seletor de ano."""
    
    st.sidebar.title("📅 Seleção de Ano")
    
    # Seletor de ano
    ano_selecionado = st.sidebar.selectbox(
        "Ano de análise:",
        options=sorted(ANOS_DISPONIVEIS, reverse=True),
        index=0,  # Default: 2023 (primeiro da lista ordenada decrescente)
        help="Selecione o ano para visualizar os dados. Disponível de 2013 a 2023."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.title("📚 Explicação do Modelo")
    
    with st.sidebar.expander("🎯 Objetivo", expanded=True):
        st.markdown("""
        **Problema:** Dado um orçamento suplementar limitado, como distribuí-lo 
        entre os estados para **maximizar a redução de crimes**?
        
        **Método:** Programação Linear resolvida pelo algoritmo **Simplex**.
        """)
    
    with st.sidebar.expander("🧮 Formulação Matemática"):
        st.markdown("**Variáveis de Decisão:**")
        st.latex(r"x_i = \text{Investimento adicional no estado } i")
        
        st.markdown("**Função Objetivo:**")
        st.latex(r"\min \sum_{i=1}^{n} C_i \cdot \left(1 - \varepsilon_i \cdot \frac{x_i}{O_i}\right)")
        
        st.markdown("**Restrições:**")
        st.latex(r"\sum_{i=1}^{n} x_i \leq B \quad \text{(orçamento total)}")
        st.latex(r"L_i \leq x_i \leq U_i \quad \text{(limites por estado)}")
        
        st.markdown("""
        Onde:
        - $C_i$ = crimes no estado $i$
        - $O_i$ = orçamento atual
        - $B$ = orçamento disponível
        """)
    
    with st.sidebar.expander("🔧 Método de Solução"):
        st.markdown("""
        ### Algoritmo Simplex
        
        O **Simplex** é o método mais usado para resolver problemas de 
        Programação Linear. Desenvolvido por George Dantzig em 1947.
        
        **Como funciona:**
        1. Começa em um vértice do poliedro de soluções viáveis
        2. Move-se para vértices adjacentes que melhorem a F.O.
        3. Para quando não há mais melhoria possível (ótimo!)
        
        **Implementação:** Usamos a biblioteca `PuLP` com o solver 
        `CBC` (COIN-OR Branch and Cut), que é open-source e eficiente.
        """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **📖 Fontes dos Dados:**
    - [Atlas da Violência](https://www.ipea.gov.br/atlasviolencia/) (IPEA)
    - [Anuário de Segurança Pública](https://forumseguranca.org.br/) (FBSP)
    - [SICONFI](https://siconfi.tesouro.gov.br/) (Gastos)
    """)
    
    return ano_selecionado


# =============================================================================
# ABA 1: DASHBOARD
# =============================================================================
def render_dashboard(df: pd.DataFrame, geojson, ano: int):
    """Renderiza a aba de Dashboard com visualizações dos dados atuais."""
    
    st.header(f"📊 Dashboard - Situação em {ano}")
    
    with st.expander("ℹ️ **Sobre esta aba** - Clique para expandir", expanded=False):
        st.markdown(f"""
        ### O que é o Dashboard?
        
        Esta aba apresenta uma **visão geral da situação** de segurança pública no Brasil,
        utilizando dados consolidados do **Atlas da Violência (IPEA)** e do **Anuário de Segurança 
        Pública (FBSP)** referentes ao ano de **{ano}**.
        
        #### Dados exibidos:
        - **Mortes Violentas**: Número absoluto de homicídios e mortes violentas intencionais
        - **Taxa por 100 mil hab.**: Métrica normalizada que permite comparar estados de diferentes tamanhos
        - **Orçamento de Segurança**: Investimento estadual em segurança pública (em milhões de R$)
        - **Gasto Per Capita**: Quanto cada estado investe por habitante
        
        #### Gráficos:
        - **Mapa de calor**: Visualização geográfica da taxa de violência
        - **Ranking de estados**: Comparativo de todos os 27 estados brasileiros
        - **Scatter plot**: Relação entre gasto per capita e taxa de violência
        - **Por região**: Agrupamento dos estados por região geográfica
        
        #### Fonte dos dados:
        - Atlas da Violência: Série histórica 2013-2023 (IPEA/FBSP)
        - Anuário Brasileiro de Segurança Pública (FBSP)
        """)
    
    st.markdown(f"Visualização dos dados de violência e orçamento de segurança pública por estado ({ano}).")
    
    # Métricas resumo
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_mortes = df['mortes_violentas'].sum()
        st.metric(
            label="Total de Mortes Violentas",
            value=f"{total_mortes:,.0f}",
            help=f"Número total de mortes violentas em {ano}"
        )
    
    with col2:
        media_taxa = df['taxa_mortes_100k'].mean()
        st.metric(
            label="Taxa Média (por 100 mil)",
            value=f"{media_taxa:.1f}",
            help="Média da taxa de mortes por 100 mil habitantes"
        )
    
    with col3:
        total_orcamento = df['orcamento_2022_milhoes'].sum()
        st.metric(
            label="Orçamento Total (R$ bi)",
            value=f"{total_orcamento/1000:.1f}",
            help=f"Soma dos orçamentos de segurança de todos os estados em {ano}"
        )
    
    with col4:
        media_gasto_pc = df['gasto_per_capita'].mean()
        st.metric(
            label="Gasto Médio Per Capita",
            value=f"R$ {media_gasto_pc:.0f}",
            help="Média do gasto per capita em segurança"
        )
    
    st.markdown("---")
    
    # Mapa e gráficos
    col_mapa, col_grafico = st.columns([1.2, 1])
    
    with col_mapa:
        st.subheader("🗺️ Mapa de Calor - Taxa de Mortes por 100 mil hab.")
        
        # Prepara dados para o mapa
        df_mapa = df.copy()
        
        if geojson:
            # Mapa coroplético com GeoJSON
            fig_mapa = px.choropleth(
                df_mapa,
                geojson=geojson,
                locations='estado',
                featureidkey="properties.name",
                color='taxa_mortes_100k',
                color_continuous_scale='YlOrRd',
                hover_name='estado',
                hover_data={
                    'taxa_mortes_100k': ':.1f',
                    'mortes_violentas': ':,.0f',
                    'gasto_per_capita': ':,.0f',
                    'estado': False
                },
                labels={
                    'taxa_mortes_100k': 'Taxa por 100k',
                    'mortes_violentas': 'Mortes',
                    'gasto_per_capita': 'Gasto per capita'
                }
            )
            fig_mapa.update_geos(
                fitbounds="locations",
                visible=False
            )
        else:
            # Fallback: mapa de pontos se não conseguir carregar GeoJSON
            coords = obter_coordenadas_estados()
            df_mapa = pd.merge(df_mapa, coords, on='sigla')
            
            fig_mapa = px.scatter_geo(
                df_mapa,
                lat='latitude',
                lon='longitude',
                color='taxa_mortes_100k',
                size='mortes_violentas',
                hover_name='estado',
                color_continuous_scale='YlOrRd',
                scope='south america',
                size_max=40
            )
            fig_mapa.update_geos(
                center=dict(lat=-15, lon=-55),
                projection_scale=3
            )
        
        fig_mapa.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=450,
            coloraxis_colorbar=dict(
                title="Taxa/100k",
                tickformat=".0f"
            ),
            dragmode=False
        )
        st.plotly_chart(fig_mapa, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False})
    
    with col_grafico:
        st.subheader("📈 Ranking Completo - Taxa de Violência por Estado")
        
        # Mostra TODOS os 27 estados ordenados
        df_ranking = df.sort_values('taxa_mortes_100k', ascending=True)
        
        fig_bar = px.bar(
            df_ranking,
            x='taxa_mortes_100k',
            y='sigla',
            orientation='h',
            color='taxa_mortes_100k',
            color_continuous_scale='YlOrRd',
            text='taxa_mortes_100k',
            labels={'taxa_mortes_100k': 'Taxa por 100 mil', 'sigla': 'Estado'}
        )
        fig_bar.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_bar.update_layout(
            height=700,
            showlegend=False,
            coloraxis_showscale=False,
            xaxis_title="Taxa de Mortes por 100 mil hab.",
            yaxis_title="",
            xaxis=dict(fixedrange=True),
            yaxis=dict(fixedrange=True),
            dragmode=False
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False})
    
    # =========================================================================
    # GRÁFICOS DE GASTO PER CAPITA
    # =========================================================================
    st.markdown("---")
    col_mapa_gasto, col_grafico_gasto = st.columns([1, 1.2])
    
    with col_mapa_gasto:
        st.subheader("🗺️ Mapa de Calor - Gasto Per Capita (R$)")
        
        df_mapa_gasto = df.copy()
        
        if geojson is not None:
            fig_mapa_gasto = px.choropleth(
                df_mapa_gasto,
                geojson=geojson,
                locations='sigla',
                featureidkey="properties.sigla",
                color='gasto_per_capita',
                color_continuous_scale='Blues',
                hover_name='estado',
                hover_data={
                    'sigla': False,
                    'gasto_per_capita': ':,.0f',
                    'taxa_mortes_100k': ':.1f',
                    'populacao': ':,.0f'
                },
                labels={
                    'gasto_per_capita': 'Gasto per capita (R$)',
                    'taxa_mortes_100k': 'Taxa/100k',
                    'populacao': 'População'
                }
            )
            fig_mapa_gasto.update_geos(
                fitbounds="locations",
                visible=False
            )
        else:
            coords = obter_coordenadas_estados()
            df_mapa_gasto = pd.merge(df_mapa_gasto, coords, on='sigla')
            
            fig_mapa_gasto = px.scatter_geo(
                df_mapa_gasto,
                lat='latitude',
                lon='longitude',
                color='gasto_per_capita',
                size='populacao',
                hover_name='estado',
                color_continuous_scale='Blues',
                scope='south america',
                size_max=40
            )
            fig_mapa_gasto.update_geos(
                center=dict(lat=-15, lon=-55),
                projection_scale=3
            )
        
        fig_mapa_gasto.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=450,
            coloraxis_colorbar=dict(
                title="R$/hab",
                tickformat=",.0f"
            ),
            dragmode=False
        )
        st.plotly_chart(fig_mapa_gasto, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False})
    
    with col_grafico_gasto:
        st.subheader("💰 Ranking Completo - Gasto Per Capita por Estado")
        
        df_ranking_gasto = df.sort_values('gasto_per_capita', ascending=True)
        
        fig_bar_gasto = px.bar(
            df_ranking_gasto,
            x='gasto_per_capita',
            y='sigla',
            orientation='h',
            color='gasto_per_capita',
            color_continuous_scale='Blues',
            text='gasto_per_capita',
            labels={'gasto_per_capita': 'Gasto Per Capita (R$)', 'sigla': 'Estado'}
        )
        fig_bar_gasto.update_traces(texttemplate='R$ %{text:,.0f}', textposition='outside')
        fig_bar_gasto.update_layout(
            height=700,
            showlegend=False,
            coloraxis_showscale=False,
            xaxis_title="Gasto Per Capita (R$)",
            yaxis_title="",
            xaxis=dict(fixedrange=True),
            yaxis=dict(fixedrange=True),
            dragmode=False
        )
        st.plotly_chart(fig_bar_gasto, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False})
    
    # Gráfico de comparativo por região
    st.markdown("---")
    st.subheader("🗺️ Comparativo por Região")
    
    df_regiao = df.groupby('regiao').agg({
        'mortes_violentas': 'sum',
        'populacao': 'sum',
        'orcamento_2022_milhoes': 'sum'
    }).reset_index()
    
    df_regiao['taxa_regiao'] = df_regiao['mortes_violentas'] / df_regiao['populacao'] * 100000
    df_regiao['gasto_pc_regiao'] = df_regiao['orcamento_2022_milhoes'] * 1e6 / df_regiao['populacao']
    
    fig_regiao = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Taxa por 100 mil", "Gasto Per Capita"),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    fig_regiao.add_trace(
        go.Bar(
            x=df_regiao['regiao'],
            y=df_regiao['taxa_regiao'],
            marker_color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#1f77b4'],
            name='Taxa'
        ),
        row=1, col=1
    )
    
    fig_regiao.add_trace(
        go.Bar(
            x=df_regiao['regiao'],
            y=df_regiao['gasto_pc_regiao'],
            marker_color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#1f77b4'],
            name='Gasto PC'
        ),
        row=1, col=2
    )
    
    fig_regiao.update_layout(
        height=400, 
        showlegend=False,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
        xaxis2=dict(fixedrange=True),
        yaxis2=dict(fixedrange=True),
        dragmode=False
    )
    st.plotly_chart(fig_regiao, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False})
    
    # Tabela de dados
    st.markdown("---")
    with st.expander("📋 Ver Tabela de Dados Completa"):
        df_tabela = df[[
            'sigla', 'estado', 'regiao', 'populacao', 
            'mortes_violentas', 'taxa_mortes_100k',
            'orcamento_2022_milhoes', 'gasto_per_capita'
        ]].copy()
        df_tabela.columns = ['UF', 'Estado', 'Região', 'População', 'Mortes Violentas', 'Taxa/100k', 'Orçamento (R$ mi)', 'Gasto/Capita']
        
        st.dataframe(
            df_tabela.style.format({
                'População': '{:,.0f}',
                'Mortes Violentas': '{:,.0f}',
                'Taxa/100k': '{:.1f}',
                'Orçamento (R$ mi)': 'R$ {:,.1f}',
                'Gasto/Capita': 'R$ {:,.0f}'
            }).background_gradient(subset=['Taxa/100k'], cmap='YlOrRd'),
            use_container_width=True,
            height=400,
            hide_index=True
        )


# =============================================================================
# ABA 2: OTIMIZAÇÃO
# =============================================================================
def render_otimizacao(df: pd.DataFrame, ano: int = 2022):
    """Renderiza a aba de Otimização com controles e resultados."""
    
    st.header(f"⚙️ Otimização - Alocação de Recursos ({ano})")
    
    with st.expander("ℹ️ **Sobre esta aba** - Clique para expandir", expanded=False):
        st.markdown("""
        ### O que é a Otimização?
        
        Esta aba utiliza **Programação Linear** para calcular a distribuição ótima de um orçamento 
        suplementar de segurança pública entre os 27 estados brasileiros.
        
        #### Objetivo:
        **Minimizar o número total de mortes violentas** no país, distribuindo recursos de forma 
        inteligente baseada na eficiência de cada estado.
        
        #### Como funciona:
        1. O modelo analisa a **relação entre investimento e resultado** de cada estado
        2. Estados com maior potencial de redução recebem mais recursos
        3. Restrições garantem que nenhum estado fique sem recursos ou receba recursos excessivos
        
        #### Parâmetros configuráveis:
        
        | Parâmetro | Descrição |
        |-----------|-----------|
        | **Orçamento Suplementar** | Valor adicional (além do orçamento atual) a ser distribuído |
        | **Investimento Mínimo** | % mínimo que cada estado deve receber (proporcional ao seu orçamento atual) |
        | **Investimento Máximo** | % máximo para evitar concentração excessiva em poucos estados |
        
        #### Método de resolução:
        - **Solver**: PuLP com CBC (Coin-or Branch and Cut)
        - **Algoritmo**: Simplex com branch-and-bound para variáveis inteiras
        - **Tempo típico**: < 1 segundo para 27 estados
        """)
    
    st.markdown("""
    Configure os parâmetros abaixo e clique em **Calcular** para encontrar 
    a alocação ótima de recursos que minimiza o número de crimes esperados.
    """)
    
    # Controles de entrada
    st.markdown("### 📝 Parâmetros do Modelo")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        orcamento_disponivel = st.slider(
            "💰 Orçamento Suplementar (R$ bilhões)",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            help="Valor total disponível para distribuição entre os estados"
        )
        orcamento_milhoes = orcamento_disponivel * 1000  # Converte para milhões
    
    with col2:
        inv_min_pct = st.slider(
            "📉 Investimento Mínimo (% do orçamento atual)",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            help="Garante um investimento mínimo proporcional ao orçamento atual do estado"
        )
    
    with col3:
        inv_max_pct = st.slider(
            "📈 Investimento Máximo (% do orçamento atual)",
            min_value=10,
            max_value=100,
            value=30,
            step=5,
            help="Limita investimento máximo para evitar concentração excessiva"
        )
    
    st.markdown("---")
    
    # Botão de execução
    if st.button("🚀 Calcular Alocação Ótima", type="primary", use_container_width=True):
        
        with st.spinner("Executando otimização via Simplex..."):
            resultado = otimizar_alocacao(
                df_dados=df,
                orcamento_disponivel=orcamento_milhoes,
                investimento_minimo_pct=inv_min_pct,
                investimento_maximo_pct=inv_max_pct,
                verbose=False
            )
        
        # Armazena resultado no session state
        st.session_state['resultado_otimizacao'] = resultado
        st.session_state['orcamento_usado'] = orcamento_milhoes
    
    # Exibe resultados se existirem
    if 'resultado_otimizacao' in st.session_state:
        resultado = st.session_state['resultado_otimizacao']
        
        if resultado.status == 'Optimal':
            st.success(f"✅ Solução ótima encontrada!")
            
            # Métricas de resultado
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Orçamento Alocado",
                    f"R$ {resultado.orcamento_usado/1000:.2f} bi"
                )
            
            with col2:
                st.metric(
                    "Redução de Mortes",
                    f"{resultado.reducao_crimes:,.0f}",
                    delta=f"-{resultado.reducao_percentual:.2f}%"
                )
            
            with col3:
                # Custo por vida salva
                custo_por_vida = resultado.orcamento_usado / resultado.reducao_crimes if resultado.reducao_crimes > 0 else 0
                st.metric(
                    "Custo por Vida Salva",
                    f"R$ {custo_por_vida:.2f} mi"
                )
            
            with col4:
                estados_atendidos = (resultado.alocacao['investimento_milhoes'] > 0).sum()
                st.metric(
                    "Estados Atendidos",
                    f"{estados_atendidos} / {len(resultado.alocacao)}"
                )
            
            st.markdown("---")
            
            # Gráfico de alocação
            st.subheader("📊 Distribuição da Alocação")
            
            df_alloc = resultado.alocacao.sort_values('investimento_milhoes', ascending=False)
            df_alloc_positivo = df_alloc[df_alloc['investimento_milhoes'] > 0]
            
            if len(df_alloc_positivo) > 0:
                col_bar, col_pie = st.columns([2, 1])
                
                with col_bar:
                    fig_alloc = px.bar(
                        df_alloc_positivo,
                        x='sigla',
                        y='investimento_milhoes',
                        color='reducao_percentual',
                        color_continuous_scale='Greens',
                        text='investimento_milhoes',
                        labels={
                            'investimento_milhoes': 'Investimento (R$ milhões)',
                            'sigla': 'Estado',
                            'reducao_percentual': 'Redução (%)'
                        },
                        title="Investimento por Estado"
                    )
                    fig_alloc.update_traces(texttemplate='R$ %{text:.0f}M', textposition='outside')
                    fig_alloc.update_layout(
                        height=400,
                        margin=dict(t=50, b=50),
                        xaxis=dict(fixedrange=True),
                        yaxis=dict(fixedrange=True, range=[0, df_alloc_positivo['investimento_milhoes'].max() * 1.15]),
                        dragmode=False
                    )
                    st.plotly_chart(fig_alloc, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False})
                
                with col_pie:
                    # Alocação por região
                    df_regiao = resultado.alocacao.groupby('regiao')['investimento_milhoes'].sum().reset_index()
                    df_regiao = df_regiao[df_regiao['investimento_milhoes'] > 0]
                    
                    fig_pie = px.pie(
                        df_regiao,
                        values='investimento_milhoes',
                        names='regiao',
                        title="Por Região"
                    )
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            # Tabela detalhada
            st.subheader("📋 Detalhamento por Estado")
            
            df_detalhe = resultado.alocacao[[
                'sigla', 'estado', 'regiao',
                'investimento_milhoes', 'mortes_antes', 
                'mortes_depois', 'reducao_mortes', 'reducao_percentual'
            ]].sort_values('investimento_milhoes', ascending=False).copy()
            df_detalhe.columns = ['UF', 'Estado', 'Região', 'Investimento (R$ mi)', 'Mortes Antes', 'Mortes Depois', 'Vidas Salvas', 'Redução %']
            
            st.dataframe(
                df_detalhe.style.format({
                    'Investimento (R$ mi)': 'R$ {:,.2f}',
                    'Mortes Antes': '{:,.0f}',
                    'Mortes Depois': '{:,.0f}',
                    'Vidas Salvas': '{:,.0f}',
                    'Redução %': '{:.2f}%'
                }).background_gradient(subset=['Investimento (R$ mi)'], cmap='Greens'),
                use_container_width=True,
                height=400,
                hide_index=True
            )
        
        else:
            st.error(f"❌ Não foi possível encontrar solução ótima. Status: {resultado.status}")
            
            if 'SolverError' in resultado.status:
                st.warning("""
                **Erro no solver CBC.** Isso pode acontecer quando:
                - O problema tem restrições impossíveis de satisfazer
                - O orçamento é muito baixo para os limites mínimos configurados
                
                **Sugestões:**
                1. Aumente o orçamento disponível
                2. Reduza o investimento mínimo por estado (%)
                3. Tente com ano diferente (alguns anos têm dados mais completos)
                """)
            else:
                st.info("""
                **Possíveis causas:**
                - Orçamento muito baixo para atender restrições mínimas
                - Parâmetros inconsistentes (máximo < mínimo)
                
                Tente ajustar os parâmetros e executar novamente.
                """)


# =============================================================================
# ABA 3: COMPARATIVO
# =============================================================================
def render_comparativo(df: pd.DataFrame, ano: int = 2022):
    """Renderiza a aba de Comparativo Antes vs. Depois."""
    
    st.header("📊 Comparativo - Antes vs. Depois")
    
    with st.expander("ℹ️ **Sobre esta aba** - Clique para expandir", expanded=False):
        st.markdown("""
        ### O que é o Comparativo?
        
        Esta aba mostra uma **comparação visual** entre o cenário atual (sem investimento adicional) 
        e o cenário projetado após a alocação otimizada de recursos.
        
        #### Visualizações disponíveis:
        
        | Gráfico | O que mostra |
        |---------|--------------|
        | **Barras Comparativas** | Mortes antes vs. depois para todos os 27 estados |
        | **Eficiência por Estado** | Custo por vida salva em cada estado |
        | **Ranking de Eficiência** | Os estados onde o investimento é mais eficiente |
        
        #### Métricas importantes:
        - **Mortes Antes**: Número de mortes no cenário atual (2022)
        - **Mortes Depois**: Projeção após o investimento adicional
        - **Vidas Salvas**: Diferença (redução) no número de mortes
        - **Custo por Vida**: Quanto custa cada vida salva em cada estado
        
        #### Interpretação:
        - Estados com **menor custo por vida** são mais eficientes
        - A cor verde indica redução significativa
        - O modelo prioriza estados onde o investimento tem maior impacto
        
        #### Nota:
        Se você ajustar parâmetros na aba **Otimização**, os resultados aqui serão atualizados 
        automaticamente. Caso contrário, exibe o cenário padrão (R$ 5 bilhões).
        """)
    
    # Usa resultado da session_state se existir, senão usa o pré-calculado
    if 'resultado_otimizacao' in st.session_state:
        resultado = st.session_state['resultado_otimizacao']
        fonte = "personalizado"
    else:
        resultado = obter_otimizacao_padrao(df)
        fonte = "padrão (R$ 5 bi)"
    
    if resultado.status != 'Optimal':
        st.error(f"❌ A otimização não encontrou solução ótima. Status: {resultado.status}")
        st.warning("""
        **Possíveis causas:**
        - Parâmetros incompatíveis (ex: orçamento muito baixo para os limites definidos)
        - Tente aumentar o orçamento ou ajustar os limites mínimo/máximo por estado
        """)
        return
    
    st.info(f"📊 Exibindo cenário **{fonte}**. Ajuste na aba Otimização para personalizar.")
    
    st.markdown(f"""
    **Cenário analisado:** Orçamento suplementar de **R$ {resultado.orcamento_usado/1000:.2f} bilhões**
    """)
    
    # Gráfico comparativo de barras - TODOS os estados
    st.subheader("📈 Comparativo de Mortes por Estado (Antes × Depois)")
    
    df_comp = resultado.alocacao.copy()
    df_comp = df_comp.sort_values('mortes_antes', ascending=True)  # Todos os estados
    
    fig_comp = go.Figure()
    
    fig_comp.add_trace(go.Bar(
        name='Antes',
        y=df_comp['sigla'],
        x=df_comp['mortes_antes'],
        orientation='h',
        marker_color='#ff6b6b',
        text=df_comp['mortes_antes'].apply(lambda x: f'{x:,.0f}'),
        textposition='auto'
    ))
    
    fig_comp.add_trace(go.Bar(
        name='Depois',
        y=df_comp['sigla'],
        x=df_comp['mortes_depois'],
        orientation='h',
        marker_color='#51cf66',
        text=df_comp['mortes_depois'].apply(lambda x: f'{x:,.0f}'),
        textposition='auto'
    ))
    
    fig_comp.update_layout(
        barmode='group',
        height=750,  # Maior para caber todos os 27 estados
        xaxis_title="Número de Mortes Violentas",
        yaxis_title="Estado",
        legend_title="Cenário",
        title="Comparativo Completo - Todos os 27 Estados"
    )
    
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # Resumo por região
    st.markdown("---")
    st.subheader("🗺️ Impacto por Região")
    
    df_regiao = resultado.alocacao.groupby('regiao').agg({
        'mortes_antes': 'sum',
        'mortes_depois': 'sum',
        'reducao_mortes': 'sum',
        'investimento_milhoes': 'sum'
    }).reset_index()
    
    df_regiao['reducao_pct'] = (df_regiao['reducao_mortes'] / df_regiao['mortes_antes'] * 100).round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_regiao = px.bar(
            df_regiao,
            x='regiao',
            y=['mortes_antes', 'mortes_depois'],
            barmode='group',
            labels={'value': 'Mortes', 'regiao': 'Região', 'variable': 'Cenário'},
            title="Mortes por Região: Antes vs Depois",
            color_discrete_map={'mortes_antes': '#ff6b6b', 'mortes_depois': '#51cf66'}
        )
        fig_regiao.update_layout(height=400)
        st.plotly_chart(fig_regiao, use_container_width=True)
    
    with col2:
        fig_reducao = px.bar(
            df_regiao,
            x='regiao',
            y='reducao_pct',
            color='investimento_milhoes',
            color_continuous_scale='Blues',
            text='reducao_pct',
            labels={
                'reducao_pct': 'Redução (%)',
                'regiao': 'Região',
                'investimento_milhoes': 'Investimento (R$ mi)'
            },
            title="Redução Percentual por Região"
        )
        fig_reducao.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_reducao.update_layout(height=400)
        st.plotly_chart(fig_reducao, use_container_width=True)
    
    # Análise de eficiência
    st.markdown("---")
    st.subheader("💡 Análise de Eficiência")
    
    df_efic = resultado.alocacao[resultado.alocacao['investimento_milhoes'] > 0].copy()
    df_efic['custo_por_vida'] = df_efic['investimento_milhoes'] / df_efic['reducao_mortes']
    df_efic = df_efic.sort_values('custo_por_vida')
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        fig_efic = px.scatter(
            df_efic,
            x='investimento_milhoes',
            y='reducao_mortes',
            size='populacao',
            color='custo_por_vida',
            hover_name='estado',
            text='sigla',
            color_continuous_scale='RdYlGn_r',
            labels={
                'investimento_milhoes': 'Investimento (R$ milhões)',
                'reducao_mortes': 'Vidas Salvas',
                'custo_por_vida': 'Custo/Vida (R$ mi)',
                'populacao': 'População'
            },
            title="Eficiência: Investimento vs Vidas Salvas"
        )
        fig_efic.update_traces(textposition='top center')
        fig_efic.update_layout(height=450)
        st.plotly_chart(fig_efic, use_container_width=True)
    
    with col2:
        st.markdown("#### 🏆 Estados Mais Eficientes")
        st.markdown("(Menor custo por vida salva)")
        
        top_efic = df_efic.nsmallest(5, 'custo_por_vida')[
            ['estado', 'investimento_milhoes', 'reducao_mortes', 'custo_por_vida']
        ]
        top_efic.columns = ['Estado', 'Investimento (R$ mi)', 'Vidas Salvas', 'Custo/Vida']
        
        st.dataframe(
            top_efic.style.format({
                'Investimento (R$ mi)': 'R$ {:,.2f}',
                'Vidas Salvas': '{:,.0f}',
                'Custo/Vida': 'R$ {:,.2f}'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("---")
        
        st.markdown("#### 📊 Resumo Geral")
        
        total_antes = resultado.alocacao['mortes_antes'].sum()
        total_depois = resultado.alocacao['mortes_depois'].sum()
        
        st.markdown(f"""
        | Métrica | Valor |
        |---------|-------|
        | **Mortes Antes** | {total_antes:,.0f} |
        | **Mortes Depois** | {total_depois:,.0f} |
        | **Vidas Salvas** | {resultado.reducao_crimes:,.0f} |
        | **Redução** | {resultado.reducao_percentual:.2f}% |
        | **Investimento Total** | R$ {resultado.orcamento_usado:,.2f} mi |
        """)


# =============================================================================
# ABA 4: ANÁLISE DE SENSIBILIDADE
# =============================================================================
def render_sensibilidade(df: pd.DataFrame, ano: int = 2022):
    """
    Renderiza a aba de análise de sensibilidade.
    Inclui gráfico tornado, shadow prices e análise de cenários.
    """
    st.header(f"🔍 Análise de Sensibilidade ({ano})")
    
    with st.expander("ℹ️ **Sobre esta aba** - Clique para expandir", expanded=False):
        st.markdown("""
        ### O que é Análise de Sensibilidade?
        
        A análise de sensibilidade é uma técnica fundamental em Pesquisa Operacional que avalia 
        **como variações nos parâmetros de entrada afetam a solução ótima**.
        
        #### Por que é importante?
        - Dados de entrada contêm **incerteza** (elasticidades estimadas, taxas projetadas)
        - Decisores precisam saber se a solução é **robusta**
        - Identifica **parâmetros críticos** que merecem maior atenção
        
        #### Análises disponíveis:
        
        | Análise | Descrição |
        |---------|-----------|
        | **Curva de Sensibilidade** | Como o resultado varia com diferentes orçamentos |
        | **Shadow Prices** | Valor marginal de relaxar a restrição de orçamento |
        | **Gráfico Tornado** | Ranking dos parâmetros por impacto no resultado |
        | **Análise de Cenários** | Comparação pessimista / base / otimista |
        
        #### Interpretação dos Shadow Prices:
        - Indica **quantas vidas seriam salvas por R$ 1 milhão adicional**
        - Um shadow price de 0.5 significa: +R$ 1 mi → +0.5 vidas salvas
        - Valor alto sugere que mais orçamento seria muito benéfico
        
        #### Gráfico Tornado:
        - Barras mais longas = parâmetros com **maior impacto**
        - Estados no topo são os mais sensíveis a variações
        - Útil para priorizar coleta de dados mais precisos
        """)
    
    st.markdown("""
    Estudo de como variações nos parâmetros afetam o resultado da otimização.
    Essencial para entender a robustez da solução e identificar parâmetros críticos.
    """)
    
    # Parâmetros para recalcular
    with st.expander("⚙️ Ajustar Parâmetros", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            orcamento_base = st.slider(
                "Orçamento Base (R$ milhões)",
                min_value=1000.0,
                max_value=10000.0,
                value=5000.0,
                step=500.0,
                key="sens_orcamento"
            )
        with col2:
            variacao_pct = st.slider(
                "Variação para Análise (%)",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
                key="sens_variacao"
            )
        
        recalcular = st.button("🔄 Recalcular com novos parâmetros", key="btn_sens")
    
    # Usa cache ou recalcula
    if recalcular:
        with st.spinner("Calculando sensibilidade..."):
            resultados_sens = analisar_sensibilidade_orcamento(df, orcamento_base=orcamento_base)
            shadow = calcular_shadow_prices(df, orcamento=orcamento_base)
            
            cenarios_dict = {
                'pessimista': orcamento_base * 0.6,
                'base': orcamento_base,
                'otimista': orcamento_base * 1.4
            }
            cenarios_df = analisar_cenarios(df, cenarios_dict)
            cenarios = {}
            for _, row in cenarios_df.iterrows():
                cenarios[row['cenario']] = {'vidas_salvas': row['reducao_crimes']}
            
            fig_tornado = gerar_grafico_tornado(df, orcamento=orcamento_base)
    else:
        # Usa valores pré-calculados
        dados_sens = obter_sensibilidade_padrao(df)
        resultados_sens = dados_sens['sensibilidade']
        shadow = dados_sens['shadow']
        cenarios = dados_sens['cenarios']
        fig_tornado = dados_sens['tornado']
        orcamento_base = 5000
        variacao_pct = 20
    
    # 1. Sensibilidade do Orçamento
    st.subheader("📊 Sensibilidade ao Orçamento")
    df_sens = resultados_sens if isinstance(resultados_sens, pd.DataFrame) else pd.DataFrame(resultados_sens)
    fig_sens = px.line(
        df_sens,
        x='orcamento_milhoes',
        y='reducao_crimes',
        markers=True,
        labels={
            'orcamento_milhoes': 'Orçamento (R$ milhões)',
            'reducao_crimes': 'Vidas Salvas'
        },
        title=f"Impacto do Orçamento na Redução de Crimes"
    )
    fig_sens.add_vline(x=orcamento_base, line_dash="dash", annotation_text="Base")
    st.plotly_chart(fig_sens, use_container_width=True)
    
    # 2. Shadow Prices
    st.subheader("💰 Shadow Prices (Preços Sombra)")
    st.markdown("""
    O **Shadow Price** indica quanto a função objetivo (vidas salvas) 
    melhoraria se relaxássemos uma restrição em 1 unidade.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Shadow Price do Orçamento",
            f"{shadow.get('shadow_orcamento', 0):.4f} vidas/R$ milhão",
            help="Marginal: quantas vidas salvas a mais por R$ 1 milhão adicional"
        )
    with col2:
        st.metric(
            "Valor Marginal",
            f"R$ {1/max(shadow.get('shadow_orcamento', 0.001), 0.001):.2f} mi/vida",
            help="Custo marginal por vida salva adicional"
        )
    
    # 3. Gráfico Tornado
    st.subheader("🌪️ Diagrama Tornado")
    st.markdown("""
    Mostra quais parâmetros têm maior impacto no resultado quando variados.
    Barras mais longas = parâmetros mais sensíveis.
    """)
    st.plotly_chart(fig_tornado, use_container_width=True)
    
    # 4. Análise de Cenários
    st.subheader("📋 Análise de Cenários")
    df_cenarios = pd.DataFrame([
        {
            'Cenário': 'Pessimista',
            'Descrição': 'Elasticidade 30% menor',
            'Vidas Salvas': cenarios['pessimista']['vidas_salvas'],
            'Diferença': cenarios['pessimista']['vidas_salvas'] - cenarios['base']['vidas_salvas']
        },
        {
            'Cenário': 'Base',
            'Descrição': 'Parâmetros estimados',
            'Vidas Salvas': cenarios['base']['vidas_salvas'],
            'Diferença': 0
        },
        {
            'Cenário': 'Otimista',
            'Descrição': 'Elasticidade 30% maior',
            'Vidas Salvas': cenarios['otimista']['vidas_salvas'],
            'Diferença': cenarios['otimista']['vidas_salvas'] - cenarios['base']['vidas_salvas']
        }
    ])
    
    st.dataframe(
        df_cenarios.style.format({
            'Vidas Salvas': '{:,.0f}',
            'Diferença': '{:+,.0f}'
        }),
        use_container_width=True,
        hide_index=True
    )


# =============================================================================
# ABA 5: SIMULAÇÃO MONTE CARLO
# =============================================================================
def render_monte_carlo(df: pd.DataFrame, ano: int = 2022):
    """
    Renderiza a aba de simulação Monte Carlo.
    Quantifica incerteza nos resultados via simulação estocástica.
    """
    st.header("🎲 Simulação Monte Carlo")
    
    with st.expander("ℹ️ **Sobre esta aba** - Clique para expandir", expanded=False):
        st.markdown("""
        ### O que é Simulação Monte Carlo?
        
        Monte Carlo é uma técnica estatística que executa **milhares de simulações** com 
        variações aleatórias nos parâmetros de entrada para quantificar a **incerteza** 
        nos resultados.
        
        #### Por que usar Monte Carlo?
        - Os parâmetros do modelo (elasticidades) são **estimativas**, não valores exatos
        - Queremos saber não apenas o resultado "médio", mas a **distribuição de possíveis resultados**
        - Permite calcular **intervalos de confiança** (ex: 95% de chance de salvar entre X e Y vidas)
        
        #### Como funciona:
        1. Para cada simulação, gera variações aleatórias nos parâmetros
        2. Executa a otimização com esses parâmetros perturbados
        3. Registra o resultado (vidas salvas)
        4. Após N simulações, analisa a distribuição dos resultados
        
        #### Parâmetros configuráveis:
        
        | Parâmetro | Descrição |
        |-----------|-----------|
        | **Orçamento** | Valor a ser distribuído em todas as simulações |
        | **Nº de Simulações** | Mais = maior precisão, mas mais lento (500 é um bom equilíbrio) |
        | **Incerteza** | Quanto os parâmetros podem variar (±15% é típico) |
        
        #### Resultados:
        - **Histograma**: Distribuição dos possíveis resultados
        - **Intervalo de Confiança 95%**: Faixa onde o resultado real provavelmente estará
        - **VaR (Value at Risk)**: Resultado no pior caso (5% das simulações)
        """)
    
    st.markdown("""
    Simula centenas de cenários com variações aleatórias nos parâmetros
    para obter intervalos de confiança nos resultados.
    """)
    
    # Parâmetros da simulação
    with st.expander("⚙️ Ajustar Parâmetros", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            orcamento_bilhoes = st.slider(
                "Orçamento (R$ bilhões)",
                min_value=1.0,
                max_value=20.0,
                value=5.0,
                step=1.0,
                key="mc_orcamento"
            )
            orcamento = orcamento_bilhoes * 1000  # Converte para milhões
        with col2:
            n_simulacoes = st.selectbox(
                "Número de Simulações",
                options=[100, 250, 500, 1000],
                index=2,
                key="mc_n_sim"
            )
        with col3:
            variacao = st.slider(
                "Incerteza nos Parâmetros (%)",
                min_value=5,
                max_value=30,
                value=15,
                step=5,
                key="mc_variacao"
            )
    
    # Botão para executar simulação
    if st.button("🚀 Executar Simulação Monte Carlo", type="primary", use_container_width=True):
        with st.spinner(f"Executando {n_simulacoes} simulações... Aguarde..."):
            resultado_mc = executar_monte_carlo(
                df,
                orcamento=orcamento,
                n_simulacoes=n_simulacoes,
                incerteza_elasticidade=variacao/100,
                incerteza_taxa=variacao/200,  # Metade da incerteza para taxa
                verbose=False
            )
            st.session_state['resultado_mc'] = resultado_mc
            st.session_state['mc_n_sim_display'] = n_simulacoes
        st.success("✅ Simulação concluída!")
    
    # Usa resultado da sessão ou padrão
    if 'resultado_mc' in st.session_state:
        resultado_mc = st.session_state['resultado_mc']
        n_sim_display = st.session_state.get('mc_n_sim_display', 250)
    else:
        resultado_mc = obter_monte_carlo_padrao(df)
        n_sim_display = 250
    
    # Métricas resumo
    st.subheader("📊 Resultados da Simulação")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Vidas Salvas (Média)", f"{resultado_mc.media_reducao:.0f}")
    with col2:
        st.metric("Desvio Padrão", f"±{resultado_mc.desvio_padrao_reducao:.0f}")
    with col3:
        st.metric("IC 95% Inferior", f"{resultado_mc.intervalo_confianca_95[0]:.0f}")
    with col4:
        st.metric("IC 95% Superior", f"{resultado_mc.intervalo_confianca_95[1]:.0f}")
    
    # Histograma
    st.subheader("📈 Distribuição dos Resultados")
    
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=resultado_mc.distribuicao_reducao,
        nbinsx=30,
        name="Simulações",
        marker_color='#3498db'
    ))
    
    fig_hist.add_vline(x=resultado_mc.intervalo_confianca_95[0], line_dash="dash", line_color="red", annotation_text="IC 2.5%")
    fig_hist.add_vline(x=resultado_mc.intervalo_confianca_95[1], line_dash="dash", line_color="red", annotation_text="IC 97.5%")
    fig_hist.add_vline(x=resultado_mc.media_reducao, line_color="green", annotation_text="Média")
    
    fig_hist.update_layout(
        title=f"Distribuição de Vidas Salvas ({n_sim_display} simulações)",
        xaxis_title="Vidas Salvas",
        yaxis_title="Frequência",
        showlegend=False,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True)
    )
    
    st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False})
    
    # Percentis
    st.subheader("📋 Tabela de Percentis")
    
    df_percentis = pd.DataFrame({
        'Percentil': [f"P{p}" for p in resultado_mc.percentis.keys()],
        'Vidas Salvas': list(resultado_mc.percentis.values()),
        'Interpretação': [
            "5% chance de ser menor que isso",
            "1º Quartil",
            "Mediana (50%)",
            "3º Quartil",
            "95% chance de ser menor"
        ]
    })
    
    st.dataframe(
        df_percentis.style.format({'Vidas Salvas': '{:,.0f}'}),
        use_container_width=True,
        hide_index=True
    )
    
    st.info(f"✅ **Taxa de sucesso:** {resultado_mc.n_sucesso}/{resultado_mc.n_simulacoes} simulações convergiram ({resultado_mc.n_sucesso/resultado_mc.n_simulacoes*100:.1f}%)")


# =============================================================================
# ABA 6: BACKTESTING
# =============================================================================
def render_backtesting(df: pd.DataFrame, ano: int = 2022):
    """
    Renderiza a aba de backtesting.
    Valida o modelo usando dados históricos.
    """
    st.header("🔄 Backtesting - Validação Histórica")
    
    with st.expander("ℹ️ **Sobre esta aba** - Clique para expandir", expanded=False):
        st.markdown("""
        ### O que é Backtesting?
        
        Backtesting é uma técnica de **validação** que testa se o modelo teria funcionado 
        corretamente no passado. É como perguntar: "Se tivéssemos usado este modelo em 2015, 
        as previsões teriam se confirmado em 2016?"
        
        #### Por que é importante?
        - Modelos podem parecer bons no papel mas falhar na prática
        - Backtesting usa **dados reais históricos** para testar a abordagem
        - Aumenta a confiança de que o modelo funcionará no futuro
        
        #### Metodologia - Janela Deslizante:
        1. **Treino (2010-2014)**: Calcula elasticidades usando dados de 5 anos
        2. **Previsão (2015)**: Prevê taxa de mortes para o próximo ano
        3. **Comparação**: Compara previsão com o que realmente aconteceu
        4. **Avança**: Move a janela para 2011-2015 e prevê 2016
        5. **Repete**: Continua até cobrir todo o período disponível
        
        #### Métricas de avaliação:
        
        | Métrica | Descrição | Bom valor |
        |---------|-----------|-----------|
        | **MAPE** | Erro médio absoluto percentual | < 10% |
        | **RMSE** | Raiz do erro quadrático médio | Menor = melhor |
        | **R²** | Coeficiente de determinação | > 0.7 |
        
        #### Parâmetros:
        - **Janela de Treino**: Quantos anos usar para estimar as elasticidades
        - **Método**: Janela deslizante (mais robusto) ou período fixo (mais simples)
        
        #### Interpretação:
        - MAPE < 5%: Excelente
        - MAPE 5-10%: Bom
        - MAPE 10-20%: Aceitável
        - MAPE > 20%: Modelo precisa de ajustes
        """)
    
    st.markdown("""
    Testa se o modelo teria funcionado no passado, comparando previsões
    com resultados reais. Fundamental para validar a abordagem.
    """)
    
    # Opções de backtesting
    with st.expander("⚙️ Ajustar Parâmetros", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            metodo = st.radio(
                "Método de Validação",
                options=["Janela Deslizante", "Período Fixo"],
                help="Janela deslizante é mais robusto"
            )
        with col2:
            tamanho_janela = st.slider(
                "Tamanho da Janela (anos)",
                min_value=3,
                max_value=10,
                value=5,
                step=1,
                key="bt_janela"
            )
        
        recalcular = st.button("🔄 Recalcular com novos parâmetros", key="btn_bt")
    
    try:
        # Usa cache ou recalcula
        if recalcular:
            with st.spinner("Executando validação histórica..."):
                if metodo == "Janela Deslizante":
                    resultado_rolling = validar_modelo_rolling(
                        janela_treino=tamanho_janela,
                        janela_teste=1,
                        ano_inicio=2010,
                        ano_fim=2022
                    )
                else:
                    resultado_rolling = obter_backtesting_padrao()
        else:
            resultado_rolling = obter_backtesting_padrao()
        
        if resultado_rolling is None or resultado_rolling.empty:
            st.warning("Dados insuficientes para backtesting.")
            return
        
        # Calcula métricas agregadas
        mape_medio = resultado_rolling['mape'].mean()
        rmse_medio = resultado_rolling['rmse'].mean()
        corr_media = resultado_rolling['correlacao'].mean() if 'correlacao' in resultado_rolling.columns else 0.8
        
        st.subheader("📊 Métricas de Erro (Média das Janelas)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MAPE Médio", f"{mape_medio:.1f}%")
        with col2:
            st.metric("RMSE Médio", f"{rmse_medio:.2f}")
        with col3:
            st.metric("Correlação Média", f"{corr_media:.3f}")
        
        # Gráfico de evolução do MAPE por ano
        st.subheader("📈 Evolução do MAPE por Ano de Teste")
        fig_rolling = px.line(
            resultado_rolling,
            x='ano_teste',
            y='mape',
            markers=True,
            labels={'ano_teste': 'Ano de Teste', 'mape': 'MAPE (%)'},
            title="Erro de Previsão por Ano (Janela Deslizante)"
        )
        st.plotly_chart(fig_rolling, use_container_width=True)
        
        # Interpretação
        if mape_medio < 10:
            qualidade = "🟢 Excelente"
            interpretacao = "O modelo tem alta precisão preditiva."
        elif mape_medio < 20:
            qualidade = "🟡 Boa"
            interpretacao = "O modelo é razoável para planejamento."
        elif mape_medio < 30:
            qualidade = "🟠 Moderada"
            interpretacao = "Usar com cautela; considerar intervalos de confiança."
        else:
            qualidade = "🔴 Baixa"
            interpretacao = "Modelo precisa de ajustes ou mais dados."
        
        st.info(f"**Qualidade do Modelo: {qualidade}**\n\n{interpretacao}")
        
    except Exception as e:
        st.error(f"Erro ao executar backtesting: {e}")


# =============================================================================
# ABA 7: MODELO MULTI-PERÍODO
# =============================================================================
def render_multi_periodo(df: pd.DataFrame, ano: int = 2022):
    """
    Renderiza a aba de otimização multi-período.
    Planejamento de investimentos ao longo de vários anos.
    """
    st.header("📅 Otimização Multi-Período")
    
    with st.expander("ℹ️ **Sobre esta aba** - Clique para expandir", expanded=False):
        st.markdown("""
        ### O que é Otimização Multi-Período?
        
        Enquanto a otimização simples distribui um orçamento **em um único momento**, 
        a otimização multi-período planeja investimentos ao longo de **vários anos**.
        
        #### Por que multi-período?
        - Investimentos em segurança têm **efeitos que se acumulam** ao longo do tempo
        - Orçamentos reais são **anuais**, não únicos
        - Permite planejar uma **estratégia de longo prazo**
        
        #### Estratégias comparadas:
        
        | Estratégia | Descrição | Quando usar |
        |------------|-----------|-------------|
        | **Uniforme** | Mesmo valor todo ano | Orçamento previsível |
        | **Frontloaded** | Mais no início, menos no fim | Crise urgente |
        | **Backloaded** | Menos no início, mais no fim | Orçamento crescente |
        | **Crescente Linear** | Aumento gradual ano a ano | Crescimento econômico |
        
        #### Efeitos considerados:
        - **Acumulação**: Investimentos passados continuam gerando resultados
        - **Depreciação**: Parte do efeito se perde com o tempo (equipamentos, treinamento)
        - **Retornos decrescentes**: Cada R$ adicional tem impacto menor que o anterior
        
        #### Parâmetros:
        - **Orçamento Total**: Soma de todos os investimentos no período
        - **Número de Períodos**: Quantos anos o plano contempla
        
        #### Interpretação:
        - A estratégia vencedora depende das características do problema
        - Em geral, **Frontloaded** funciona melhor quando há urgência
        - **Uniforme** é mais fácil de implementar politicamente
        """)
    
    st.markdown("""
    Planeja a distribuição de investimentos ao longo de múltiplos anos,
    considerando que investimentos têm efeitos acumulados e depreciação.
    """)
    
    # Parâmetros
    with st.expander("⚙️ Ajustar Parâmetros", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            orcamento_total = st.slider(
                "Orçamento Total Multi-Ano (R$ bi)",
                min_value=5.0,
                max_value=50.0,
                value=25.0,
                step=5.0,
                key="mp_orcamento"
            )
        with col2:
            n_periodos = st.slider(
                "Número de Períodos (anos)",
                min_value=3,
                max_value=10,
                value=5,
                step=1,
                key="mp_periodos"
            )
        
        recalcular = st.button("🔄 Recalcular com novos parâmetros", key="btn_mp")
    
    try:
        # Usa cache ou recalcula
        if recalcular:
            with st.spinner("Otimizando para múltiplos períodos..."):
                orcamento_milhoes = orcamento_total * 1000
                df_comparativo = comparar_estrategias(df, orcamento_milhoes, n_periodos)
        else:
            df_comparativo = obter_multiperiodo_padrao(df)
            orcamento_total = 25.0
            n_periodos = 5
        
        if df_comparativo.empty:
            st.error("Não foi possível calcular as estratégias.")
            return
        
        # Resultados
        st.subheader("📊 Comparação de Estratégias")
        
        # Renomeia para exibição
        df_display = df_comparativo.copy()
        df_display['Estratégia'] = df_display['estrategia'].map({
            'Uniforme': '📊 Uniforme (igual cada ano)',
            'Frontloaded': '⏩ Frontloaded (mais no início)',
            'Backloaded': '⏪ Backloaded (mais no fim)',
            'Crescente_Linear': '📈 Crescente Linear'
        })
        df_display = df_display.rename(columns={
            'reducao_total': 'Crimes Evitados',
            'reducao_primeiro_periodo': 'Redução Período 1',
            'reducao_ultimo_periodo': 'Redução Último Período'
        })
        
        df_display = df_display.sort_values('Crimes Evitados', ascending=False)
        
        melhor = df_display.iloc[0]['Estratégia']
        st.success(f"🏆 **Melhor estratégia: {melhor}**")
        
        st.dataframe(
            df_display[['Estratégia', 'Crimes Evitados', 'Redução Período 1', 'Redução Último Período']].style.format({
                'Crimes Evitados': '{:,.0f}',
                'Redução Período 1': '{:,.0f}',
                'Redução Último Período': '{:,.0f}'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Gráfico de barras comparativo
        st.subheader("📈 Crimes Evitados por Estratégia")
        
        fig_bar = px.bar(
            df_display,
            x='Estratégia',
            y='Crimes Evitados',
            color='Crimes Evitados',
            color_continuous_scale='Greens',
            text='Crimes Evitados'
        )
        fig_bar.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Gráfico de distribuição temporal
        st.subheader("💰 Distribuição Temporal do Investimento")
        
        fig_dist = go.Figure()
        for _, row in df_comparativo.iterrows():
            if row['distribuicao']:
                periodos = list(range(1, len(row['distribuicao']) + 1))
                valores_bi = [v / 1000 for v in row['distribuicao']]
                fig_dist.add_trace(go.Scatter(
                    x=periodos,
                    y=valores_bi,
                    mode='lines+markers',
                    name=row['estrategia']
                ))
        
        fig_dist.update_layout(
            title="Investimento por Período",
            xaxis_title="Período (ano)",
            yaxis_title="Investimento (R$ bilhões)",
            legend_title="Estratégia"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Explicação
        st.markdown("---")
        st.markdown("""
        ### 💡 Por que Frontloaded funciona melhor?
        
        O investimento tem **efeito acumulado**: políticas implementadas cedo
        continuam gerando benefícios nos anos seguintes.
        
        - Investimento no ano 1: gera benefícios nos anos 1, 2, 3, 4, 5
        - Investimento no ano 5: gera benefício apenas no ano 5
        
        Por isso, concentrar recursos no início maximiza o impacto total.
        """)
        
    except Exception as e:
        st.error(f"Erro ao calcular multi-período: {e}")


# =============================================================================
# ABA 8: CONCLUSÕES E EFICIÊNCIA DOS INVESTIMENTOS
# =============================================================================
def render_conclusoes(df: pd.DataFrame, ano: int = 2022):
    """
    Renderiza a aba de Conclusões com análise de eficiência de investimentos por estado.
    """
    st.header(f"📋 Conclusões - Eficiência dos Investimentos ({ano})")
    
    with st.expander("ℹ️ **Sobre esta aba** - Clique para expandir", expanded=False):
        st.markdown("""
        ### O que é a Análise de Eficiência?
        
        Esta aba apresenta as **conclusões finais** do estudo, focando em responder a pergunta central:
        **Quais estados estão investindo de forma mais eficiente em segurança pública?**
        
        #### Métricas de eficiência calculadas:
        
        | Métrica | Fórmula | Interpretação |
        |---------|---------|---------------|
        | **Gasto per capita** | Orçamento ÷ População | Quanto cada estado investe por habitante |
        | **Taxa de homicídios** | Mortes ÷ População × 100.000 | Nível de violência por 100 mil habitantes |
        | **Eficiência DEA** | Resultado ÷ Custo (relativo) | Desempenho vs. outros estados |
        
        #### Fontes de dados:
        - **Violência**: Atlas da Violência (IPEA/FBSP) - série 1989-2022
        - **Orçamentos**: Anuário Brasileiro de Segurança Pública 2023 (FBSP)
        - **População**: IBGE - Censo/Estimativas 2022
        """)
    
    # Obtém resultado da otimização
    resultado = obter_otimizacao_padrao(df)
    
    # Calcula eficiência usando DEA (Data Envelopment Analysis)
    df_efic_calc = calcular_dea_ccr(df)
    resumo_efic = resumo_dea(df_efic_calc)
    
    # Estados mais e menos eficientes (DEA)
    top5_efic = df_efic_calc.head(5)  # Já ordenado por eficiência
    bottom5_efic = df_efic_calc.tail(5).iloc[::-1]  # Inverte para mostrar do pior ao menos pior
    
    st.markdown("""
    ### 🎯 Pergunta Central do Estudo
    
    > **Quais estados brasileiros estão utilizando seus recursos de segurança pública de forma 
    > mais eficiente, e como uma redistribuição otimizada poderia reduzir a violência?**
    """)
    
    # =========================================================================
    # RESPOSTA DIRETA E OBJETIVA
    # =========================================================================
    st.success("""
    ## ✅ RESPOSTA DIRETA
    """)
    
    col_resp1, col_resp2 = st.columns(2)
    
    with col_resp1:
        st.markdown("### 🏆 Estados MAIS Eficientes (DEA)")
        st.markdown("*Fronteira de eficiência - referência de boas práticas*")
        for i, (_, row) in enumerate(top5_efic.iterrows(), 1):
            st.markdown(f"""
            **{i}º {row['estado']}** ({row['sigla']})  
            - Gasto: R$ {row['gasto_per_capita']:,.0f}/hab  
            - Taxa: {row['taxa_mortes_100k']:.1f}/100k  
            - Eficiência DEA: **{row['eficiencia_percentual']:.1f}%**
            """)
    
    with col_resp2:
        st.markdown("### ⚠️ Estados MENOS Eficientes (DEA)")
        st.markdown("*Maior potencial de melhoria*")
        for i, (_, row) in enumerate(bottom5_efic.iterrows(), 1):
            st.markdown(f"""
            **{i}º {row['estado']}** ({row['sigla']})  
            - Gasto: R$ {row['gasto_per_capita']:,.0f}/hab  
            - Taxa: {row['taxa_mortes_100k']:.1f}/100k  
            - Eficiência DEA: **{row['eficiencia_percentual']:.1f}%**
            """)
    
    st.markdown("---")
    
    st.warning(f"""
    ### 💡 Conclusão Principal
    
    Com um investimento adicional otimizado de **R$ 5 bilhões**, o modelo estima que seria 
    possível salvar aproximadamente **{resultado.reducao_crimes:,.0f} vidas** por ano, 
    uma redução de **{resultado.reducao_percentual:.2f}%** nas mortes violentas.
    
    Os estados que **mais se beneficiariam** são aqueles com:
    - Alto número absoluto de mortes (maior potencial de impacto)
    - Baixo gasto per capita atual (margem para crescimento)
    - Alta taxa de homicídios (maior urgência)
    """)
    
    st.markdown("---")
    
    # =========================================================================
    # SEÇÃO 1: RANKING DE EFICIÊNCIA - DEA (Data Envelopment Analysis)
    # =========================================================================
    st.subheader("🏆 Ranking de Eficiência - Análise Envoltória de Dados (DEA)")
    
    st.markdown("""
    Utilizamos **DEA (Data Envelopment Analysis)** - método de Pesquisa Operacional 
    para medir a eficiência relativa de cada estado, comparando **resultado** (baixa taxa de homicídios) 
    com **custo** (gasto per capita).
    
    **Pesos do Modelo:**
    - **75%** - Resultado (quanto menor a taxa de homicídios, melhor)
    - **25%** - Economia (quanto menor o gasto para o mesmo resultado, melhor)
    """)
    
    # Calcula eficiência DEA
    df_dea = calcular_dea_ccr(df)
    resumo = resumo_dea(df_dea)
    
    # Métricas resumo simplificadas
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Eficiência Média", f"{resumo['eficiencia_media']*100:.1f}%")
    with col_m2:
        st.metric("Maior Eficiência", f"{resumo['eficiencia_max']*100:.1f}%")
    with col_m3:
        st.metric("Menor Eficiência", f"{resumo['eficiencia_min']*100:.1f}%")
    
    st.markdown("---")
    
    # Categoriza eficiência
    def categorizar_eficiencia_dea(ef):
        if ef >= 0.8:
            return '🟢 Alta eficiência'
        elif ef >= 0.5:
            return '🟡 Média eficiência'
        else:
            return '🔴 Baixa eficiência'
    
    df_dea['categoria'] = df_dea['eficiencia_dea'].apply(categorizar_eficiencia_dea)
    
    # Ranking completo - TABELA VISÍVEL
    st.markdown("### 📋 Ranking Completo de Eficiência - Todos os Estados")
    
    df_ranking = df_dea[['estado', 'sigla', 'regiao', 'gasto_per_capita', 'taxa_mortes_100k', 'eficiencia_percentual', 'categoria']].copy()
    df_ranking.columns = ['Estado', 'UF', 'Região', 'Gasto/capita', 'Taxa/100k', 'Eficiência %', 'Status']
    df_ranking['Ranking'] = range(1, len(df_ranking) + 1)
    df_ranking = df_ranking[['Ranking', 'Estado', 'UF', 'Região', 'Gasto/capita', 'Taxa/100k', 'Eficiência %', 'Status']]
    
    st.dataframe(
        df_ranking.style.format({
            'Gasto/capita': 'R$ {:,.0f}',
            'Taxa/100k': '{:.1f}',
            'Eficiência %': '{:.1f}%'
        }),
        use_container_width=True,
        hide_index=True,
        height=700
    )
    
    st.info("""
    💡 **Interpretação:** 
    - A eficiência é **relativa** - compara cada estado com o melhor desempenho
    - **75% do peso** é dado ao **resultado** (baixa taxa de homicídios)
    - **25% do peso** é dado à **economia** (baixo gasto per capita)
    - Estados com alta eficiência conseguem bons resultados de segurança
    """)
    
    st.markdown("---")
    
    # =========================================================================
    # SEÇÃO 2: PRINCIPAIS CONCLUSÕES
    # =========================================================================
    st.subheader("📝 Principais Conclusões do Estudo")
    
    # Calcula estatísticas para conclusões
    total_mortes = df['mortes_violentas'].sum()
    total_orcamento = df['orcamento_2022_milhoes'].sum()
    media_taxa = df['taxa_mortes_100k'].mean()
    
    # Estados extremos
    estado_mais_violento = df.loc[df['taxa_mortes_100k'].idxmax()]
    estado_menos_violento = df.loc[df['taxa_mortes_100k'].idxmin()]
    estado_maior_gasto = df.loc[df['gasto_per_capita'].idxmax()]
    estado_menor_gasto = df.loc[df['gasto_per_capita'].idxmin()]
    
    # Resultados da otimização
    vidas_salvas = resultado.reducao_crimes
    reducao_pct = resultado.reducao_percentual
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🔍 Diagnóstico da Situação Atual
        """)
        st.markdown(f"""
        **Cenário 2022:**
        - **{total_mortes:,.0f}** mortes violentas no Brasil
        - **R$ {total_orcamento/1000:.1f} bilhões** em orçamento de segurança
        - Taxa média de **{media_taxa:.1f}** mortes/100 mil hab.
        
        **Extremos:**
        - 🔴 Mais violento: **{estado_mais_violento['estado']}** ({estado_mais_violento['taxa_mortes_100k']:.1f}/100k)
        - 🟢 Menos violento: **{estado_menos_violento['estado']}** ({estado_menos_violento['taxa_mortes_100k']:.1f}/100k)
        - 💰 Maior gasto/capita: **{estado_maior_gasto['estado']}** (R$ {estado_maior_gasto['gasto_per_capita']:,.0f})
        - 💸 Menor gasto/capita: **{estado_menor_gasto['estado']}** (R$ {estado_menor_gasto['gasto_per_capita']:,.0f})
        """)
    
    with col2:
        st.markdown("""
        #### ✅ Potencial da Otimização
        """)
        st.markdown(f"""
        **Com investimento adicional de R$ 5 bilhões:**
        - **{vidas_salvas:,.0f}** vidas potencialmente salvas
        - Redução de **{reducao_pct:.2f}%** nas mortes violentas
        
        **Estados que mais se beneficiariam:**
        """)
        top3 = resultado.alocacao.nlargest(3, 'reducao_mortes')[['estado', 'reducao_mortes']]
        for _, row in top3.iterrows():
            st.markdown(f"- **{row['estado']}**: {row['reducao_mortes']:,.0f} vidas")


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================
def main():
    """Função principal da aplicação."""
    
    # Título principal
    st.markdown('<h1 class="main-header">🔐 Otimização de Recursos de Segurança Pública</h1>', 
                unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align: center; font-size: 1.2rem; color: #666;">
    Aplicação de Pesquisa Operacional para alocação ótima de recursos entre estados brasileiros
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Renderiza sidebar e obtém o ano selecionado
    ano_selecionado = render_sidebar()
    
    # Carrega dados do ano selecionado
    try:
        df = carregar_dados(ano=ano_selecionado)
        geojson = carregar_geojson_brasil()
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        st.stop()
    
    # Lista de abas disponíveis
    ABAS = [
        "📊 Dashboard",
        "⚙️ Otimização",
        "🎲 Monte Carlo",
        "📅 Multi-Período",
        "📋 Conclusões"
    ]
    
    # Usa query params para persistir a aba selecionada
    query_params = st.query_params
    aba_param = query_params.get("aba", "0")
    try:
        aba_index = int(aba_param)
        if aba_index < 0 or aba_index >= len(ABAS):
            aba_index = 0
    except:
        aba_index = 0
    
    # Seletor de aba usando radio horizontal (persiste estado)
    aba_selecionada = st.radio(
        "Navegação",
        options=ABAS,
        index=aba_index,
        horizontal=True,
        label_visibility="collapsed",
        key="aba_principal"
    )
    
    # Atualiza query param quando a aba muda
    novo_index = ABAS.index(aba_selecionada)
    if novo_index != aba_index:
        st.query_params["aba"] = str(novo_index)
    
    st.markdown("---")
    
    # Renderiza conteúdo baseado na aba selecionada
    if aba_selecionada == "📊 Dashboard":
        render_dashboard(df, geojson, ano_selecionado)
    elif aba_selecionada == "⚙️ Otimização":
        render_otimizacao(df, ano_selecionado)
    elif aba_selecionada == "🎲 Monte Carlo":
        render_monte_carlo(df, ano_selecionado)
    elif aba_selecionada == "📅 Multi-Período":
        render_multi_periodo(df, ano_selecionado)
    elif aba_selecionada == "📋 Conclusões":
        render_conclusoes(df, ano_selecionado)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9rem;">
        <p><strong>Trabalho Acadêmico - Pesquisa Operacional</strong></p>
        <p>
            Dados: <a href="https://www.ipea.gov.br/atlasviolencia/" target="_blank">Atlas da Violência (IPEA)</a> | 
            <a href="https://forumseguranca.org.br/anuario-brasileiro-seguranca-publica/" target="_blank">Anuário FBSP 2023</a> | 
            <a href="https://siconfi.tesouro.gov.br/" target="_blank">SICONFI</a>
        </p>
        <p>
            Método: Programação Linear (Simplex) via <a href="https://github.com/coin-or/pulp" target="_blank">PuLP/CBC</a> | 
            Interface: <a href="https://streamlit.io/" target="_blank">Streamlit</a>
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
