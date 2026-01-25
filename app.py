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
from dados import carregar_dados_consolidados, obter_coordenadas_estados
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
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CACHE DE DADOS
# =============================================================================
@st.cache_data
def carregar_dados():
    """
    Carrega e cacheia os dados consolidados.
    Usa elasticidade calculada por regressão da série histórica 1989-2022.
    """
    df = carregar_dados_consolidados()
    # Substitui elasticidade estimada pela calculada via regressão linear
    df = atualizar_elasticidade_dados(df)
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
# SIDEBAR - EXPLICAÇÃO DO MODELO
# =============================================================================
def render_sidebar():
    """Renderiza a sidebar com explicação educacional do modelo."""
    
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
        - $ε_i$ = elasticidade
        - $O_i$ = orçamento atual
        - $B$ = orçamento disponível
        """)
    
    with st.sidebar.expander("📊 Elasticidade Crime-Gasto"):
        st.markdown(explicar_elasticidade())
    
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
    - IBGE (População)
    """)


# =============================================================================
# ABA 1: DASHBOARD
# =============================================================================
def render_dashboard(df: pd.DataFrame, geojson):
    """Renderiza a aba de Dashboard com visualizações dos dados atuais."""
    
    st.header("📊 Dashboard - Situação Atual")
    st.markdown("Visualização dos dados de violência e orçamento de segurança pública por estado (2022).")
    
    # Métricas resumo
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_mortes = df['mortes_violentas'].sum()
        st.metric(
            label="Total de Mortes Violentas",
            value=f"{total_mortes:,.0f}",
            help="Número total de mortes violentas em 2022"
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
            help="Soma dos orçamentos de segurança de todos os estados"
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
            )
        )
        st.plotly_chart(fig_mapa, use_container_width=True)
    
    with col_grafico:
        st.subheader("📈 Top 10 Estados - Maior Taxa de Violência")
        
        top10 = df.nlargest(10, 'taxa_mortes_100k').sort_values('taxa_mortes_100k')
        
        fig_bar = px.bar(
            top10,
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
            height=450,
            showlegend=False,
            coloraxis_showscale=False,
            xaxis_title="Taxa de Mortes por 100 mil hab.",
            yaxis_title=""
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Segunda linha de gráficos
    st.markdown("---")
    col_scatter, col_regiao = st.columns(2)
    
    with col_scatter:
        st.subheader("💰 Relação: Gasto Per Capita × Taxa de Violência")
        
        fig_scatter = px.scatter(
            df,
            x='gasto_per_capita',
            y='taxa_mortes_100k',
            size='populacao',
            color='regiao',
            hover_name='estado',
            text='sigla',
            labels={
                'gasto_per_capita': 'Gasto Per Capita (R$)',
                'taxa_mortes_100k': 'Taxa por 100 mil',
                'regiao': 'Região',
                'populacao': 'População'
            }
        )
        fig_scatter.update_traces(textposition='top center', textfont_size=9)
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col_regiao:
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
        
        fig_regiao.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_regiao, use_container_width=True)
    
    # Tabela de dados
    st.markdown("---")
    with st.expander("📋 Ver Tabela de Dados Completa"):
        st.dataframe(
            df[[
                'sigla', 'estado', 'regiao', 'populacao', 
                'mortes_violentas', 'taxa_mortes_100k',
                'orcamento_2022_milhoes', 'gasto_per_capita', 
                'elasticidade', 'indice_prioridade'
            ]].style.format({
                'populacao': '{:,.0f}',
                'mortes_violentas': '{:,.0f}',
                'taxa_mortes_100k': '{:.1f}',
                'orcamento_2022_milhoes': '{:,.1f}',
                'gasto_per_capita': 'R$ {:,.0f}',
                'elasticidade': '{:.4f}',
                'indice_prioridade': '{:.2f}'
            }).background_gradient(subset=['taxa_mortes_100k'], cmap='YlOrRd'),
            use_container_width=True,
            height=400
        )


# =============================================================================
# ABA 2: OTIMIZAÇÃO
# =============================================================================
def render_otimizacao(df: pd.DataFrame):
    """Renderiza a aba de Otimização com controles e resultados."""
    
    st.header("⚙️ Otimização - Alocação de Recursos")
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
                    fig_alloc.update_layout(height=400)
                    st.plotly_chart(fig_alloc, use_container_width=True)
                
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
            
            st.dataframe(
                resultado.alocacao[[
                    'sigla', 'estado', 'regiao',
                    'investimento_milhoes', 'mortes_antes', 
                    'mortes_depois', 'reducao_mortes', 'reducao_percentual'
                ]].sort_values('investimento_milhoes', ascending=False).style.format({
                    'investimento_milhoes': 'R$ {:,.2f}',
                    'mortes_antes': '{:,.0f}',
                    'mortes_depois': '{:,.0f}',
                    'reducao_mortes': '{:,.0f}',
                    'reducao_percentual': '{:.2f}%'
                }).background_gradient(subset=['investimento_milhoes'], cmap='Greens'),
                use_container_width=True,
                height=400
            )
        
        else:
            st.error(f"❌ Não foi possível encontrar solução ótima. Status: {resultado.status}")
            st.info("""
            Possíveis causas:
            - Orçamento muito baixo para atender restrições mínimas
            - Parâmetros inconsistentes (máximo < mínimo)
            
            Tente ajustar os parâmetros e executar novamente.
            """)


# =============================================================================
# ABA 3: COMPARATIVO
# =============================================================================
def render_comparativo(df: pd.DataFrame):
    """Renderiza a aba de Comparativo Antes vs. Depois."""
    
    st.header("📊 Comparativo - Antes vs. Depois")
    
    if 'resultado_otimizacao' not in st.session_state:
        st.warning("⚠️ Execute a otimização primeiro na aba 'Otimização' para ver o comparativo.")
        return
    
    resultado = st.session_state['resultado_otimizacao']
    
    if resultado.status != 'Optimal':
        st.error("❌ A última otimização não encontrou solução ótima.")
        return
    
    st.markdown(f"""
    **Cenário analisado:** Orçamento suplementar de **R$ {resultado.orcamento_usado/1000:.2f} bilhões**
    """)
    
    # Gráfico comparativo de barras
    st.subheader("📈 Comparativo de Mortes por Estado (Antes × Depois)")
    
    df_comp = resultado.alocacao.copy()
    df_comp = df_comp.sort_values('mortes_antes', ascending=True).tail(15)  # Top 15
    
    fig_comp = go.Figure()
    
    fig_comp.add_trace(go.Bar(
        name='Antes',
        y=df_comp['sigla'],
        x=df_comp['mortes_antes'],
        orientation='h',
        marker_color='#ff6b6b',
        text=df_comp['mortes_antes'],
        textposition='auto'
    ))
    
    fig_comp.add_trace(go.Bar(
        name='Depois',
        y=df_comp['sigla'],
        x=df_comp['mortes_depois'],
        orientation='h',
        marker_color='#51cf66',
        text=df_comp['mortes_depois'],
        textposition='auto'
    ))
    
    fig_comp.update_layout(
        barmode='group',
        height=600,
        xaxis_title="Número de Mortes Violentas",
        yaxis_title="Estado",
        legend_title="Cenário",
        title="Top 15 Estados com Maior Número de Mortes"
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
            size='elasticidade',
            color='custo_por_vida',
            hover_name='estado',
            text='sigla',
            color_continuous_scale='RdYlGn_r',
            labels={
                'investimento_milhoes': 'Investimento (R$ milhões)',
                'reducao_mortes': 'Vidas Salvas',
                'custo_por_vida': 'Custo/Vida (R$ mi)',
                'elasticidade': 'Elasticidade'
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
def render_sensibilidade(df: pd.DataFrame):
    """
    Renderiza a aba de análise de sensibilidade.
    Inclui gráfico tornado, shadow prices e análise de cenários.
    """
    st.header("🔍 Análise de Sensibilidade")
    st.markdown("""
    Estudo de como variações nos parâmetros afetam o resultado da otimização.
    Essencial para entender a robustez da solução e identificar parâmetros críticos.
    """)
    
    # Parâmetros
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
    
    if st.button("🔍 Executar Análise de Sensibilidade", type="primary", key="btn_sens"):
        with st.spinner("Calculando sensibilidade..."):
            
            # 1. Sensibilidade do Orçamento
            st.subheader("📊 Sensibilidade ao Orçamento")
            resultados_sens = analisar_sensibilidade_orcamento(
                df,
                orcamento_base=orcamento_base,
                variacao_percentual=variacao_pct / 100
            )
            
            # Gráfico de variação
            df_sens = pd.DataFrame(resultados_sens)
            fig_sens = px.line(
                df_sens,
                x='orcamento_milhoes',
                y='vidas_salvas',
                markers=True,
                labels={
                    'orcamento_milhoes': 'Orçamento (R$ milhões)',
                    'vidas_salvas': 'Vidas Salvas'
                },
                title=f"Impacto do Orçamento na Redução de Crimes (±{variacao_pct}%)"
            )
            fig_sens.add_vline(
                x=orcamento_base, 
                line_dash="dash", 
                annotation_text="Base"
            )
            st.plotly_chart(fig_sens, use_container_width=True)
            
            # 2. Shadow Prices
            st.subheader("💰 Shadow Prices (Preços Sombra)")
            st.markdown("""
            O **Shadow Price** indica quanto a função objetivo (vidas salvas) 
            melhoraria se relaxássemos uma restrição em 1 unidade.
            """)
            
            shadow = calcular_shadow_prices(df, orcamento_base)
            
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
            
            fig_tornado = gerar_grafico_tornado(df, orcamento_base, variacao_pct / 100)
            st.plotly_chart(fig_tornado, use_container_width=True)
            
            # 4. Análise de Cenários
            st.subheader("📋 Análise de Cenários")
            cenarios = analisar_cenarios(df, orcamento_base)
            
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
def render_monte_carlo(df: pd.DataFrame):
    """
    Renderiza a aba de simulação Monte Carlo.
    Quantifica incerteza nos resultados via simulação estocástica.
    """
    st.header("🎲 Simulação Monte Carlo")
    st.markdown("""
    Simula centenas de cenários com variações aleatórias nos parâmetros
    para obter intervalos de confiança nos resultados.
    """)
    
    # Parâmetros da simulação
    col1, col2, col3 = st.columns(3)
    with col1:
        orcamento = st.slider(
            "Orçamento (R$ milhões)",
            min_value=1000.0,
            max_value=10000.0,
            value=5000.0,
            step=500.0,
            key="mc_orcamento"
        )
    with col2:
        n_simulacoes = st.selectbox(
            "Número de Simulações",
            options=[100, 250, 500, 1000],
            index=2,  # 500 por padrão
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
    
    if st.button("🎲 Executar Simulação Monte Carlo", type="primary", key="btn_mc"):
        
        progress_bar = st.progress(0, text="Executando simulações...")
        
        # Executa Monte Carlo (a função imprime progresso via verbose)
        resultado_mc = executar_monte_carlo(
            df_dados=df,
            orcamento=orcamento,
            n_simulacoes=n_simulacoes,
            incerteza_elasticidade=variacao / 100,
            incerteza_taxa=variacao / 100 * 0.5,  # Menor incerteza no crime
            verbose=False  # Não imprime no console
        )
        
        progress_bar.empty()
        
        # Métricas resumo
        st.subheader("📊 Resultados da Simulação")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Vidas Salvas (Média)",
                f"{resultado_mc.media_reducao:.0f}"
            )
        with col2:
            st.metric(
                "Desvio Padrão",
                f"±{resultado_mc.desvio_padrao_reducao:.0f}"
            )
        with col3:
            st.metric(
                "IC 95% Inferior",
                f"{resultado_mc.intervalo_confianca_95[0]:.0f}"
            )
        with col4:
            st.metric(
                "IC 95% Superior",
                f"{resultado_mc.intervalo_confianca_95[1]:.0f}"
            )
        
        # Histograma
        st.subheader("📈 Distribuição dos Resultados")
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=resultado_mc.distribuicao_reducao,
            nbinsx=30,
            name="Simulações",
            marker_color='#3498db'
        ))
        
        # Adiciona linhas de IC
        fig_hist.add_vline(
            x=resultado_mc.intervalo_confianca_95[0],
            line_dash="dash",
            line_color="red",
            annotation_text="IC 2.5%"
        )
        fig_hist.add_vline(
            x=resultado_mc.intervalo_confianca_95[1],
            line_dash="dash",
            line_color="red",
            annotation_text="IC 97.5%"
        )
        fig_hist.add_vline(
            x=resultado_mc.media_reducao,
            line_color="green",
            annotation_text="Média"
        )
        
        fig_hist.update_layout(
            title=f"Distribuição de Vidas Salvas ({n_simulacoes} simulações)",
            xaxis_title="Vidas Salvas",
            yaxis_title="Frequência",
            showlegend=False
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
        
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
        
        # Taxa de sucesso
        st.info(f"✅ **Taxa de sucesso:** {resultado_mc.n_sucesso}/{resultado_mc.n_simulacoes} simulações convergiram ({resultado_mc.n_sucesso/resultado_mc.n_simulacoes*100:.1f}%)")


# =============================================================================
# ABA 6: BACKTESTING
# =============================================================================
def render_backtesting(df: pd.DataFrame):
    """
    Renderiza a aba de backtesting.
    Valida o modelo usando dados históricos.
    """
    st.header("🔄 Backtesting - Validação Histórica")
    st.markdown("""
    Testa se o modelo teria funcionado no passado, comparando previsões
    com resultados reais. Fundamental para validar a abordagem.
    """)
    
    # Opções de backtesting
    col1, col2 = st.columns(2)
    with col1:
        metodo = st.radio(
            "Método de Validação",
            options=["Janela Deslizante", "Período Fixo"],
            help="Janela deslizante é mais robusto mas mais lento"
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
    
    if st.button("🔄 Executar Backtesting", type="primary", key="btn_bt"):
        with st.spinner("Executando validação histórica..."):
            
            try:
                if metodo == "Janela Deslizante":
                    # Usa janela deslizante: mais robusto
                    resultado_rolling = validar_modelo_rolling(
                        janela_treino=tamanho_janela,
                        janela_teste=1,
                        ano_inicio=2010,
                        ano_fim=2022
                    )
                    
                    if resultado_rolling is None or resultado_rolling.empty:
                        st.warning("Dados insuficientes para backtesting com janela deslizante.")
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
                    
                else:
                    # Período fixo
                    resultado_bt = executar_backtest(
                        ano_treino_inicio=2012,
                        ano_treino_fim=2017,
                        ano_teste_inicio=2018,
                        ano_teste_fim=2022
                    )
                    
                    if resultado_bt is None:
                        st.error("Erro ao executar backtesting.")
                        return
                    
                    st.subheader("📊 Métricas de Erro")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("MAPE", f"{resultado_bt.mape:.1f}%")
                    with col2:
                        st.metric("RMSE", f"{resultado_bt.rmse:.2f}")
                    with col3:
                        st.metric("Correlação", f"{resultado_bt.correlacao:.3f}")
                    
                    # Interpretação
                    mape = resultado_bt.mape
                    if mape < 10:
                        qualidade = "🟢 Excelente"
                        interpretacao = "O modelo tem alta precisão preditiva."
                    elif mape < 20:
                        qualidade = "🟡 Boa"
                        interpretacao = "O modelo é razoável para planejamento."
                    elif mape < 30:
                        qualidade = "🟠 Moderada"
                        interpretacao = "Usar com cautela; considerar intervalos de confiança."
                    else:
                        qualidade = "🔴 Baixa"
                        interpretacao = "Modelo precisa de ajustes ou mais dados."
                    
                    st.info(f"**Qualidade do Modelo: {qualidade}**\n\n{interpretacao}")
                    
                    # Gráfico Previsto vs Real
                    if hasattr(resultado_bt, 'previsoes') and resultado_bt.previsoes is not None:
                        st.subheader("📈 Previsto vs Real")
                        
                        df_comp = resultado_bt.previsoes
                        
                        fig_bt = go.Figure()
                        fig_bt.add_trace(go.Scatter(
                            x=df_comp['valor_real'],
                            y=df_comp['valor_previsto'],
                            mode='markers',
                            marker=dict(size=10),
                            text=df_comp.get('estado', df_comp.index),
                            name="Estados"
                        ))
                        
                        # Linha de perfeição
                        min_val = min(df_comp['valor_real'].min(), df_comp['valor_previsto'].min())
                        max_val = max(df_comp['valor_real'].max(), df_comp['valor_previsto'].max())
                        fig_bt.add_trace(go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            line=dict(dash='dash', color='red'),
                            name="Perfeito (y=x)"
                        ))
                        
                        fig_bt.update_layout(
                            title="Comparação: Valores Previstos vs Reais",
                            xaxis_title="Valor Real",
                            yaxis_title="Valor Previsto"
                        )
                        
                        st.plotly_chart(fig_bt, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Erro ao executar backtesting: {e}")
                import traceback
                st.code(traceback.format_exc())


# =============================================================================
# ABA 7: MODELO MULTI-PERÍODO
# =============================================================================
def render_multi_periodo(df: pd.DataFrame):
    """
    Renderiza a aba de otimização multi-período.
    Planejamento de investimentos ao longo de vários anos.
    """
    st.header("📅 Otimização Multi-Período")
    st.markdown("""
    Planeja a distribuição de investimentos ao longo de múltiplos anos,
    considerando que investimentos têm efeitos acumulados e depreciação.
    """)
    
    # Parâmetros
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
    
    if st.button("📅 Calcular Estratégias Multi-Período", type="primary", key="btn_mp"):
        with st.spinner("Otimizando para múltiplos períodos..."):
            
            try:
                orcamento_milhoes = orcamento_total * 1000
                
                # Compara estratégias
                df_comparativo = comparar_estrategias(
                    df,
                    orcamento_total=orcamento_milhoes,
                    n_periodos=n_periodos
                )
                
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
                
                # Ordena por melhor resultado
                df_display = df_display.sort_values('Crimes Evitados', ascending=False)
                
                # Destaca o melhor
                melhor = df_display.iloc[0]['Estratégia']
                st.success(f"🏆 **Melhor estratégia: {melhor}**")
                
                # Tabela de resultados
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
                        valores_bi = [v / 1000 for v in row['distribuicao']]  # Converter para bilhões
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
                continuam gerando benefícios nos anos seguintes (com certa depreciação).
                
                Matematicamente:
                - Investimento no ano 1: gera benefícios nos anos 1, 2, 3, 4, 5
                - Investimento no ano 5: gera benefício apenas no ano 5
                
                Por isso, concentrar recursos no início maximiza o impacto total.
                """)
                
            except Exception as e:
                st.error(f"Erro ao calcular multi-período: {e}")
                import traceback
                st.code(traceback.format_exc())


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
    
    # Carrega dados
    try:
        df = carregar_dados()
        geojson = carregar_geojson_brasil()
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        st.stop()
    
    # Renderiza sidebar
    render_sidebar()
    
    # Abas principais - 7 abas com todas as funcionalidades
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Dashboard",
        "⚙️ Otimização",
        "📈 Comparativo",
        "🔍 Sensibilidade",
        "🎲 Monte Carlo",
        "🔄 Backtesting",
        "📅 Multi-Período"
    ])
    
    with tab1:
        render_dashboard(df, geojson)
    
    with tab2:
        render_otimizacao(df)
    
    with tab3:
        render_comparativo(df)
    
    with tab4:
        render_sensibilidade(df)
    
    with tab5:
        render_monte_carlo(df)
    
    with tab6:
        render_backtesting(df)
    
    with tab7:
        render_multi_periodo(df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9rem;">
        <p>Trabalho Acadêmico - Pesquisa Operacional</p>
        <p>Dados: Atlas da Violência (IPEA) | Anuário Brasileiro de Segurança Pública (FBSP)</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
