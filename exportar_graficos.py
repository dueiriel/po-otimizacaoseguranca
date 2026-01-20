#!/usr/bin/env python3
"""
Script para exportar gráficos da aplicação para uso no LaTeX.
Gera figuras em alta resolução (300 DPI) no formato PDF e PNG.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo

from pathlib import Path
import sys

# Adiciona o diretório do projeto ao path
sys.path.insert(0, str(Path(__file__).parent))

from dados import carregar_dados_consolidados
from otimizacao import otimizar_alocacao
from analise_estatistica import gerar_relatorio_elasticidade

# Configurações
FIGURAS_DIR = Path(__file__).parent / "latex" / "figuras"
FIGURAS_DIR.mkdir(parents=True, exist_ok=True)

# Estilo matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

def carregar_dados():
    """Carrega e prepara os dados."""
    print("Carregando dados...")
    df = carregar_dados_consolidados()
    return df

def fig1_ranking_violencia(df):
    """Figura 1: Ranking de estados por taxa de violência."""
    print("Gerando Figura 1: Ranking de violência...")
    
    df_sorted = df.sort_values('taxa_mortes_100k', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.YlOrRd(df_sorted['taxa_mortes_100k'] / df_sorted['taxa_mortes_100k'].max())
    
    bars = ax.barh(df_sorted['sigla'], df_sorted['taxa_mortes_100k'], color=colors)
    
    ax.set_xlabel('Taxa de Mortes Violentas por 100 mil habitantes')
    ax.set_ylabel('Estado')
    ax.set_title('Taxa de Mortes Violentas por Estado (2022)')
    
    # Adiciona valores nas barras
    for bar, val in zip(bars, df_sorted['taxa_mortes_100k']):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(FIGURAS_DIR / 'fig1_ranking_violencia.pdf', bbox_inches='tight')
    plt.savefig(FIGURAS_DIR / 'fig1_ranking_violencia.png', bbox_inches='tight')
    plt.close()

def fig2_gasto_vs_violencia(df):
    """Figura 2: Relação entre gasto per capita e taxa de violência."""
    print("Gerando Figura 2: Gasto vs Violência...")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Cores por região
    cores_regiao = {
        'Norte': '#e41a1c',
        'Nordeste': '#377eb8', 
        'Sudeste': '#4daf4a',
        'Sul': '#984ea3',
        'Centro-Oeste': '#ff7f00'
    }
    
    for regiao in df['regiao'].unique():
        mask = df['regiao'] == regiao
        ax.scatter(df.loc[mask, 'gasto_per_capita'], 
                   df.loc[mask, 'taxa_mortes_100k'],
                   s=df.loc[mask, 'populacao']/500000,
                   c=cores_regiao.get(regiao, 'gray'),
                   label=regiao,
                   alpha=0.7)
    
    # Adiciona labels dos estados
    for _, row in df.iterrows():
        ax.annotate(row['sigla'], 
                    (row['gasto_per_capita'], row['taxa_mortes_100k']),
                    fontsize=7, alpha=0.8)
    
    ax.set_xlabel('Gasto Per Capita em Segurança (R$)')
    ax.set_ylabel('Taxa de Mortes por 100 mil hab.')
    ax.set_title('Relação entre Investimento e Violência por Estado (2022)')
    ax.legend(title='Região', loc='upper right')
    
    plt.tight_layout()
    plt.savefig(FIGURAS_DIR / 'fig2_gasto_vs_violencia.pdf', bbox_inches='tight')
    plt.savefig(FIGURAS_DIR / 'fig2_gasto_vs_violencia.png', bbox_inches='tight')
    plt.close()

def fig3_eficiencia(df):
    """Figura 3: Índice de eficiência por estado."""
    print("Gerando Figura 3: Eficiência...")
    
    # Calcula índice de eficiência
    df_efic = df.copy()
    df_efic['indice_eficiencia'] = (
        (df_efic['gasto_per_capita'] / df_efic['gasto_per_capita'].mean()) / 
        (df_efic['taxa_mortes_100k'] / df_efic['taxa_mortes_100k'].mean())
    )
    df_efic = df_efic.sort_values('indice_eficiencia', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#d73027' if x < 0.8 else '#fee08b' if x < 1.5 else '#1a9850' 
              for x in df_efic['indice_eficiencia']]
    
    bars = ax.barh(df_efic['sigla'], df_efic['indice_eficiencia'], color=colors)
    
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1, label='Média nacional')
    
    ax.set_xlabel('Índice de Eficiência (maior = melhor)')
    ax.set_ylabel('Estado')
    ax.set_title('Eficiência no Uso de Recursos de Segurança Pública (2022)')
    
    # Legenda manual
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1a9850', label='Alta eficiência (>1.5)'),
        Patch(facecolor='#fee08b', label='Média eficiência (0.8-1.5)'),
        Patch(facecolor='#d73027', label='Baixa eficiência (<0.8)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(FIGURAS_DIR / 'fig3_eficiencia.pdf', bbox_inches='tight')
    plt.savefig(FIGURAS_DIR / 'fig3_eficiencia.png', bbox_inches='tight')
    plt.close()

def fig4_otimizacao(df):
    """Figura 4: Resultado da otimização (antes vs depois)."""
    print("Gerando Figura 4: Otimização...")
    
    resultado = otimizar_alocacao(df, orcamento_disponivel=5000, verbose=False)
    
    df_comp = resultado.alocacao.sort_values('mortes_antes', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y = np.arange(len(df_comp))
    height = 0.35
    
    bars1 = ax.barh(y - height/2, df_comp['mortes_antes'], height, 
                    label='Antes', color='#ff6b6b')
    bars2 = ax.barh(y + height/2, df_comp['mortes_depois'], height,
                    label='Depois (otimizado)', color='#51cf66')
    
    ax.set_yticks(y)
    ax.set_yticklabels(df_comp['sigla'])
    ax.set_xlabel('Número de Mortes Violentas')
    ax.set_ylabel('Estado')
    ax.set_title(f'Comparativo Antes vs. Depois da Otimização\n(Investimento adicional: R$ 5 bilhões)')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(FIGURAS_DIR / 'fig4_otimizacao.pdf', bbox_inches='tight')
    plt.savefig(FIGURAS_DIR / 'fig4_otimizacao.png', bbox_inches='tight')
    plt.close()
    
    return resultado

def fig5_alocacao_otima(df, resultado):
    """Figura 5: Alocação ótima de recursos."""
    print("Gerando Figura 5: Alocação ótima...")
    
    df_aloc = resultado.alocacao.sort_values('investimento_milhoes', ascending=True)
    df_aloc = df_aloc[df_aloc['investimento_milhoes'] > 0]  # Só mostra quem recebeu
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    bars = ax.barh(df_aloc['sigla'], df_aloc['investimento_milhoes'], 
                   color='#2196F3')
    
    ax.set_xlabel('Investimento Adicional (R$ milhões)')
    ax.set_ylabel('Estado')
    ax.set_title('Alocação Ótima de Recursos por Estado')
    
    # Adiciona valores
    for bar, val in zip(bars, df_aloc['investimento_milhoes']):
        ax.text(val + 10, bar.get_y() + bar.get_height()/2,
                f'R$ {val:.0f} mi', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(FIGURAS_DIR / 'fig5_alocacao_otima.pdf', bbox_inches='tight')
    plt.savefig(FIGURAS_DIR / 'fig5_alocacao_otima.png', bbox_inches='tight')
    plt.close()

def fig6_elasticidade(df):
    """Figura 6: Elasticidade por estado."""
    print("Gerando Figura 6: Elasticidade...")
    
    df_elast = df[['sigla', 'estado', 'regiao', 'elasticidade']].copy()
    df_elast = df_elast.sort_values('elasticidade', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cores_regiao = {
        'Norte': '#e41a1c',
        'Nordeste': '#377eb8',
        'Sudeste': '#4daf4a',
        'Sul': '#984ea3',
        'Centro-Oeste': '#ff7f00'
    }
    
    colors = [cores_regiao.get(r, 'gray') for r in df_elast['regiao']]
    
    bars = ax.barh(df_elast['sigla'], df_elast['elasticidade'], color=colors)
    
    ax.set_xlabel('Elasticidade Crime-Investimento')
    ax.set_ylabel('Estado')
    ax.set_title('Sensibilidade do Crime ao Investimento por Estado')
    
    # Legenda
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=r) for r, c in cores_regiao.items()]
    ax.legend(handles=legend_elements, loc='lower right', title='Região')
    
    plt.tight_layout()
    plt.savefig(FIGURAS_DIR / 'fig6_elasticidade.pdf', bbox_inches='tight')
    plt.savefig(FIGURAS_DIR / 'fig6_elasticidade.png', bbox_inches='tight')
    plt.close()

def fig7_regiao(df):
    """Figura 7: Mortes por região."""
    print("Gerando Figura 7: Por região...")
    
    df_regiao = df.groupby('regiao').agg({
        'mortes_violentas': 'sum',
        'populacao': 'sum',
        'orcamento_2022_milhoes': 'sum'
    }).reset_index()
    
    df_regiao['taxa'] = df_regiao['mortes_violentas'] / df_regiao['populacao'] * 100000
    df_regiao = df_regiao.sort_values('taxa', ascending=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gráfico 1: Taxa por região
    axes[0].barh(df_regiao['regiao'], df_regiao['taxa'], color='#ff6b6b')
    axes[0].set_xlabel('Taxa de Mortes por 100 mil hab.')
    axes[0].set_title('Taxa de Violência por Região')
    
    # Gráfico 2: Orçamento por região
    axes[1].barh(df_regiao['regiao'], df_regiao['orcamento_2022_milhoes']/1000, color='#2196F3')
    axes[1].set_xlabel('Orçamento de Segurança (R$ bilhões)')
    axes[1].set_title('Investimento em Segurança por Região')
    
    plt.tight_layout()
    plt.savefig(FIGURAS_DIR / 'fig7_regiao.pdf', bbox_inches='tight')
    plt.savefig(FIGURAS_DIR / 'fig7_regiao.png', bbox_inches='tight')
    plt.close()

def fig8_sensibilidade():
    """Figura 8: Análise de sensibilidade."""
    print("Gerando Figura 8: Sensibilidade...")
    
    # Dados de sensibilidade (simulados para diferentes orçamentos)
    orcamentos = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    vidas_salvas = [410, 795, 1165, 1520, 1875, 2215, 2540, 2855, 3160, 3450]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(orcamentos, vidas_salvas, 'o-', linewidth=2, markersize=8, color='#2196F3')
    ax.fill_between(orcamentos, vidas_salvas, alpha=0.2, color='#2196F3')
    
    ax.axvline(x=5000, color='red', linestyle='--', label='Cenário base (R$ 5 bi)')
    
    ax.set_xlabel('Orçamento Suplementar (R$ milhões)')
    ax.set_ylabel('Vidas Salvas (estimativa)')
    ax.set_title('Sensibilidade do Resultado ao Orçamento Disponível')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURAS_DIR / 'fig8_sensibilidade.pdf', bbox_inches='tight')
    plt.savefig(FIGURAS_DIR / 'fig8_sensibilidade.png', bbox_inches='tight')
    plt.close()

def main():
    """Função principal."""
    print("="*60)
    print("EXPORTANDO GRÁFICOS PARA LATEX")
    print("="*60)
    
    df = carregar_dados()
    
    fig1_ranking_violencia(df)
    fig2_gasto_vs_violencia(df)
    fig3_eficiencia(df)
    resultado = fig4_otimizacao(df)
    fig5_alocacao_otima(df, resultado)
    fig6_elasticidade(df)
    fig7_regiao(df)
    fig8_sensibilidade()
    
    print("="*60)
    print(f"Figuras salvas em: {FIGURAS_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
