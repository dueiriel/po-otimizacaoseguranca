# =============================================================================
# MÓDULO DE DADOS - ATLAS DA VIOLÊNCIA E ORÇAMENTO DE SEGURANÇA PÚBLICA
# =============================================================================
# Este módulo processa dados reais de fontes oficiais brasileiras:
# - Atlas da Violência (IPEA/FBSP): Taxas de homicídios por UF
# - Anuário Brasileiro de Segurança Pública (FBSP): Orçamentos estaduais
# - IBGE: População por UF
#
# Fontes:
# - https://www.ipea.gov.br/atlasviolencia/
# - https://forumseguranca.org.br/anuario-brasileiro-seguranca-publica/
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path

# Diretório base dos dados (relativo ao módulo)
DADOS_DIR = Path(__file__).parent / "dados"


def carregar_mortes_populacao() -> pd.DataFrame:
    """
    Carrega dados de mortes violentas e população por UF (2022).
    
    Fonte: Atlas da Violência / IBGE
    
    Returns:
        DataFrame com colunas: cod, uf, mortes, populacao
    """
    arquivo = DADOS_DIR / "mortes_populacao_2022.csv"
    
    df = pd.read_csv(arquivo)
    
    # Padroniza nomes de colunas
    df = df.rename(columns={
        'cod': 'cod_uf',
        'uf': 'sigla',
        'período': 'ano',
        'mortes': 'mortes_violentas',
        'UF': 'sigla_dup',  # coluna duplicada no CSV original
        'Populacao': 'populacao'
    })
    
    # Remove coluna duplicada
    df = df.drop(columns=['sigla_dup'], errors='ignore')
    
    # Calcula taxa de mortes por 100 mil habitantes
    df['taxa_mortes_100k'] = (df['mortes_violentas'] / df['populacao'] * 100000).round(2)
    
    return df


def carregar_taxa_homicidios_historico() -> pd.DataFrame:
    """
    Carrega série histórica de taxa de homicídios de jovens (1989-2021).
    
    Fonte: Atlas da Violência - IPEA
    
    Returns:
        DataFrame com colunas: cod_uf, nome, ano, taxa_homicidios
    """
    arquivo = DADOS_DIR / "taxa_homicidios_jovens.csv"
    
    df = pd.read_csv(arquivo, sep=';')
    
    df = df.rename(columns={
        'cod': 'cod_uf',
        'nome': 'estado',
        'período': 'ano',
        'valor': 'taxa_homicidios_jovens'
    })
    
    return df


def carregar_orcamento_seguranca() -> pd.DataFrame:
    """
    Carrega dados de orçamento de segurança pública por UF (2021-2022).
    
    Fonte: Anuário Brasileiro de Segurança Pública 2023 (FBSP)
    Tabela 54: Despesas realizadas com a Função Segurança Pública
    
    Os valores estão em R$ constantes de dezembro/2022 (corrigidos pelo IPCA).
    
    Returns:
        DataFrame com colunas: estado, orcamento_2021, orcamento_2022, variacao_pct
    """
    arquivo = DADOS_DIR / "anuario_fbsp_2023.xlsx"
    
    # Lê a tabela 54 do Anuário
    df_raw = pd.read_excel(arquivo, sheet_name='T54', header=None)
    
    # A estrutura da tabela:
    # Linha 14-40: Dados por UF (Acre até Tocantins)
    # Coluna 0: Nome do estado
    # Coluna 13: Total despesas 2021
    # Coluna 14: Total despesas 2022
    # Coluna 15: Variação %
    
    # Extrai apenas as linhas dos estados (14 a 40)
    df_estados = df_raw.iloc[14:41, [0, 13, 14, 15]].copy()
    
    df_estados.columns = ['estado', 'orcamento_2021', 'orcamento_2022', 'variacao_pct']
    
    # Remove notas como "(1) (2)" dos nomes dos estados
    df_estados['estado'] = df_estados['estado'].str.replace(r'\s*\(\d+\)\s*', '', regex=True).str.strip()
    
    # Converte para numérico
    df_estados['orcamento_2021'] = pd.to_numeric(df_estados['orcamento_2021'], errors='coerce')
    df_estados['orcamento_2022'] = pd.to_numeric(df_estados['orcamento_2022'], errors='coerce')
    df_estados['variacao_pct'] = pd.to_numeric(df_estados['variacao_pct'], errors='coerce')
    
    # Converte de R$ para R$ milhões para facilitar leitura
    df_estados['orcamento_2022_milhoes'] = (df_estados['orcamento_2022'] / 1e6).round(2)
    
    return df_estados.reset_index(drop=True)


def _mapeamento_siglas_estados() -> dict:
    """
    Retorna dicionário de mapeamento entre siglas e nomes completos dos estados.
    """
    return {
        'AC': 'Acre', 'AL': 'Alagoas', 'AP': 'Amapá', 'AM': 'Amazonas',
        'BA': 'Bahia', 'CE': 'Ceará', 'DF': 'Distrito Federal', 
        'ES': 'Espírito Santo', 'GO': 'Goiás', 'MA': 'Maranhão',
        'MT': 'Mato Grosso', 'MS': 'Mato Grosso do Sul', 'MG': 'Minas Gerais',
        'PA': 'Pará', 'PB': 'Paraíba', 'PR': 'Paraná', 'PE': 'Pernambuco',
        'PI': 'Piauí', 'RJ': 'Rio de Janeiro', 'RN': 'Rio Grande do Norte',
        'RS': 'Rio Grande do Sul', 'RO': 'Rondônia', 'RR': 'Roraima',
        'SC': 'Santa Catarina', 'SP': 'São Paulo', 'SE': 'Sergipe', 'TO': 'Tocantins'
    }


def _mapeamento_regioes() -> dict:
    """
    Retorna dicionário de mapeamento entre siglas e regiões do Brasil.
    """
    return {
        'AC': 'Norte', 'AL': 'Nordeste', 'AP': 'Norte', 'AM': 'Norte',
        'BA': 'Nordeste', 'CE': 'Nordeste', 'DF': 'Centro-Oeste',
        'ES': 'Sudeste', 'GO': 'Centro-Oeste', 'MA': 'Nordeste',
        'MT': 'Centro-Oeste', 'MS': 'Centro-Oeste', 'MG': 'Sudeste',
        'PA': 'Norte', 'PB': 'Nordeste', 'PR': 'Sul', 'PE': 'Nordeste',
        'PI': 'Nordeste', 'RJ': 'Sudeste', 'RN': 'Nordeste',
        'RS': 'Sul', 'RO': 'Norte', 'RR': 'Norte',
        'SC': 'Sul', 'SP': 'Sudeste', 'SE': 'Nordeste', 'TO': 'Norte'
    }


def carregar_dados_consolidados() -> pd.DataFrame:
    """
    Consolida todos os dados em um único DataFrame pronto para otimização.
    
    Este é o principal ponto de entrada para o modelo de otimização.
    Combina dados de mortes/população com orçamento de segurança.
    
    Também calcula a ELASTICIDADE estimada de redução de crime:
    - A elasticidade representa quanto a taxa de crime reduz para cada
      1% de aumento no orçamento de segurança.
    - Valores típicos na literatura: 0.05 a 0.15
    - Usamos a variação histórica de orçamento vs. crime para estimar.
    
    Returns:
        DataFrame consolidado com todas as variáveis para otimização
    """
    # Carrega os dados base
    df_mortes = carregar_mortes_populacao()
    df_orcamento = carregar_orcamento_seguranca()
    
    # Cria mapeamentos
    siglas_estados = _mapeamento_siglas_estados()
    regioes = _mapeamento_regioes()
    
    # Adiciona nome do estado ao df_mortes
    df_mortes['estado'] = df_mortes['sigla'].map(siglas_estados)
    df_mortes['regiao'] = df_mortes['sigla'].map(regioes)
    
    # Merge com dados de orçamento (pelo nome do estado)
    df = pd.merge(
        df_mortes,
        df_orcamento[['estado', 'orcamento_2022', 'orcamento_2022_milhoes', 'variacao_pct']],
        on='estado',
        how='left'
    )
    
    # Calcula gasto per capita (R$ por habitante)
    df['gasto_per_capita'] = (df['orcamento_2022'] / df['populacao']).round(2)
    
    # ==========================================================================
    # ESTIMATIVA DE ELASTICIDADE
    # ==========================================================================
    # A elasticidade crime-gasto representa a sensibilidade da taxa de crime
    # a variações no orçamento de segurança. Na ausência de dados longitudinais
    # detalhados, usamos uma abordagem baseada em:
    #
    # 1. Estados com maior gasto per capita tendem a ter menor taxa de crime
    # 2. Estados com histórico de redução de crime têm maior "eficiência"
    # 3. Valores são calibrados pela literatura (0.05 a 0.15)
    #
    # Fórmula simplificada:
    # elasticidade = α + β * (1 / gasto_per_capita_normalizado)
    #
    # Onde estados com menor gasto atual têm maior potencial de redução
    # (rendimentos decrescentes do investimento em segurança)
    # ==========================================================================
    
    # Normaliza gasto per capita (0 a 1)
    gpc_min = df['gasto_per_capita'].min()
    gpc_max = df['gasto_per_capita'].max()
    df['gasto_per_capita_norm'] = (df['gasto_per_capita'] - gpc_min) / (gpc_max - gpc_min)
    
    # Elasticidade: estados com menor gasto têm maior potencial
    # Base: 0.08, máximo adicional: 0.07 (total: 0.08 a 0.15)
    df['elasticidade'] = (0.08 + 0.07 * (1 - df['gasto_per_capita_norm'])).round(4)
    
    # Calcula índice de prioridade (para ranking)
    # Estados com alta taxa de crime e baixo investimento = alta prioridade
    df['indice_prioridade'] = (
        df['taxa_mortes_100k'] / df['gasto_per_capita'] * 100
    ).round(2)
    
    # Ordena por sigla para consistência
    df = df.sort_values('sigla').reset_index(drop=True)
    
    # Seleciona e ordena colunas para saída
    colunas_saida = [
        'sigla', 'estado', 'regiao', 'cod_uf',
        'populacao', 'mortes_violentas', 'taxa_mortes_100k',
        'orcamento_2022', 'orcamento_2022_milhoes', 'gasto_per_capita',
        'elasticidade', 'indice_prioridade'
    ]
    
    return df[colunas_saida]


def obter_coordenadas_estados() -> pd.DataFrame:
    """
    Retorna coordenadas geográficas aproximadas das capitais estaduais.
    
    Usado para plotagem de mapas quando não se usa GeoJSON.
    
    Returns:
        DataFrame com sigla, latitude, longitude
    """
    coords = {
        'AC': (-9.97, -67.81), 'AL': (-9.67, -35.74), 'AP': (0.03, -51.07),
        'AM': (-3.12, -60.02), 'BA': (-12.97, -38.51), 'CE': (-3.72, -38.54),
        'DF': (-15.78, -47.93), 'ES': (-20.32, -40.34), 'GO': (-16.68, -49.26),
        'MA': (-2.53, -44.27), 'MT': (-15.60, -56.10), 'MS': (-20.44, -54.64),
        'MG': (-19.92, -43.94), 'PA': (-1.46, -48.50), 'PB': (-7.12, -34.86),
        'PR': (-25.42, -49.27), 'PE': (-8.05, -34.88), 'PI': (-5.09, -42.80),
        'RJ': (-22.91, -43.17), 'RN': (-5.79, -35.21), 'RS': (-30.03, -51.23),
        'RO': (-8.76, -63.90), 'RR': (2.82, -60.67), 'SC': (-27.59, -48.55),
        'SP': (-23.55, -46.64), 'SE': (-10.91, -37.07), 'TO': (-10.18, -48.33)
    }
    
    df = pd.DataFrame([
        {'sigla': sigla, 'latitude': lat, 'longitude': lon}
        for sigla, (lat, lon) in coords.items()
    ])
    
    return df


# =============================================================================
# TESTE DO MÓDULO
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TESTE: CARREGAMENTO DE DADOS CONSOLIDADOS")
    print("=" * 70)
    
    try:
        df = carregar_dados_consolidados()
        print(f"\n✓ Dados carregados com sucesso: {len(df)} estados\n")
        
        print("AMOSTRA DOS DADOS:")
        print("-" * 70)
        colunas_exibir = ['sigla', 'estado', 'taxa_mortes_100k', 
                          'orcamento_2022_milhoes', 'elasticidade']
        print(df[colunas_exibir].to_string(index=False))
        
        print("\n" + "=" * 70)
        print("ESTATÍSTICAS DESCRITIVAS")
        print("=" * 70)
        print(df[['taxa_mortes_100k', 'orcamento_2022_milhoes', 
                  'gasto_per_capita', 'elasticidade']].describe())
        
        print("\n" + "=" * 70)
        print("TOP 5 ESTADOS POR ÍNDICE DE PRIORIDADE")
        print("(Maior taxa de crime / menor investimento)")
        print("=" * 70)
        top5 = df.nlargest(5, 'indice_prioridade')[
            ['sigla', 'estado', 'taxa_mortes_100k', 'gasto_per_capita', 'indice_prioridade']
        ]
        print(top5.to_string(index=False))
        
    except Exception as e:
        print(f"\n✗ Erro ao carregar dados: {e}")
        import traceback
        traceback.print_exc()
