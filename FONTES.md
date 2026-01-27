# Fontes de Dados - Documentação Completa

Este documento descreve em detalhes todas as fontes de dados utilizadas neste projeto, incluindo links diretos para download, metodologia de coleta e limitações conhecidas.

---

## 1. Dados de Violência

### 1.1 Atlas da Violência (IPEA/FBSP)

**O que é:** O Atlas da Violência é uma publicação anual produzida pelo Instituto de Pesquisa Econômica Aplicada (IPEA) em parceria com o Fórum Brasileiro de Segurança Pública (FBSP). Consolida estatísticas de mortes violentas no Brasil desde 1989.

**Dados utilizados:**
- Taxa de homicídios de jovens (15-29 anos) por UF: 1989-2022
- Número absoluto de mortes violentas intencionais por UF: 2022

**Fonte primária:**
- Portal: https://www.ipea.gov.br/atlasviolencia/
- Download de dados: https://www.ipea.gov.br/atlasviolencia/dados-series/40
- Metodologia: https://www.ipea.gov.br/atlasviolencia/arquivos/artigos/5765-atlasdaviolencia2021completo.pdf

**Arquivo no projeto:** `dados/taxa_homicidios_jovens.csv`

**Origem dos dados brutos:** Sistema de Informação sobre Mortalidade (SIM) do Ministério da Saúde, via DATASUS.

---

### 1.2 Mortes Violentas Intencionais (MVI)

**Definição oficial (FBSP):** Soma de:
- Homicídios dolosos
- Latrocínios (roubo seguido de morte)
- Lesões corporais seguidas de morte
- Mortes decorrentes de intervenção policial

**Dados utilizados:** Número de MVI por UF em 2022

**Fonte:**
- Anuário Brasileiro de Segurança Pública 2023, Tabela 1
- Download: https://forumseguranca.org.br/wp-content/uploads/2023/07/anuario-2023.pdf
- Dados em Excel: https://forumseguranca.org.br/estatisticas/

**Arquivo no projeto:** `dados/mortes_populacao_2022.csv`

---

## 2. Dados de Orçamento/Investimento em Segurança Pública

### 2.1 Anuário Brasileiro de Segurança Pública 2023 (FBSP)

**O que é:** Publicação anual do Fórum Brasileiro de Segurança Pública com estatísticas criminais e dados de gestão de segurança pública no Brasil.

**Dados utilizados:**
- Tabela 54: "Despesas realizadas com a Função Segurança Pública"
- Colunas: Total de despesas por UF em 2021 e 2022 (R$ correntes)

**Fonte:**
- Publicação: https://forumseguranca.org.br/anuario-brasileiro-seguranca-publica/
- Download direto (PDF): https://forumseguranca.org.br/wp-content/uploads/2023/07/anuario-2023.pdf
- Tabelas em Excel: https://forumseguranca.org.br/estatisticas/

**Arquivo no projeto:** `dados/anuario_fbsp_2023.xlsx`

**Origem dos dados brutos:** 
- SICONFI (Sistema de Informações Contábeis e Fiscais do Setor Público Brasileiro)
- Secretaria do Tesouro Nacional
- Portal: https://siconfi.tesouro.gov.br/

**Metodologia:**
- Os valores representam a execução orçamentária na Função 06 (Segurança Pública)
- Incluem: polícias civil e militar, bombeiros, administração de segurança
- Valores em R$ correntes do respectivo ano

**Limitações:**
- Tocantins: dados não disponíveis na tabela original (usamos média da região Norte)
- Valores não incluem gastos federais diretos (Polícia Federal, PRF)
- Não refletem efetividade, apenas volume financeiro

---

## 3. Dados Demográficos

### 3.1 População por UF (IBGE)

**Fonte:**
- Projeção populacional IBGE 2022
- Portal: https://www.ibge.gov.br/estatisticas/sociais/populacao.html
- Download: https://sidra.ibge.gov.br/tabela/6579

**Dados utilizados:** População estimada por UF em 2022

**Arquivo no projeto:** Coluna "Populacao" em `dados/mortes_populacao_2022.csv`

---

## 4. Dados Geográficos

### 4.1 GeoJSON dos Estados Brasileiros

**Fonte:** Repositório "Click That Hood" (dados originais do IBGE)
- URL: https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson

**Uso:** Mapa coroplético do Brasil na aba Dashboard

---

## 5. Processamento e Transformações

### 5.1 Cálculo da Taxa de Mortes por 100 mil habitantes

```
taxa_mortes_100k = (mortes_violentas / populacao) × 100.000
```

### 5.2 Cálculo do Gasto Per Capita

```
gasto_per_capita = orcamento_2022 / populacao
```

### 5.3 Cálculo da Elasticidade (por Regressão)

Para cada estado, calculamos a elasticidade crime-investimento usando regressão linear sobre a série histórica 1989-2022:

```python
# Variação percentual do crime
Δ_crime = (taxa_ano_atual - taxa_ano_anterior) / taxa_ano_anterior

# Variação percentual do investimento (proxy: PIB estadual)
Δ_investimento = variação proxy de investimento

# Elasticidade = coeficiente angular da regressão
elasticidade = β₁ da regressão Δ_crime ~ Δ_investimento
```

**Limitação importante:** Como não temos série histórica de orçamento de segurança por estado (apenas 2021-2022), usamos a variação da taxa de homicídios ao longo do tempo como proxy da "eficiência histórica" de cada estado.

---

## 6. Referências Acadêmicas

### Pesquisa Operacional e Programação Linear
- Winston, W. L. (2003). *Operations Research: Applications and Algorithms*. 4th ed. Duxbury Press.
- Hillier, F. S.; Lieberman, G. J. (2015). *Introduction to Operations Research*. 10th ed. McGraw-Hill.

### Simulação Monte Carlo
- Rubinstein, R. Y.; Kroese, D. P. (2016). *Simulation and the Monte Carlo Method*. 3rd ed. Wiley.

### Economia do Crime
- Becker, G. S. (1968). "Crime and Punishment: An Economic Approach". *Journal of Political Economy*, 76(2), 169-217.
- Cerqueira, D.; Lobão, W. (2004). "Determinantes da criminalidade: arcabouços teóricos e resultados empíricos". *Dados*, 47(2), 233-269.

### Análise de Eficiência em Segurança Pública
- Pereira Filho, O. A. (2016). "Eficiência técnica das polícias civis brasileiras". *Revista Brasileira de Segurança Pública*, 10(2).

---

## 7. Contato para Dados

- **IPEA/Atlas da Violência:** atlasviolencia@ipea.gov.br
- **FBSP/Anuário:** contato@forumseguranca.org.br
- **IBGE:** ibge@ibge.gov.br
- **SICONFI:** siconfi@tesouro.gov.br

---

*Documento atualizado em: Janeiro/2026*
