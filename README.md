# Otimização de Recursos de Segurança Pública

Trabalho acadêmico de Pesquisa Operacional que aplica Programação Linear para otimizar a alocação de verbas de segurança pública entre os estados brasileiros.

O projeto usa dados reais do [Atlas da Violência](https://www.ipea.gov.br/atlasviolencia/) (IPEA) e do [Anuário Brasileiro de Segurança Pública](https://forumseguranca.org.br/) (FBSP), cobrindo a série histórica de 1989 a 2022.

## O Problema

Dado um orçamento suplementar limitado, como distribuí-lo entre os 27 estados de forma a **maximizar a redução de mortes violentas**?

A premissa é que existe uma relação (elasticidade) entre investimento em segurança e redução de crimes - relação essa que calculamos a partir dos dados históricos de 34 anos.

## Modelo Matemático

**Variáveis de decisão:** `x_i` = investimento adicional no estado i (R$ milhões)

**Função objetivo:**
```
Min Z = Σ [ C_i × (1 - ε_i × x_i / O_i) ]
```

Onde `C_i` são as mortes violentas, `ε_i` é a elasticidade e `O_i` o orçamento atual.

**Restrições:**
- Orçamento total: `Σ x_i ≤ B`
- Limites por estado: `L_i ≤ x_i ≤ U_i`
- Não-negatividade: `x_i ≥ 0`

Resolvemos via **Simplex** usando PuLP + CBC solver.

## Instalação

```bash
git clone https://github.com/dueiriel/po-atlasviolencia.git
cd po-atlasviolencia

python -m venv venv
source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt

streamlit run app.py
```

Acesse `http://localhost:8501` no navegador.

## Funcionalidades

A aplicação tem 7 abas:

### 1. Dashboard
Visão geral da situação atual - mapa coroplético do Brasil, ranking de estados por taxa de violência, relação entre gasto per capita e criminalidade.

### 2. Otimização
Interface principal. Define-se o orçamento disponível (slider de R$ 1-10 bilhões) e os limites por estado. Ao clicar em "Calcular", o Simplex roda e mostra a alocação ótima com métricas de impacto.

### 3. Comparativo
Gráficos de "antes × depois" mostrando a redução esperada por estado e região. Inclui análise de eficiência (custo por vida salva).

### 4. Análise de Sensibilidade
Responde: "e se o orçamento mudar?". Gera diagrama tornado mostrando quais parâmetros mais afetam o resultado. Calcula shadow prices (valor marginal de R$ 1 adicional).

### 5. Monte Carlo
Simulação estocástica com 500+ cenários. Varia os parâmetros aleatoriamente (dentro de intervalos realistas) para gerar intervalos de confiança de 95%. Útil para entender a incerteza do modelo.

### 6. Backtesting
Validação histórica - usa dados de 2012-2017 para "prever" 2018-2022 e compara com a realidade. O modelo atinge MAPE de ~18% com janela deslizante de 5 anos.

### 7. Multi-Período
Planejamento para vários anos. Compara estratégias: uniforme (mesmo valor todo ano), frontloaded (mais no início) ou backloaded (mais no fim). Spoiler: frontloaded ganha por ~4%.

## Estrutura

```
├── app.py                    # Interface Streamlit
├── dados.py                  # Carrega e processa os CSVs/Excel
├── otimizacao.py             # Modelo PuLP
├── analise_estatistica.py    # Cálculo de elasticidade por regressão
├── sensibilidade.py          # Shadow prices e tornado
├── monte_carlo.py            # Simulação estocástica
├── backtesting.py            # Validação com dados históricos
├── multi_periodo.py          # Otimização em múltiplos anos
├── requirements.txt
└── dados/
    ├── taxa_homicidios_jovens.csv    # 1989-2022
    ├── mortes_populacao_2022.csv
    ├── mortes_violentas_2022.csv
    └── anuario_fbsp_2023.xlsx
```

## Resultados Principais

Com um orçamento hipotético de R$ 5 bilhões:

| Métrica | Valor |
|---------|-------|
| Vidas salvas (estimativa) | ~1.875 |
| IC 95% (Monte Carlo) | [1.604 - 2.452] |
| Custo médio por vida | R$ 2,67 milhões |
| MAPE do backtesting | 17-20% |

Estados prioritários (maior razão crime/investimento atual): BA, PE, CE, MA, PI.

Estados com maior elasticidade histórica: SP, MG, DF - indicando que políticas passadas tiveram efeito mensurável.

## Limitações

- **Elasticidade é uma simplificação.** A relação real entre gasto e crime é muito mais complexa e depende de como o dinheiro é aplicado.
- **Dados de orçamento** só estão disponíveis para 2021-2022 no Anuário.
- **Tocantins** aparece com dados incompletos (não encontrado na tabela do FBSP).
- O modelo assume linearidade, o que pode não valer para investimentos muito grandes.

## Fontes

- [Atlas da Violência](https://www.ipea.gov.br/atlasviolencia/) - IPEA/FBSP
- [Anuário Brasileiro de Segurança Pública](https://forumseguranca.org.br/) - FBSP
- Dados processados a partir de repositórios públicos no GitHub

## Referências

- Winston, W. L. (2003). *Operations Research: Applications and Algorithms*. Duxbury.
- Hillier, F. S.; Lieberman, G. J. (2015). *Introduction to Operations Research*. McGraw-Hill.
- Rubinstein, R. Y. (1981). *Simulation and the Monte Carlo Method*. Wiley.

---

Projeto acadêmico - uso educacional.
