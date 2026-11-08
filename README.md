# Apriori em Python para análise de cestas de supermercado

Quais produtos aparecem juntos no carrinho e podem orientar testes de ofertas? Este projeto implementa o algoritmo Apriori com a biblioteca padrão do Python e analisa **9.835 compras reais**, distribuídas em **169 categorias de produtos**, da base [Groceries](https://michael.hahsler.net/arules/reference/Groceries.html).

Com confiança mínima de 30%, aumentar o suporte de 0,5% para 1,0% reduziu o total de **482 para 125 regras**, uma queda de cerca de 74%.

## Resultados

| Suporte mínimo | Regras com confiança ≥ 30% | ≥ 50% | ≥ 70% | Conjuntos frequentes | Tempo do Apriori |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0,5% | 482 | 120 | 1 | 1.001 | 0,13750 s |
| 1,0% | 125 | 15 | 0 | 333 | 0,09857 s |

As seis contagens da execução local coincidiram com os resultados publicados para esses parâmetros na [vignette de análise de cestas do tidylearn](https://mirrors.ibiblio.org/pub/mirrors/pub/mirrors/CRAN/web/packages/tidylearn/vignettes/market-basket.html#choosing-the-thresholds). Os tempos da tabela correspondem à mineração dos conjuntos frequentes em uma execução local, uma vez por nível de suporte, e variam conforme o ambiente. [CSV da comparação](resultados/comparacao_parametros.csv)

### Uma regra na prática

**Outros vegetais + iogurte → leite integral**

Das 427 cestas que continham outros vegetais e iogurte, 219 também continham leite integral. Isso corresponde a **51,3% de confiança**, **2,23% de suporte** e **lift de 2,01**. Na base analisada, a frequência de leite integral nessas cestas foi cerca de duas vezes a frequência no total de compras. [Regras com suporte mínimo de 1% e confiança mínima de 30%](resultados/regras_supp_0_010_conf_0_30.csv)

O supermercado poderia testar uma sugestão de leite integral no carrinho online e comparar a taxa de inclusão do produto entre clientes que receberam a sugestão e um grupo de controle. A regra descreve uma associação em compras observadas durante 30 dias. O efeito da sugestão sobre as vendas dependeria do teste.

## Como funciona

O Apriori começa pelos produtos frequentes e gera candidatos de tamanho crescente. Se algum subconjunto de um candidato ficar abaixo do suporte mínimo, o candidato é descartado. A implementação gera regras com um único produto no consequente e calcula três métricas:

| Métrica | O que mede |
| --- | --- |
| Suporte | Proporção de todas as cestas que contêm os itens dos dois lados da regra |
| Confiança | Proporção das cestas com o antecedente que também contêm o consequente |
| Lift | Razão entre a confiança e a frequência do consequente na base inteira |

## Como reproduzir

Na pasta do projeto, com **Python 3.10 ou superior**, execute:

```bash
python apriori_supermercado.py
```

Na primeira execução, o script baixa o [CSV da base Groceries](https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/groceries.csv) se o arquivo estiver ausente. Os arquivos gerados ficam em `resultados/`.

| Arquivo gerado | Conteúdo |
| --- | --- |
| `comparacao_parametros.csv` e `comparacao_parametros.json` | Resumo das seis combinações de parâmetros |
| `regras_*.csv` | Regras, suporte, confiança, lift e número de cestas |
| `estatisticas_base.json` e `desempenho_suporte_*.json` | Perfil da base e desempenho por nível de suporte |

Para gerar também os gráficos PNG, instale `matplotlib` com `python -m pip install matplotlib` e execute o script novamente.

## Créditos

Projeto acadêmico de Introdução à Ciência de Dados, desenvolvido por **Matheus dos Reis Montovaneli** e **Caio Henrique de Sousa Cleim**. A base Groceries foi disponibilizada para o pacote *arules* por Michael Hahsler, Kurt Hornik e Thomas Reutterer. [Documentação e referência bibliográfica da base](https://michael.hahsler.net/arules/reference/Groceries.html)
