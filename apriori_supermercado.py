#!/usr/bin/env python3
"""Regras de associacao para cestas de supermercado (Apriori puro em Python).

Disciplina: Introducao a Ciencia de Dados - estudo reproduzivel.
Dataset: Groceries (9.835 transacoes), do pacote R arules.
Fonte dos dados: https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/groceries.csv

Execute:
    python apriori_supermercado.py
    python apriori_supermercado.py --dataset groceries.csv
    python apriori_supermercado.py --dataset amostra_teste.csv --out resultados_teste

Sem bibliotecas externas para executar o algoritmo; matplotlib e opcional para graficos.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path
from time import perf_counter
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

DATASET_URL = (
    "https://raw.githubusercontent.com/stedy/"
    "Machine-Learning-with-R-datasets/master/groceries.csv"
)
SUPORTES = (0.005, 0.010)
CONFIANCAS = (0.30, 0.50, 0.70)
MAX_TAMANHO_ITEMSET = 10


def download_dataset(path: Path) -> None:
    if path.exists():
        return
    print(f"Baixando dataset real em: {path}")
    req = Request(DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(req, timeout=35) as response:
            data = response.read()
    except (URLError, HTTPError, TimeoutError) as exc:
        raise SystemExit(
            "Nao foi possivel obter o dataset automaticamente.\n"
            "Baixe o CSV manualmente em:\n"
            f"{DATASET_URL}\n"
            f"Salve com o nome {path.name} na pasta do script.\n"
            f"Erro: {exc}"
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def ler_transacoes(path: Path) -> list[tuple[str, ...]]:
    transacoes = []
    with path.open("r", encoding="utf-8-sig", newline="") as arquivo:
        for linha in csv.reader(arquivo):
            itens = tuple(sorted({x.strip() for x in linha if x.strip()}))
            if itens:
                transacoes.append(itens)
    if not transacoes:
        raise ValueError("A base nao possui transacoes validas.")
    return transacoes


def gerar_candidatos(frequentes_anterior: set[tuple[str, ...]], k: int) -> set[tuple[str, ...]]:
    """Passo de juncao e poda da propriedade Apriori."""
    lista = sorted(frequentes_anterior)
    candidatos = set()
    for i, esquerda in enumerate(lista):
        for direita in lista[i + 1 :]:
            if esquerda[:-1] != direita[:-1]:
                # Pares com prefixos distintos nunca formam novo candidato.
                if k > 2:
                    break
                continue
            candidato = esquerda + (direita[-1],)
            if all(sub in frequentes_anterior for sub in combinations(candidato, k - 1)):
                candidatos.add(candidato)
    return candidatos


def apriori(
    transacoes: list[tuple[str, ...]], suporte_min: float,
    tamanho_max: int = MAX_TAMANHO_ITEMSET,
) -> tuple[dict[tuple[str, ...], int], dict]:
    """Conta itemsets frequentes com Apriori, retornando contagens absolutas.

    A poda se baseia em: todo subconjunto de um conjunto frequente e frequente.
    """
    inicio = perf_counter()
    n = len(transacoes)
    minimo_absoluto = math.ceil((suporte_min * n) - 1e-10)
    frequencias_1 = Counter(item for t in transacoes for item in t)
    anteriores = {(item,) for item, qtde in frequencias_1.items() if qtde >= minimo_absoluto}
    contagens = {(item,): frequencias_1[item] for (item,) in anteriores}
    item_frequente = {tupla[0] for tupla in anteriores}
    trans_filtradas = [tuple(i for i in t if i in item_frequente) for t in transacoes]
    camadas = {1: len(anteriores)}
    candidatos_verificados = 0

    for k in range(2, tamanho_max + 1):
        candidatos = gerar_candidatos(anteriores, k)
        if not candidatos:
            break
        candidatos_verificados += len(candidatos)
        encontrados = Counter()
        for cesta in trans_filtradas:
            if len(cesta) >= k:
                encontrados.update(
                    comb for comb in combinations(cesta, k) if comb in candidatos
                )
        atuais = {tup for tup, qtd in encontrados.items() if qtd >= minimo_absoluto}
        if not atuais:
            break
        contagens.update({tup: encontrados[tup] for tup in atuais})
        camadas[k] = len(atuais)
        anteriores = atuais

    resumo = {
        "suporte_minimo": suporte_min,
        "contagem_minima": minimo_absoluto,
        "itemsets_por_tamanho": camadas,
        "total_itemsets_frequentes": len(contagens),
        "total_candidatos_verificados": candidatos_verificados,
        "tempo_apriori_segundos": round(perf_counter() - inicio, 5),
    }
    return contagens, resumo


def gerar_regras(
    contagens: dict[tuple[str, ...], int], n: int, confianca_min: float
) -> list[dict]:
    inicio = perf_counter()
    regras = []
    # Consequente unitario, para reproduzir a configuracao classica de arules::apriori.
    for conjunto, juntos in contagens.items():
        if len(conjunto) < 2:
            continue
        for item_direita in conjunto:
            antecedente = tuple(x for x in conjunto if x != item_direita)
            qtd_esquerda = contagens[antecedente]
            confianca = juntos / qtd_esquerda
            if confianca + 1e-12 < confianca_min:
                continue
            suporte_direita = contagens[(item_direita,)] / n
            regras.append({
                "antecedente": " + ".join(antecedente),
                "consequente": item_direita,
                "suporte": juntos / n,
                "confianca": confianca,
                "lift": confianca / suporte_direita,
                "ocorrencias": juntos,
                "transacoes_antecedente": qtd_esquerda,
                "tempo_geracao_segundos": 0.0,
            })
    duracao = perf_counter() - inicio
    for regra in regras:
        regra["tempo_geracao_segundos"] = round(duracao, 5)
    return sorted(regras, key=lambda r: (-r["lift"], -r["ocorrencias"]))


def salvar_csv(path: Path, registros: list[dict]) -> None:
    if not registros:
        path.write_text("Nenhum resultado encontrado\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as arquivo:
        writer = csv.DictWriter(arquivo, fieldnames=registros[0].keys(), delimiter=";")
        writer.writeheader()
        writer.writerows(registros)


def graficos(resultados: list[dict], top_itens: list[tuple[str, int]], pasta: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    # Comparacao de parametros
    fig, ax = plt.subplots(figsize=(10, 5.5))
    cores = ["#1d7190", "#1baf96", "#f59e0b"]
    for i, confianca in enumerate(CONFIANCAS):
        xs = [j + (i - 1) * .24 for j in range(len(SUPORTES))]
        ys = [next(r["numero_regras"] for r in resultados
                   if r["suporte"] == suporte and r["confianca"] == confianca)
              for suporte in SUPORTES]
        bars = ax.bar(xs, ys, .22, label=f"Confianca >= {confianca:.0%}", color=cores[i])
        ax.bar_label(bars, padding=3, fontsize=10)
    ax.set_xticks(range(len(SUPORTES)), [f"{s:.1%}" for s in SUPORTES])
    ax.set_ylabel("Quantidade de regras")
    ax.set_xlabel("Suporte minimo")
    ax.set_title("Impacto dos parametros no numero de regras")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(pasta / "comparacao_parametros.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    nomes = [i for i, _ in top_itens][:10][::-1]
    counts = [q for _, q in top_itens][:10][::-1]
    bars = ax.barh(nomes, counts, color="#1baf96")
    ax.bar_label(bars, padding=5)
    ax.set_xlabel("Numero de transacoes em que o produto aparece")
    ax.set_title("Os 10 itens mais frequentes")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(pasta / "itens_mais_frequentes.png", dpi=180)
    plt.close(fig)
    return True


def executar(dataset: Path, out: Path) -> None:
    download_dataset(dataset)
    transacoes = ler_transacoes(dataset)
    n = len(transacoes)
    itens = Counter(item for cesta in transacoes for item in cesta)
    media = sum(map(len, transacoes)) / n
    out.mkdir(parents=True, exist_ok=True)

    estatisticas = {
        "dataset": dataset.name,
        "origem_dados": DATASET_URL,
        "numero_transacoes": n,
        "itens_distintos": len(itens),
        "itens_medios_por_transacao": round(media, 3),
        "itens_mais_comuns": itens.most_common(10),
        "algoritmo": "Apriori (implementacao propria em Python)",
        "observacao": "Algoritmo nao supervisionado; nao usa acuracia ou matriz de confusao.",
    }
    (out / "estatisticas_base.json").write_text(
        json.dumps(estatisticas, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    resultados = []
    for suporte in SUPORTES:
        contagens, desempenho = apriori(transacoes, suporte)
        (out / f"desempenho_suporte_{suporte:.3f}.json").write_text(
            json.dumps(desempenho, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        for confianca in CONFIANCAS:
            regras = gerar_regras(contagens, n, confianca)
            sufixo = f"supp_{suporte:.3f}_conf_{confianca:.2f}".replace(".", "_")
            salvar_csv(out / f"regras_{sufixo}.csv", regras)
            resultados.append({
                "suporte": suporte,
                "confianca": confianca,
                "numero_regras": len(regras),
                "itemsets_frequentes": len(contagens),
                "tempo_apriori_segundos": desempenho["tempo_apriori_segundos"],
                "tempo_geracao_regras_segundos": (
                    regras[0]["tempo_geracao_segundos"] if regras else 0
                ),
                "maior_lift": round(max((r["lift"] for r in regras), default=0), 4),
            })

    salvar_csv(out / "comparacao_parametros.csv", resultados)
    (out / "comparacao_parametros.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    graficos_gerados = graficos(resultados, itens.most_common(10), out)

    print("\nBASE DE DADOS")
    print(f"  Transacoes: {n} | Itens: {len(itens)} | Media/cesta: {media:.2f}")
    print("\nSUPORTE   CONFIANCA   REGRAS   ITEMSETS   TEMPO-APRIORI")
    for linha in resultados:
        print(f"{linha['suporte']:>6.1%}    {linha['confianca']:>6.0%}      "
              f"{linha['numero_regras']:>5}      {linha['itemsets_frequentes']:>5}     "
              f"{linha['tempo_apriori_segundos']:.5f}s")
    if n == 9835 and len(itens) == 169:
        referencia = {(0.005, 0.30): 482, (0.005, 0.50): 120, (0.005, 0.70): 1,
                      (0.010, 0.30): 125, (0.010, 0.50): 15, (0.010, 0.70): 0}
        difs = [(r["suporte"], r["confianca"], r["numero_regras"],
                 referencia[(r["suporte"], r["confianca"])])
                for r in resultados if r["numero_regras"] != referencia[(r["suporte"], r["confianca"])]]
        if difs:
            print("\nAVISO: contagens diferentes das publicadas em vignette tidylearn:")
            for s, c, v, ref in difs:
                print(f"  supp {s}, conf {c}: codigo={v}, referencia={ref}")
        else:
            print("\nContagens iguais as referencias publicadas para a base Groceries.")
    print(f"\nResultados salvos em {out.resolve()}")
    if not graficos_gerados:
        print("Para gerar graficos, instale: python -m pip install matplotlib")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mineracao de cestas: Apriori sem mlxtend.")
    pasta = Path(__file__).resolve().parent
    parser.add_argument("--dataset", type=Path, default=pasta / "groceries.csv")
    parser.add_argument("--out", type=Path, default=pasta / "resultados")
    args = parser.parse_args()
    executar(args.dataset, args.out)


if __name__ == "__main__":
    main()
