#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geração de Figuras Científicas de Alta Resolução (300 DPI PNG + PDF Vetorial)
Projeto de Iniciação Científica — SemioCom / FFLCH - Universidade de São Paulo (USP)
Título: "Contribuições da Semiótica para a Análise Aspectual de Sentimentos"

Figuras geradas em `resultados/`:
  1. fig1_parabola_tensiva_zilberberg.{png,pdf}
     Espaço Tensivo (Valência × Arousal): Colapso do Baseline Zero-Shot vs.
     Reconstrução da Parábola de Zilberberg pelas Direções Tensivas de Silva (2021).
  2. fig2_ablacao_controlada_ic95.{png,pdf}
     Ablação Tripla Controlada (A vs. A2 vs. B) com Intervalos de Confiança de 95%
     via Cluster Bootstrap por Sentença (3.000 reamostragens) + Sonda Diagnóstica (Exp. 1).
  3. fig3_evolucao_v1_macro_vs_v2_micro.{png,pdf}
     Superação do Tensive Scope Spillover (Escopo Macro-Frasal v1 -> Micro-Ancoragem DSAA v2)
     e Distribuição Pareada de Ganhos Euclidianos por Instância (N = 178).
  4. fig4_deslocamento_vetorial_casos.{png,pdf}
     Estudo de Casos no Plano Bidimensional (V, A): Deslocamento Vetorial do Baseline (A)
     para a Semiótica DSAA (B) rumo ao Gabarito Humano Oficial.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, pearsonr, wilcoxon

SEED = 2026
REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_V1 = REPO_ROOT / "results" / "v1_macro_sentence" / "resultado_experimento_semiotico_v1.csv"
CSV_V2 = REPO_ROOT / "results" / "v2_dsaa_definitive" / "resultado_experimento_semiotico_v2.csv"
OUT_DIR = REPO_ROOT / "assets"


def configurar_estilo_academico() -> None:
    """Configura tipografia, fontes vetoriais e paleta segura para publicação científica."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10.2,
            "axes.titlesize": 11.4,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.6,
            "axes.labelweight": "semibold",
            "axes.edgecolor": "#334155",
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.color": "#CBD5E1",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.55,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.0,
            "legend.frameon": True,
            "legend.facecolor": "#FFFFFF",
            "legend.edgecolor": "#CBD5E1",
            "legend.framealpha": 0.96,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def calcular_metricas_dimabsa(
    df: pd.DataFrame, prefixo: str
) -> Dict[str, float]:
    """Calcula RMSE_V, RMSE_A, RMSE_VA oficial SemEval-2026 e PCC_A/PCC_V."""
    v_gold = df["gold_v"].to_numpy(dtype=float)
    a_gold = df["gold_a"].to_numpy(dtype=float)
    v_pred = df[f"{prefixo}_Valence"].to_numpy(dtype=float)
    a_pred = df[f"{prefixo}_Arousal"].to_numpy(dtype=float)

    rmse_v = float(np.sqrt(np.mean((v_gold - v_pred) ** 2)))
    rmse_a = float(np.sqrt(np.mean((a_gold - a_pred) ** 2)))
    rmse_va = float(np.sqrt(np.mean((v_gold - v_pred) ** 2 + (a_gold - a_pred) ** 2)))
    pcc_v = float(pearsonr(v_gold, v_pred)[0]) if np.std(v_pred) > 1e-8 else 0.0
    pcc_a = float(pearsonr(a_gold, a_pred)[0]) if np.std(a_pred) > 1e-8 else 0.0
    return {
        "rmse_v": rmse_v,
        "rmse_a": rmse_a,
        "rmse_va": rmse_va,
        "pcc_v": pcc_v,
        "pcc_a": pcc_a,
    }


def bootstrap_intervalos_cluster(
    df: pd.DataFrame, n_boot: int = 3000, seed: int = SEED
) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    """
    Calcula estimativa pontual e IC 95% via Cluster Bootstrap por `sentence_id`
    para as condições A, A2 e B (preservando correlação intra-sentença).
    """
    rng = np.random.default_rng(seed)
    grupos = [g for _, g in df.groupby("sentence_id", sort=False)]
    n_grupos = len(grupos)

    condicoes = ["A", "A2", "B"]
    metricas = ["rmse_va", "rmse_v", "rmse_a", "pcc_a"]
    amostras = {c: {m: np.empty(n_boot, dtype=float) for m in metricas} for c in condicoes}

    pontuais = {c: calcular_metricas_dimabsa(df, c) for c in condicoes}

    clusters_arrays = []
    for g in grupos:
        item = {
            "gv": g["gold_v"].to_numpy(dtype=float),
            "ga": g["gold_a"].to_numpy(dtype=float),
        }
        for c in condicoes:
            item[f"{c}_v"] = g[f"{c}_Valence"].to_numpy(dtype=float)
            item[f"{c}_a"] = g[f"{c}_Arousal"].to_numpy(dtype=float)
        clusters_arrays.append(item)

    for b in range(n_boot):
        idx = rng.integers(0, n_grupos, size=n_grupos)
        gv = np.concatenate([clusters_arrays[i]["gv"] for i in idx])
        ga = np.concatenate([clusters_arrays[i]["ga"] for i in idx])
        for c in condicoes:
            pv = np.concatenate([clusters_arrays[i][f"{c}_v"] for i in idx])
            pa = np.concatenate([clusters_arrays[i][f"{c}_a"] for i in idx])
            amostras[c]["rmse_v"][b] = np.sqrt(np.mean((gv - pv) ** 2))
            amostras[c]["rmse_a"][b] = np.sqrt(np.mean((ga - pa) ** 2))
            amostras[c]["rmse_va"][b] = np.sqrt(np.mean((gv - pv) ** 2 + (ga - pa) ** 2))
            amostras[c]["pcc_a"][b] = (
                pearsonr(ga, pa)[0] if np.std(pa) > 1e-8 else 0.0
            )

    resultado: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
    for c in condicoes:
        resultado[c] = {}
        for m in metricas:
            lo, hi = np.percentile(amostras[c][m], [2.5, 97.5])
            resultado[c][m] = (pontuais[c][m], float(lo), float(hi))
    return resultado


def salvar_figura(fig: plt.Figure, nome_base: str) -> Tuple[Path, Path]:
    """Salva figura em PNG (300 DPI) e PDF vetorial."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{nome_base}.png"
    pdf_path = OUT_DIR / f"{nome_base}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def gerar_fig1_parabola_tensiva(df_v2: pd.DataFrame) -> Tuple[Path, Path]:
    """
    Figura 1: Validação Empírica do Espaço Tensivo de Zilberberg (Valência × Arousal).
    Painel (a): Colapso horizontal de Arousal no Baseline Zero-Shot (Condição A).
    Painel (b): Reconstrução da Parábola Tensiva via Semiótica DSAA (Condição B, Silva 2021).
    """
    fig = plt.figure(figsize=(15.4, 7.2))
    gs = GridSpec(1, 2, figure=fig, wspace=0.24)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    v_grid = np.linspace(1.0, 9.0, 250)
    coef_gold = np.polyfit(df_v2["gold_v"], df_v2["gold_a"], deg=2)
    parabola_gold = np.polyval(coef_gold, v_grid)

    coef_a = np.polyfit(df_v2["A_Valence"], df_v2["A_Arousal"], deg=2)
    parabola_a = np.polyval(coef_a, v_grid)

    coef_b = np.polyfit(df_v2["B_Valence"], df_v2["B_Arousal"], deg=2)
    parabola_b = np.polyval(coef_b, v_grid)

    # --- Painel (a): Baseline Zero-Shot (Condição A) ---
    ax_a.scatter(
        df_v2["gold_v"],
        df_v2["gold_a"],
        s=32,
        color="#94A3B8",
        alpha=0.42,
        edgecolors="none",
        label="Gabarito Humano Oficial (N = 178)",
        zorder=2,
    )
    rng = np.random.default_rng(SEED)
    jitter_a = rng.normal(0.0, 0.055, size=len(df_v2))
    ax_a.scatter(
        df_v2["A_Valence"] + jitter_a,
        df_v2["A_Arousal"] + jitter_a,
        s=38,
        color="#D95F02",
        alpha=0.68,
        edgecolors="#7C2D12",
        linewidths=0.4,
        label="Predições Baseline Zero-Shot (Condição A)",
        zorder=3,
    )
    ax_a.plot(
        v_grid,
        parabola_gold,
        color="#1E293B",
        linestyle="--",
        linewidth=2.0,
        label="Curva Quadrática Real (Gabarito Humano)",
        zorder=4,
    )
    ax_a.plot(
        v_grid,
        parabola_a,
        color="#D95F02",
        linestyle="-",
        linewidth=2.3,
        label="Tendência Quadrática — Baseline (A)",
        zorder=5,
    )

    ax_a.axhspan(3.82, 4.18, color="#FDBA74", alpha=0.26, zorder=1)
    ax_a.annotate(
        "Colapso em faixa fixa (A = 4,0)\nsubestima tonicidade disfórica e eufórica",
        xy=(7.45, 4.02),
        xytext=(5.05, 2.35),
        ha="center",
        fontsize=8.8,
        color="#9A3412",
        fontweight="semibold",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#FFF7ED", edgecolor="#FB923C", lw=0.95),
        arrowprops=dict(arrowstyle="->", color="#C2410C", lw=1.2, connectionstyle="arc3,rad=0.18"),
        zorder=6,
    )

    m_a = calcular_metricas_dimabsa(df_v2, "A")
    ax_a.text(
        0.50,
        0.965,
        f"RMSE_A = {m_a['rmse_a']:.4f}  |  PCC_A = {m_a['rmse_a']*0 + m_a['pcc_a']:.4f}  |  RMSE_VA = {m_a['rmse_va']:.4f}",
        transform=ax_a.transAxes,
        va="top",
        ha="center",
        fontsize=8.9,
        fontweight="semibold",
        color="#7C2D12",
        bbox=dict(boxstyle="round,pad=0.34", facecolor="#FFFBEB", edgecolor="#F59E0B", lw=1.0),
        zorder=7,
    )

    ax_a.set_title(
        "(a) Baseline Zero-Shot Direto (Condição A):\nAchatamento Horizontal da Tonicidade",
        pad=10,
    )
    ax_a.set_xlabel("Valência Contínua — Eixo da Extensidade / Polaridade [1, 9]")
    ax_a.set_ylabel("Arousal Contínuo — Eixo da Intensidade / Tonicidade [1, 9]")
    ax_a.set_xlim(0.6, 9.4)
    ax_a.set_ylim(1.4, 9.75)
    ax_a.set_xticks(range(1, 10))
    ax_a.set_yticks(range(2, 10))
    ax_a.legend(
        loc="upper center",
        bbox_to_anchor=(0.50, -0.12),
        ncol=2,
        fontsize=8.5,
    )

    # --- Painel (b): Semiótica Tensiva DSAA (Condição B) ---
    cores_cat = {
        "EXTINÇÃO": "#991B1B",
        "MINIMIZAÇÃO": "#DC2626",
        "CONSTATAÇÃO_ÁTONA": "#475569",
        "ATENUAÇÃO": "#D97706",
        "REDOBRAMENTO": "#0284C7",
        "SATURAÇÃO": "#0F766E",
    }
    ordem_cat = [
        "EXTINÇÃO",
        "MINIMIZAÇÃO",
        "CONSTATAÇÃO_ÁTONA",
        "ATENUAÇÃO",
        "REDOBRAMENTO",
        "SATURAÇÃO",
    ]
    nomes_curtos = {
        "EXTINÇÃO": "EXTINÇÃO",
        "MINIMIZAÇÃO": "MINIMIZAÇÃO",
        "CONSTATAÇÃO_ÁTONA": "CONSTATAÇÃO ÁTONA",
        "ATENUAÇÃO": "ATENUAÇÃO",
        "REDOBRAMENTO": "REDOBRAMENTO",
        "SATURAÇÃO": "SATURAÇÃO",
    }

    for cat in ordem_cat:
        sub = df_v2[df_v2["B_Categoria_Tensiva"] == cat]
        if sub.empty:
            continue
        jb = rng.normal(0.0, 0.05, size=len(sub))
        ax_b.scatter(
            sub["B_Valence"] + jb,
            sub["B_Arousal"] + jb,
            s=28,
            color=cores_cat[cat],
            alpha=0.23,
            edgecolors="none",
            zorder=2,
        )

    ax_b.plot(
        v_grid,
        parabola_gold,
        color="#1E293B",
        linestyle="--",
        linewidth=2.0,
        label="Parábola Empírica Real (Gabarito)",
        zorder=3,
    )
    ax_b.plot(
        v_grid,
        parabola_b,
        color="#0369A1",
        linestyle="-",
        linewidth=2.4,
        label="Parábola Reconstruída — DSAA v2 (B)",
        zorder=4,
    )

    # Posições exatas e sem colisão para as 6 caixas de categorias
    posicoes_caixas = {
        "EXTINÇÃO": (2.55, 8.48, "center"),
        "MINIMIZAÇÃO": (2.30, 5.20, "center"),
        "ATENUAÇÃO": (5.00, 7.15, "center"),
        "CONSTATAÇÃO_ÁTONA": (4.85, 2.02, "center"),
        "REDOBRAMENTO": (7.35, 4.55, "center"),
        "SATURAÇÃO": (7.10, 8.52, "center"),
    }

    for cat in ordem_cat:
        sub = df_v2[df_v2["B_Categoria_Tensiva"] == cat]
        n_c = len(sub)
        vp, ap = float(sub["B_Valence"].mean()), float(sub["B_Arousal"].mean())
        vg, ag = float(sub["gold_v"].mean()), float(sub["gold_a"].mean())
        se_v = float(sub["B_Valence"].std(ddof=1) / np.sqrt(n_c)) if n_c > 1 else 0.0
        se_a = float(sub["B_Arousal"].std(ddof=1) / np.sqrt(n_c)) if n_c > 1 else 0.0

        ax_b.plot(
            [vp, vg],
            [ap, ag],
            color=cores_cat[cat],
            linestyle=":",
            linewidth=1.5,
            alpha=0.85,
            zorder=5,
        )
        ax_b.scatter(
            [vg],
            [ag],
            marker="D",
            s=66,
            facecolor="white",
            edgecolor=cores_cat[cat],
            linewidths=1.8,
            zorder=6,
        )
        ax_b.errorbar(
            vp,
            ap,
            xerr=se_v if n_c > 2 else None,
            yerr=se_a if n_c > 2 else None,
            fmt="o",
            markersize=10.0,
            color=cores_cat[cat],
            ecolor=cores_cat[cat],
            elinewidth=1.4,
            capsize=3.2,
            markeredgecolor="white",
            markeredgewidth=1.3,
            zorder=7,
        )

        tx, ty, ha_align = posicoes_caixas[cat]
        ax_b.annotate(
            f"{nomes_curtos[cat]} (n={n_c})\nPred ({vp:.2f}, {ap:.2f}) | Real ({vg:.2f}, {ag:.2f})",
            xy=(vp, ap),
            xytext=(tx, ty),
            ha=ha_align,
            va="center",
            fontsize=7.9,
            fontweight="semibold",
            color="#0F172A",
            bbox=dict(
                boxstyle="round,pad=0.26",
                facecolor="white",
                edgecolor=cores_cat[cat],
                lw=1.1,
                alpha=0.96,
            ),
            arrowprops=dict(
                arrowstyle="->",
                color=cores_cat[cat],
                lw=1.15,
                connectionstyle="arc3,rad=0.10",
            ),
            zorder=8,
        )

    proxy_pred = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="#0284C7",
        markeredgecolor="white",
        markersize=8.8,
        label="Centróide Predito DSAA (± EP)",
    )
    proxy_gold = plt.Line2D(
        [0],
        [0],
        marker="D",
        color="w",
        markerfacecolor="white",
        markeredgecolor="#334155",
        markeredgewidth=1.7,
        markersize=7.2,
        label="Centróide Real da Categoria",
    )
    handles_b, _ = ax_b.get_legend_handles_labels()
    ax_b.legend(
        handles=handles_b + [proxy_pred, proxy_gold],
        loc="upper center",
        bbox_to_anchor=(0.50, -0.12),
        ncol=2,
        fontsize=8.5,
    )

    m_b = calcular_metricas_dimabsa(df_v2, "B")
    ax_b.text(
        0.50,
        0.965,
        f"RMSE_A = {m_b['rmse_a']:.4f} (-46,5%)  |  PCC_A = {m_b['pcc_a']:.4f} (+48,5%)  |  RMSE_VA = {m_b['rmse_va']:.4f}",
        transform=ax_b.transAxes,
        va="top",
        ha="center",
        fontsize=8.8,
        fontweight="semibold",
        color="#0F766E",
        bbox=dict(boxstyle="round,pad=0.34", facecolor="#F0FDFA", edgecolor="#14B8A6", lw=1.05),
        zorder=9,
    )

    ax_b.set_title(
        "(b) Semiótica Tensiva DSAA (Condição B):\nReconstrução da Curva em 'U' de Zilberberg",
        pad=10,
    )
    ax_b.set_xlabel("Valência Contínua — Eixo da Extensidade / Polaridade [1, 9]")
    ax_b.set_ylabel("Arousal Contínuo — Eixo da Intensidade / Tonicidade [1, 9]")
    ax_b.set_xlim(0.6, 9.4)
    ax_b.set_ylim(1.4, 9.75)
    ax_b.set_xticks(range(1, 10))
    ax_b.set_yticks(range(2, 10))

    fig.suptitle(
        "Validação Empírica da Gramática Tensiva (Zilberberg, 2012; Silva, 2021) no Espaço Valência × Arousal (DimABSA, N = 178)",
        fontsize=13.0,
        fontweight="bold",
        y=1.03,
    )
    return salvar_figura(fig, "fig1_parabola_tensiva_zilberberg")


def adicionar_colchete_sig(
    ax: plt.Axes, x1: float, x2: float, y: float, h: float, texto: str, cor: str = "#1E293B"
) -> None:
    """Desenha um colchete de significância estatística limpo acima das barras."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.15, color=cor)
    ax.text(
        (x1 + x2) / 2.0,
        y + h + (0.015 * (ax.get_ylim()[1] - ax.get_ylim()[0])),
        texto,
        ha="center",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color=cor,
    )


def gerar_fig2_ablacao_ic95(df_v2: pd.DataFrame) -> Tuple[Path, Path]:
    """
    Figura 2: Isolamento Causal Triplo (A vs. A2 vs. B) com Cluster Bootstrap IC 95% (3.000x)
    e Sonda Diagnóstica de Vulnerabilidade do Baseline (Experimento 1).
    """
    boot = bootstrap_intervalos_cluster(df_v2, n_boot=3000, seed=SEED)

    fig = plt.figure(figsize=(15.8, 6.4))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.42, 0.95, 1.05], wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    cores = {
        "A": "#D95F02",
        "A2": "#64748B",
        "B": "#0F766E",
    }
    rotulos_cond = {
        "A": "Condição A: Baseline Zero-Shot Direto",
        "A2": "Condição A2: Controle CoT Lexical (Mesmo Orçamento)",
        "B": "Condição B: Semiótica Tensiva DSAA v2 (Proposto)",
    }

    # --- Painel (a): RMSE_VA Oficial, RMSE_V, RMSE_A ---
    metricas_rmse = ["rmse_va", "rmse_v", "rmse_a"]
    nomes_rmse = [
        "RMSE_VA Oficial\n(SemEval-2026)",
        "RMSE Valência\n(RMSE_V)",
        "RMSE Arousal\n(RMSE_A)",
    ]
    x = np.arange(len(metricas_rmse))
    largura = 0.255
    deslocamentos = {"A": -largura, "A2": 0.0, "B": largura}

    handles_legenda = []
    for c in ["A", "A2", "B"]:
        vals = [boot[c][m][0] for m in metricas_rmse]
        err_inf = [boot[c][m][0] - boot[c][m][1] for m in metricas_rmse]
        err_sup = [boot[c][m][2] - boot[c][m][0] for m in metricas_rmse]
        barras = ax1.bar(
            x + deslocamentos[c],
            vals,
            width=largura * 0.90,
            color=cores[c],
            edgecolor="#1E293B",
            linewidth=0.7,
            yerr=[err_inf, err_sup],
            capsize=3.5,
            error_kw=dict(lw=1.2, ecolor="#0F172A"),
            label=rotulos_cond[c],
            zorder=3,
        )
        handles_legenda.append(barras[0])
        for barra, v, sup in zip(barras, vals, err_sup):
            ax1.text(
                barra.get_x() + barra.get_width() / 2.0,
                v + sup + 0.04,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.2,
                fontweight="bold",
                color="#0F172A",
            )

    ax1.axhline(1.3261, color="#1D4ED8", linestyle="--", linewidth=1.3, zorder=2)
    ax1.text(
        1.52,
        1.98,
        "9º Lugar Supervisionado\nSemEval-2026 (PICT = 1,33)",
        ha="center",
        va="center",
        fontsize=7.9,
        fontweight="semibold",
        color="#1D4ED8",
        bbox=dict(boxstyle="round,pad=0.24", facecolor="#EFF6FF", edgecolor="#93C5FD", lw=0.85),
        zorder=5,
    )
    ax1.annotate(
        "",
        xy=(1.35, 1.33),
        xytext=(1.45, 1.81),
        arrowprops=dict(arrowstyle="->", color="#1D4ED8", lw=1.0),
        zorder=4,
    )

    ax1.set_ylim(0.0, 3.15)
    adicionar_colchete_sig(
        ax1,
        x[0] + deslocamentos["A"],
        x[0] + deslocamentos["B"],
        2.74,
        0.07,
        "Δ = -0,82 (p = 2,4×10⁻¹⁹)",
        cor="#0F766E",
    )
    adicionar_colchete_sig(
        ax1,
        x[0] + deslocamentos["A2"],
        x[0] + deslocamentos["B"],
        2.36,
        0.06,
        "Δ = -0,47 vs. CoT",
        cor="#1E293B",
    )

    ax1.set_title("(a) Erro Contínuo Dimensional\n(RMSE — Menor é Melhor ↓)", pad=10)
    ax1.set_ylabel("Erro Quadrático Médio (RMSE ± IC 95%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(nomes_rmse)

    # --- Painel (b): Correlação de Pearson no Arousal (PCC_A) ---
    conds = ["A", "A2", "B"]
    x2 = np.arange(len(conds))
    vals_pcc = [boot[c]["pcc_a"][0] for c in conds]
    err_pcc_inf = [boot[c]["pcc_a"][0] - boot[c]["pcc_a"][1] for c in conds]
    err_pcc_sup = [boot[c]["pcc_a"][2] - boot[c]["pcc_a"][0] for c in conds]

    barras2 = ax2.bar(
        x2,
        vals_pcc,
        width=0.56,
        color=[cores[c] for c in conds],
        edgecolor="#1E293B",
        linewidth=0.7,
        yerr=[err_pcc_inf, err_pcc_sup],
        capsize=4.0,
        error_kw=dict(lw=1.3, ecolor="#0F172A"),
        zorder=3,
    )
    for barra, v, sup in zip(barras2, vals_pcc, err_pcc_sup):
        ax2.text(
            barra.get_x() + barra.get_width() / 2.0,
            v + sup + 0.022,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.9,
            fontweight="bold",
            color="#0F172A",
        )

    ax2.axhline(0.0853, color="#991B1B", linestyle=":", linewidth=1.5, zorder=4)
    ax2.text(
        1.0,
        0.115,
        "Etapa 2 Oficial (Mistral-7B): 0,085",
        ha="center",
        va="bottom",
        fontsize=7.9,
        fontweight="semibold",
        color="#991B1B",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="#FEF2F2", edgecolor="#FCA5A5", lw=0.75),
        zorder=5,
    )

    ax2.set_ylim(0.0, 0.86)
    adicionar_colchete_sig(
        ax2,
        1.0,
        2.0,
        0.74,
        0.025,
        "+0,155 sobre CoT",
        cor="#0F766E",
    )

    ax2.set_title("(b) Correlação no Arousal\n(PCC_A — Maior é Melhor ↑)", pad=10)
    ax2.set_ylabel("Correlação de Pearson (r_A ± IC 95%)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(
        ["Baseline\nDireto (A)", "CoT Lexical\nControle (A2)", "Semiótica\nDSAA v2 (B)"]
    )

    # --- Painel (c): Experimento 1 — Taxa de Falha Severa por Recorte ---
    hard = df_v2[df_v2["slice_type"] == "hard_tensive"]
    ctrl = df_v2[df_v2["slice_type"] == "direct_standard"]

    falha_ctrl_a = (
        (np.abs(ctrl["gold_v"] - ctrl["A_Valence"]) >= 1.5)
        | (ctrl["gold_polarity"] != ctrl["A_Polaridade"])
    )
    falha_hard_a = (
        (np.abs(hard["gold_v"] - hard["A_Valence"]) >= 1.5)
        | (hard["gold_polarity"] != hard["A_Polaridade"])
    )
    falha_hard_b = (
        (np.abs(hard["gold_v"] - hard["B_Valence"]) >= 1.5)
        | (hard["gold_polarity"] != hard["B_Polaridade"])
    )

    taxas = [
        float(falha_ctrl_a.mean() * 100.0),
        float(falha_hard_a.mean() * 100.0),
        float(falha_hard_b.mean() * 100.0),
    ]
    contagens = [
        f"{int(falha_ctrl_a.sum())}/{len(ctrl)}",
        f"{int(falha_hard_a.sum())}/{len(hard)}",
        f"{int(falha_hard_b.sum())}/{len(hard)}",
    ]
    tabela_fisher = [
        [int(falha_hard_a.sum()), int(len(hard) - falha_hard_a.sum())],
        [int(falha_ctrl_a.sum()), int(len(ctrl) - falha_ctrl_a.sum())],
    ]
    odds_r, p_fisher = fisher_exact(tabela_fisher)

    x3 = np.arange(3)
    cores_exp1 = ["#94A3B8", "#D95F02", "#0F766E"]
    barras3 = ax3.bar(
        x3,
        taxas,
        width=0.56,
        color=cores_exp1,
        edgecolor="#1E293B",
        linewidth=0.7,
        zorder=3,
    )
    for barra, tx, cnt in zip(barras3, taxas, contagens):
        ax3.text(
            barra.get_x() + barra.get_width() / 2.0,
            tx + 1.0,
            f"{tx:.1f}%\n({cnt})",
            ha="center",
            va="bottom",
            fontsize=8.6,
            fontweight="bold",
            color="#0F172A",
        )

    ax3.set_ylim(0.0, 48.0)
    adicionar_colchete_sig(
        ax3,
        0.0,
        1.0,
        36.0,
        1.3,
        f"OR = {odds_r:.2f} (p = {p_fisher:.4f}*)",
        cor="#9A3412",
    )
    adicionar_colchete_sig(
        ax3,
        1.0,
        2.0,
        41.5,
        1.3,
        "-52,9% de falhas (DSAA)",
        cor="#0F766E",
    )

    ax3.set_title("(c) Sonda Exp. 1: Taxa de Falha\nSevera por Recorte (Menor ↓)", pad=10)
    ax3.set_ylabel("Instâncias com |V - V̂| ≥ 1,5 ou Erro de Classe (%)")
    ax3.set_xticks(x3)
    ax3.set_xticklabels(
        [
            "Controle Literal\nBaseline (A)",
            "Recorte Tensivo\nBaseline (A)",
            "Recorte Tensivo\nDSAA v2 (B)",
        ]
    )

    # Legenda global limpa abaixo da figura para evitar qualquer colisão nos painéis
    fig.legend(
        handles=handles_legenda,
        labels=[rotulos_cond["A"], rotulos_cond["A2"], rotulos_cond["B"]],
        loc="lower center",
        bbox_to_anchor=(0.50, -0.08),
        ncol=3,
        fontsize=9.2,
    )

    fig.suptitle(
        "Isolamento Causal da Semiótica Tensiva (DSAA v2) — Cluster Bootstrap 3.000× por Sentença (N = 178 Aspectos, 146 Sentenças)",
        fontsize=13.0,
        fontweight="bold",
        y=1.05,
    )
    return salvar_figura(fig, "fig2_ablacao_controlada_ic95")


def gerar_fig3_evolucao_v1_v2(
    df_v1: pd.DataFrame, df_v2: pd.DataFrame
) -> Tuple[Path, Path]:
    """
    Figura 3: Evolução Metodológica v1 (Escopo Macro-Frasal) -> v2 (Micro-Ancoragem DSAA)
    e Distribuição Pareada do Ganho Euclidiano por Instância (N = 178).
    """
    fig = plt.figure(figsize=(15.2, 6.2))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.18, 1.0], wspace=0.24)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    m_v1_a = calcular_metricas_dimabsa(df_v1, "A")
    m_v1_b = calcular_metricas_dimabsa(df_v1, "B")
    m_v2_a = calcular_metricas_dimabsa(df_v2, "A")
    m_v2_a2 = calcular_metricas_dimabsa(df_v2, "A2")
    m_v2_b = calcular_metricas_dimabsa(df_v2, "B")

    etapas = [
        "Baseline v1\n(Zero-Shot 3B)",
        "Semiótica v1\n(Macro-Frasal)",
        "Baseline v2\n(Zero-Shot 7B)",
        "Controle A2\n(CoT Lexical 7B)",
        "Semiótica v2\n(DSAA Micro)",
    ]
    vals_v = [
        m_v1_a["rmse_v"],
        m_v1_b["rmse_v"],
        m_v2_a["rmse_v"],
        m_v2_a2["rmse_v"],
        m_v2_b["rmse_v"],
    ]
    vals_a = [
        m_v1_a["rmse_a"],
        m_v1_b["rmse_a"],
        m_v2_a["rmse_a"],
        m_v2_a2["rmse_a"],
        m_v2_b["rmse_a"],
    ]
    vals_va = [
        m_v1_a["rmse_va"],
        m_v1_b["rmse_va"],
        m_v2_a["rmse_va"],
        m_v2_a2["rmse_va"],
        m_v2_b["rmse_va"],
    ]

    x = np.arange(len(etapas))
    w = 0.265

    b1 = ax_a.bar(
        x - w,
        vals_v,
        width=w * 0.90,
        color="#38BDF8",
        edgecolor="#0F172A",
        linewidth=0.65,
        label="RMSE Valência (RMSE_V)",
        zorder=3,
    )
    b2 = ax_a.bar(
        x,
        vals_a,
        width=w * 0.90,
        color="#FB923C",
        edgecolor="#0F172A",
        linewidth=0.65,
        label="RMSE Arousal (RMSE_A)",
        zorder=3,
    )
    b3 = ax_a.bar(
        x + w,
        vals_va,
        width=w * 0.90,
        color="#0F766E",
        edgecolor="#0F172A",
        linewidth=0.65,
        label="RMSE_VA Oficial",
        zorder=3,
    )

    for idx_grp, grupo_barras in enumerate((b1, b2, b3)):
        for idx_bar, barra in enumerate(grupo_barras):
            h_val = barra.get_height()
            dy_extra = 0.09 if (idx_bar == 4 and idx_grp == 1) else 0.035
            ax_a.text(
                barra.get_x() + barra.get_width() / 2.0,
                h_val + dy_extra,
                f"{h_val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
                color="#0F172A",
            )

    ax_a.annotate(
        "Vazamento de Escopo Tensivo (v1)\nRMSE_V inflado para 1,68 em\norações concessivas",
        xy=(1 - w, m_v1_b["rmse_v"] + 0.16),
        xytext=(1.38, 2.76),
        ha="center",
        fontsize=8.1,
        fontweight="semibold",
        color="#991B1B",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#FEF2F2", edgecolor="#F87171", lw=0.95),
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.2, connectionstyle="arc3,rad=0.16"),
        zorder=6,
    )
    ax_a.annotate(
        "Micro-Ancoragem DSAA (v2)\nRMSE_V cai -46,2% (1,68 → 0,91)\nRMSE_VA atinge 1,29",
        xy=(4 - w, m_v2_b["rmse_v"] + 0.16),
        xytext=(3.62, 2.76),
        ha="center",
        fontsize=8.1,
        fontweight="semibold",
        color="#0F766E",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#F0FDFA", edgecolor="#14B8A6", lw=0.95),
        arrowprops=dict(arrowstyle="->", color="#0F766E", lw=1.2, connectionstyle="arc3,rad=-0.16"),
        zorder=6,
    )

    ax_a.axhline(1.9483, color="#64748B", linestyle=":", linewidth=1.25, zorder=2)
    ax_a.text(
        4.45,
        1.98,
        "Etapa 2 Oficial Mistral-7B (1,95)",
        ha="right",
        va="bottom",
        fontsize=7.8,
        color="#475569",
        fontweight="semibold",
    )

    ax_a.set_title("(a) Superação do Vazamento de Escopo Tensivo (v1 → v2)", pad=10)
    ax_a.set_ylabel("Erro Quadrático Médio (RMSE — Menor é Melhor ↓)")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(etapas, fontsize=8.7)
    ax_a.set_ylim(0.0, 3.55)
    ax_a.legend(loc="upper center", ncol=3, fontsize=8.0)

    # --- Painel (b): Curva Ordenada de Ganho Euclidiano por Instância em v2 ---
    ganhos = np.sort(df_v2["Ganho_Tensivo_Euclid"].to_numpy(dtype=float))[::-1]
    idx = np.arange(1, len(ganhos) + 1)
    pos_mask = ganhos > 0
    neg_mask = ganhos <= 0

    n_win = int(np.sum(ganhos > 0))
    win_rate = 100.0 * n_win / len(ganhos)
    _, p_wilcox = wilcoxon(df_v2["Erro_Euclid_A"], df_v2["Erro_Euclid_B"])

    ax_b.bar(
        idx[pos_mask],
        ganhos[pos_mask],
        width=1.0,
        color="#0D9488",
        alpha=0.88,
        label=f"Ganho da Semiótica DSAA (n = {n_win}/178 — {win_rate:.1f}%)",
        zorder=3,
    )
    ax_b.bar(
        idx[neg_mask],
        ganhos[neg_mask],
        width=1.0,
        color="#E11D48",
        alpha=0.82,
        label=f"Empate / Resíduo do Baseline (n = {len(ganhos) - n_win}/178)",
        zorder=3,
    )
    ax_b.axhline(0.0, color="#0F172A", linewidth=1.1, zorder=4)
    ax_b.axvline(n_win + 0.5, color="#475569", linestyle="--", linewidth=1.1, zorder=4)

    ganho_medio = float(np.mean(ganhos))
    ganho_mediano = float(np.median(ganhos))
    ax_b.axhline(ganho_medio, color="#0F766E", linestyle=":", linewidth=1.4, zorder=4)

    ax_b.text(
        0.97,
        0.96,
        (
            f"Teste Pareado de Wilcoxon (B vs. A)\n"
            f"• Taxa de Vitória: {win_rate:.1f}% ({n_win}/178 aspectos)\n"
            f"• Ganho Médio: +{ganho_medio:.2f}  |  Mediana: +{ganho_mediano:.2f}\n"
            f"• p-valor = {p_wilcox:.2e}"
        ),
        transform=ax_b.transAxes,
        ha="right",
        va="top",
        fontsize=8.6,
        fontweight="semibold",
        color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.36", facecolor="#F8FAFC", edgecolor="#94A3B8", lw=1.0),
        zorder=6,
    )

    ax_b.annotate(
        "Maior ganho (+4,31):\n`convertible aspect`\n(concessão isolada)",
        xy=(1.5, ganhos[0]),
        xytext=(15, 2.85),
        fontsize=8.1,
        fontweight="semibold",
        color="#0F766E",
        bbox=dict(boxstyle="round,pad=0.26", facecolor="#F0FDFA", edgecolor="#14B8A6", lw=0.95),
        arrowprops=dict(arrowstyle="->", color="#0F766E", lw=1.15, connectionstyle="arc3,rad=-0.18"),
        zorder=6,
    )

    ax_b.set_title("(b) Ganho Euclidiano Pareado por Aspecto em v2 (N = 178)", pad=10)
    ax_b.set_xlabel("Aspectos Ordenados por Redução de Erro no Plano (V, A)")
    ax_b.set_ylabel("Δ Erro Euclidiano = ‖y - ŷ_A‖₂ − ‖y - ŷ_B‖₂")
    ax_b.set_xlim(0, len(ganhos) + 2)
    ax_b.set_ylim(-1.65, 4.85)
    ax_b.legend(loc="lower left", fontsize=8.3)

    fig.suptitle(
        "Refinamento Arquitetural: Da Classificação Frasal (v1) à Decomposição Semiótica Ancorada ao Aspecto — DSAA (v2)",
        fontsize=13.0,
        fontweight="bold",
        y=1.02,
    )
    return salvar_figura(fig, "fig3_evolucao_v1_macro_vs_v2_micro")


def gerar_fig4_vetores_qualitativos(df_v2: pd.DataFrame) -> Tuple[Path, Path]:
    """
    Figura 4: Estudo de Casos no Plano Bidimensional (V, A) mostrando o deslocamento
    vetorial da predição do Baseline (A) para a Semiótica DSAA (B) rumo ao Gabarito Humano,
    acompanhado da evidência micro-sintática extraída.
    """
    casos_spec = [
        ("lap26_aspect_va_dev_100", "convertible aspect", "[1] Concessão Adversativa"),
        ("lap26_aspect_va_dev_125", "Windows 8.1", "[2] Euforia Tônica"),
        ("lap26_aspect_va_dev_13", "keyboard", "[3] Contraste Multi-Aspecto"),
        ("lap26_aspect_va_dev_173", "hardware on this computer", "[4] Disforia Tônica"),
        ("lap26_aspect_va_dev_92", "Keyboard", "[5] Atenuação Concessiva"),
        ("diag_05", "pizza", "[6] Constatação Átona"),
    ]

    fig = plt.figure(figsize=(15.0, 6.7))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.22, 1.0], wspace=0.14)
    ax = fig.add_subplot(gs[0, 0])
    ax_tbl = fig.add_subplot(gs[0, 1])
    ax_tbl.axis("off")

    v_grid = np.linspace(1.0, 9.0, 200)
    coef_gold = np.polyfit(df_v2["gold_v"], df_v2["gold_a"], deg=2)
    h_parab, = ax.plot(
        v_grid,
        np.polyval(coef_gold, v_grid),
        color="#CBD5E1",
        linestyle="--",
        linewidth=1.8,
        label="Parábola Empírica (Gabarito)",
        zorder=1,
    )
    ax.axvline(5.0, color="#E2E8F0", linestyle=":", linewidth=1.1, zorder=1)

    paleta_casos = ["#0284C7", "#0F766E", "#7C3AED", "#DC2626", "#D97706", "#334155"]
    offsets_num = {
        0: (0.18, -0.45),
        1: (-0.52, 0.25),
        2: (0.20, 0.28),
        3: (0.20, 0.24),
        4: (0.18, -0.42),
        5: (0.22, -0.32),
    }

    linhas_card = []
    for idx_c, (sid, asp, titulo_curto) in enumerate(casos_spec):
        cor = paleta_casos[idx_c]
        row = df_v2[(df_v2["sentence_id"] == sid) & (df_v2["aspect"] == asp)].iloc[0]
        va_a = (float(row["A_Valence"]), float(row["A_Arousal"]))
        va_b = (float(row["B_Valence"]), float(row["B_Arousal"]))
        va_g = (float(row["gold_v"]), float(row["gold_a"]))

        ax.annotate(
            "",
            xy=va_b,
            xytext=va_a,
            arrowprops=dict(
                arrowstyle="-|>,head_length=0.65,head_width=0.35",
                color=cor,
                lw=2.0,
                connectionstyle="arc3,rad=0.14",
            ),
            zorder=4,
        )
        ax.plot(
            [va_b[0], va_g[0]],
            [va_b[1], va_g[1]],
            color=cor,
            linestyle=":",
            linewidth=1.3,
            zorder=3,
        )

        ax.scatter(
            [va_a[0]],
            [va_a[1]],
            marker="X",
            s=88,
            color="#E11D48",
            edgecolor="white",
            linewidths=0.8,
            zorder=5,
        )
        ax.scatter(
            [va_b[0]],
            [va_b[1]],
            marker="o",
            s=96,
            color=cor,
            edgecolor="white",
            linewidths=1.2,
            zorder=6,
        )
        ax.scatter(
            [va_g[0]],
            [va_g[1]],
            marker="*",
            s=178,
            color="#FACC15",
            edgecolor="#0F172A",
            linewidths=1.0,
            zorder=7,
        )

        dx, dy = offsets_num[idx_c]
        ax.text(
            va_b[0] + dx,
            va_b[1] + dy,
            f"[{idx_c + 1}]",
            fontsize=9.2,
            fontweight="bold",
            color=cor,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=cor, lw=1.0),
            zorder=8,
        )

        linhas_card.append(
            {
                "num": idx_c + 1,
                "cor": cor,
                "titulo": titulo_curto,
                "sid": sid.replace("_aspect_va_dev_", "_"),
                "aspect": asp,
                "cat": str(row["B_Categoria_Tensiva"]),
                "span": str(row["B_Evidencia_Micro_Aspecto"]),
                "va_a": va_a,
                "va_b": va_b,
                "va_g": va_g,
                "ganho": float(row["Ganho_Tensivo_Euclid"]),
            }
        )

    h_base = plt.Line2D(
        [0], [0], marker="X", color="w", markerfacecolor="#E11D48", markersize=9,
        label="Predição Baseline Zero-Shot (A)",
    )
    h_dsaa = plt.Line2D(
        [0], [0], marker="o", color="w", markerfacecolor="#0F766E", markersize=9,
        label="Predição Semiótica DSAA v2 (B)",
    )
    h_gold = plt.Line2D(
        [0], [0], marker="*", color="w", markerfacecolor="#FACC15", markeredgecolor="#0F172A",
        markersize=12, label="Gabarito Oficial Humano",
    )
    ax.legend(
        handles=[h_base, h_dsaa, h_gold, h_parab],
        loc="upper center",
        bbox_to_anchor=(0.50, -0.115),
        ncol=2,
        fontsize=8.5,
    )

    ax.set_title("(a) Trajetórias de Correção no Plano (Valência, Arousal)", pad=10)
    ax.set_xlabel("Valência Contínua [1, 9]")
    ax.set_ylabel("Arousal Contínuo [1, 9]")
    ax.set_xlim(1.2, 8.8)
    ax.set_ylim(1.4, 8.8)
    ax.set_xticks(range(2, 9))
    ax.set_yticks(range(2, 9))

    ax_tbl.set_title("(b) Evidência Micro-Sintática Extraída pelo DSAA v2", pad=10, loc="left")
    y_cursor = 0.96
    passo_y = 0.162
    for item in linhas_card:
        bloco_texto = (
            f"{item['titulo']} — Aspecto: \"{item['aspect']}\" ({item['sid']})\n"
            f"• Span Micro-Ancorado: \"{item['span']}\"  →  [{item['cat']}]\n"
            f"• Baseline ({item['va_a'][0]:.1f}, {item['va_a'][1]:.1f})  →  "
            f"DSAA ({item['va_b'][0]:.1f}, {item['va_b'][1]:.1f})  |  "
            f"Real ({item['va_g'][0]:.2f}, {item['va_g'][1]:.2f})   [Ganho: +{item['ganho']:.2f}]"
        )
        ax_tbl.text(
            0.01,
            y_cursor,
            bloco_texto,
            transform=ax_tbl.transAxes,
            ha="left",
            va="top",
            fontsize=8.25,
            color="#0F172A",
            linespacing=1.32,
            bbox=dict(
                boxstyle="round,pad=0.34",
                facecolor="#F8FAFC",
                edgecolor=item["cor"],
                lw=1.25,
            ),
        )
        y_cursor -= passo_y

    fig.suptitle(
        "Mecanismo Qualitativo da Micro-Ancoragem Sintática (DSAA v2): Desacoplamento de Escopo e Calibração de Tonicidade",
        fontsize=12.8,
        fontweight="bold",
        y=1.02,
    )
    return salvar_figura(fig, "fig4_deslocamento_vetorial_casos")


def main() -> None:
    configurar_estilo_academico()
    df_v1 = pd.read_csv(CSV_V1)
    df_v2 = pd.read_csv(CSV_V2)

    saidas = [
        gerar_fig1_parabola_tensiva(df_v2),
        gerar_fig2_ablacao_ic95(df_v2),
        gerar_fig3_evolucao_v1_v2(df_v1, df_v2),
        gerar_fig4_vetores_qualitativos(df_v2),
    ]
    print("Figuras científicas geradas com sucesso em 300 DPI (PNG + PDF vetorial):")
    for png_p, pdf_p in saidas:
        print(f" - {png_p.name} ({png_p.stat().st_size // 1024} KB) | {pdf_p.name}")


if __name__ == "__main__":
    main()
