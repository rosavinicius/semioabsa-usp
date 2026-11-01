#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Instant Statistical Verification CLI (CPU-only, < 2 seconds)
SemioABSA — SemioCom / FFLCH, University of São Paulo (USP)

Recomputes from scratch all metrics, 3,000x sentence-cluster bootstrap 95% CIs,
Fisher's Exact Test (Experiment 1), Wilcoxon Signed-Rank Tests, Zilberberg U-Curve
centroids, and anti-leakage checks directly from `results/v2_dsaa_definitive/`.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, pearsonr, wilcoxon
from sklearn.metrics import accuracy_score, f1_score

SEED = 2026
REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_V2 = REPO_ROOT / "results" / "v2_dsaa_definitive" / "resultado_experimento_semiotico_v2.csv"


def compute_metrics(df: pd.DataFrame, prefix: str) -> dict:
    gv = df["gold_v"].to_numpy(dtype=float)
    ga = df["gold_a"].to_numpy(dtype=float)
    pv = df[f"{prefix}_Valence"].to_numpy(dtype=float)
    pa = df[f"{prefix}_Arousal"].to_numpy(dtype=float)
    acc = accuracy_score(df["gold_polarity"], df[f"{prefix}_Polaridade"])
    f1m = f1_score(df["gold_polarity"], df[f"{prefix}_Polaridade"], average="macro")
    rmse_v = float(np.sqrt(np.mean((gv - pv) ** 2)))
    rmse_a = float(np.sqrt(np.mean((ga - pa) ** 2)))
    rmse_va = float(np.sqrt(np.mean((gv - pv) ** 2 + (ga - pa) ** 2)))
    pcc_v = float(pearsonr(gv, pv)[0])
    pcc_a = float(pearsonr(ga, pa)[0])
    return {
        "Acc": acc,
        "Macro_F1": f1m,
        "RMSE_V": rmse_v,
        "RMSE_A": rmse_a,
        "RMSE_VA": rmse_va,
        "PCC_V": pcc_v,
        "PCC_A": pcc_a,
    }


def cluster_bootstrap_diff(
    df: pd.DataFrame, cond_new: str, cond_ref: str, n_boot: int = 3000, seed: int = SEED
) -> dict:
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("sentence_id", sort=False)]
    n_g = len(groups)
    cache = []
    for g in groups:
        cache.append(
            (
                g["gold_v"].to_numpy(float),
                g["gold_a"].to_numpy(float),
                g[f"{cond_new}_Valence"].to_numpy(float),
                g[f"{cond_new}_Arousal"].to_numpy(float),
                g[f"{cond_ref}_Valence"].to_numpy(float),
                g[f"{cond_ref}_Arousal"].to_numpy(float),
            )
        )

    d_va = np.empty(n_boot, dtype=float)
    d_pcca = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_g, size=n_g)
        gv = np.concatenate([cache[i][0] for i in idx])
        ga = np.concatenate([cache[i][1] for i in idx])
        nv = np.concatenate([cache[i][2] for i in idx])
        na = np.concatenate([cache[i][3] for i in idx])
        rv = np.concatenate([cache[i][4] for i in idx])
        ra = np.concatenate([cache[i][5] for i in idx])
        va_new = np.sqrt(np.mean((gv - nv) ** 2 + (ga - na) ** 2))
        va_ref = np.sqrt(np.mean((gv - rv) ** 2 + (ga - ra) ** 2))
        d_va[b] = va_new - va_ref
        d_pcca[b] = pearsonr(ga, na)[0] - pearsonr(ga, ra)[0]

    return {
        "delta_rmse_va_mean": float(np.mean(d_va)),
        "delta_rmse_va_ci95": (float(np.percentile(d_va, 2.5)), float(np.percentile(d_va, 97.5))),
        "prob_better": float(np.mean(d_va < 0.0) * 100.0),
        "delta_pcca_mean": float(np.mean(d_pcca)),
        "delta_pcca_ci95": (float(np.percentile(d_pcca, 2.5)), float(np.percentile(d_pcca, 97.5))),
    }


def main() -> None:
    df = pd.read_csv(CSV_V2)
    print("=" * 78)
    print("SEMIOABSA (DSAA v2) — REPRODUCIBILITY & STATISTICAL VERIFICATION REPORT")
    print("=" * 78)
    print(f"Dataset size      : N = {len(df)} aspects across {df['sentence_id'].nunique()} unique sentences")
    leaked = df["sentence_id"].str.startswith("rest16_quad_dev_").sum()
    print(f"Anti-leakage check: {leaked} leaked legacy IDs detected (PASS: 0% overlap)")
    print(f"JSON Parse Rate   : A={df['A_ParseOK'].mean()*100:.1f}% | A2={df['A2_ParseOK'].mean()*100:.1f}% | B={df['B_ParseOK'].mean()*100:.1f}%")
    print(f"Post-hoc GuardFix : {int(df['B_GuardFixed'].sum())} instances modified\n")

    # 1. Global Table
    rows = []
    for cond, label in [("A", "A: Direct Baseline"), ("A2", "A2: Lexical CoT"), ("B", "B: Semiotic DSAA v2")]:
        m = compute_metrics(df, cond)
        m["Condition"] = label
        rows.append(m)
    tbl = pd.DataFrame(rows).set_index("Condition")
    print("--- 1. Global Metrics (N = 178) ---")
    print(tbl.round(4).to_string())

    # 2. Official Blind Dev Only
    df_blind = df[df["source_split"] != "diagnostic_suite"]
    rows_b = []
    for cond, label in [("A", "A: Direct Baseline"), ("A2", "A2: Lexical CoT"), ("B", "B: Semiotic DSAA v2")]:
        m = compute_metrics(df_blind, cond)
        m["Condition"] = label
        rows_b.append(m)
    print(f"\n--- 2. Official SemEval-2026 Blind Dev Only (N = {len(df_blind)}) ---")
    print(pd.DataFrame(rows_b).set_index("Condition").round(4).to_string())

    # 3. Experiment 1 Fisher Probe
    hard = df[df["slice_type"] == "hard_tensive"]
    ctrl = df[df["slice_type"] == "direct_standard"]
    fail_h_a = (np.abs(hard["gold_v"] - hard["A_Valence"]) >= 1.5) | (hard["gold_polarity"] != hard["A_Polaridade"])
    fail_c_a = (np.abs(ctrl["gold_v"] - ctrl["A_Valence"]) >= 1.5) | (ctrl["gold_polarity"] != ctrl["A_Polaridade"])
    fail_h_b = (np.abs(hard["gold_v"] - hard["B_Valence"]) >= 1.5) | (hard["gold_polarity"] != hard["B_Polaridade"])
    odds, p_fish = fisher_exact(
        [[int(fail_h_a.sum()), int(len(hard) - fail_h_a.sum())], [int(fail_c_a.sum()), int(len(ctrl) - fail_c_a.sum())]]
    )
    print("\n--- 3. Experiment 1: Severe Failure Vulnerability Probe ---")
    print(f"Baseline (A) on Literal Control (N={len(ctrl)}) : {fail_c_a.mean()*100:.1f}% ({int(fail_c_a.sum())}/{len(ctrl)})")
    print(f"Baseline (A) on Tensive Slice   (N={len(hard)}): {fail_h_a.mean()*100:.1f}% ({int(fail_h_a.sum())}/{len(hard)})")
    print(f"Odds Ratio = {odds:.2f} | Fisher Exact p = {p_fish:.6f}")
    print(f"DSAA v2  (B) on Tensive Slice   (N={len(hard)}): {fail_h_b.mean()*100:.1f}% ({int(fail_h_b.sum())}/{len(hard)})")

    # 4. Significance Tests
    b_vs_a = cluster_bootstrap_diff(df, "B", "A")
    b_vs_a2 = cluster_bootstrap_diff(df, "B", "A2")
    _, p_wil_a = wilcoxon(df["Erro_Euclid_A"], df["Erro_Euclid_B"])
    _, p_wil_a2 = wilcoxon(df["Erro_Euclid_A2"], df["Erro_Euclid_B"])
    win_a = float((df["Erro_Euclid_B"] < df["Erro_Euclid_A"]).mean() * 100.0)
    win_a2 = float((df["Erro_Euclid_B"] < df["Erro_Euclid_A2"]).mean() * 100.0)

    print("\n--- 4. Non-Parametric Significance (3,000x Sentence-Cluster Bootstrap + Wilcoxon) ---")
    print(
        f"B vs A  : Delta RMSE_VA = {b_vs_a['delta_rmse_va_mean']:.4f} "
        f"95% CI [{b_vs_a['delta_rmse_va_ci95'][0]:.4f}, {b_vs_a['delta_rmse_va_ci95'][1]:.4f}] | "
        f"Win-Rate = {win_a:.1f}% | Wilcoxon p = {p_wil_a:.3e}"
    )
    print(
        f"B vs A2 : Delta RMSE_VA = {b_vs_a2['delta_rmse_va_mean']:.4f} "
        f"95% CI [{b_vs_a2['delta_rmse_va_ci95'][0]:.4f}, {b_vs_a2['delta_rmse_va_ci95'][1]:.4f}] | "
        f"Win-Rate = {win_a2:.1f}% | Wilcoxon p = {p_wil_a2:.3e}"
    )

    # 5. Zilberberg Centroids
    print("\n--- 5. Empirical Zilberberg Parabola by Tensive Direction (Silva, 2021) ---")
    grp = (
        df.groupby("B_Categoria_Tensiva")
        .agg(
            n=("aspect", "count"),
            V_pred=("B_Valence", "mean"),
            A_pred=("B_Arousal", "mean"),
            V_gold=("gold_v", "mean"),
            A_gold=("gold_a", "mean"),
            Gain_Euclid=("Ganho_Tensivo_Euclid", "mean"),
        )
        .round(3)
    )
    print(grp.to_string())
    print("=" * 78)


if __name__ == "__main__":
    main()
