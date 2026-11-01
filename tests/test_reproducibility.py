#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Reproducibility & Anti-Leakage Test Suite
Compatible with both `python3 -m unittest discover -s tests` and `pytest`.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, pearsonr, wilcoxon

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_V1 = REPO_ROOT / "results" / "v1_macro_sentence" / "resultado_experimento_semiotico_v1.csv"
CSV_V2 = REPO_ROOT / "results" / "v2_dsaa_definitive" / "resultado_experimento_semiotico_v2.csv"
DIAG_JSON = REPO_ROOT / "data" / "diagnostic_suite_18.json"
ASSETS_DIR = REPO_ROOT / "assets"


class TestSemioABSAReproducibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df_v1 = pd.read_csv(CSV_V1)
        cls.df_v2 = pd.read_csv(CSV_V2)

    def test_sample_sizes_and_parse_integrity(self) -> None:
        self.assertEqual(len(self.df_v1), 78)
        self.assertEqual(len(self.df_v2), 178)
        self.assertEqual(self.df_v2["sentence_id"].nunique(), 146)
        for col in ("A_ParseOK", "A2_ParseOK", "B_ParseOK"):
            self.assertTrue(bool(self.df_v2[col].all()), f"Expected 100% JSON parse in {col}")
        self.assertEqual(int(self.df_v2["B_GuardFixed"].sum()), 0)

    def test_zero_data_leakage_guarantee(self) -> None:
        """Verifies strict exclusion of legacy `rest16_quad_dev_*` / training splits."""
        ids = self.df_v2["sentence_id"].astype(str)
        self.assertTrue(
            all(
                s.startswith(("rest26_aspect_va_dev_", "lap26_aspect_va_dev_", "diag_"))
                for s in ids
            ),
            "Unexpected split ID detected in v2 benchmark!",
        )
        self.assertEqual(int(ids.str.startswith("rest16_").sum()), 0)

    def test_official_semeval_metrics_match_reported_numbers(self) -> None:
        def rmse_va(prefix: str, frame: pd.DataFrame) -> float:
            dv = frame["gold_v"] - frame[f"{prefix}_Valence"]
            da = frame["gold_a"] - frame[f"{prefix}_Arousal"]
            return float(np.sqrt(np.mean(dv**2 + da**2)))

        self.assertAlmostEqual(rmse_va("A", self.df_v2), 2.1132, places=4)
        self.assertAlmostEqual(rmse_va("A2", self.df_v2), 1.7571, places=4)
        self.assertAlmostEqual(rmse_va("B", self.df_v2), 1.2926, places=4)

        blind = self.df_v2[self.df_v2["source_split"] != "diagnostic_suite"]
        self.assertEqual(len(blind), 160)
        self.assertAlmostEqual(rmse_va("B", blind), 1.3033, places=4)

        pcc_a_b = float(pearsonr(self.df_v2["gold_a"], self.df_v2["B_Arousal"])[0])
        self.assertAlmostEqual(pcc_a_b, 0.5541, places=4)

    def test_statistical_significance_and_zilberberg_minimum(self) -> None:
        _, p_ab = wilcoxon(self.df_v2["Erro_Euclid_A"], self.df_v2["Erro_Euclid_B"])
        _, p_a2b = wilcoxon(self.df_v2["Erro_Euclid_A2"], self.df_v2["Erro_Euclid_B"])
        self.assertLess(p_ab, 1e-18)
        self.assertLess(p_a2b, 1e-17)

        hard = self.df_v2[self.df_v2["slice_type"] == "hard_tensive"]
        ctrl = self.df_v2[self.df_v2["slice_type"] == "direct_standard"]
        fh = (np.abs(hard["gold_v"] - hard["A_Valence"]) >= 1.5) | (hard["gold_polarity"] != hard["A_Polaridade"])
        fc = (np.abs(ctrl["gold_v"] - ctrl["A_Valence"]) >= 1.5) | (ctrl["gold_polarity"] != ctrl["A_Polaridade"])
        odds, p_fish = fisher_exact(
            [[int(fh.sum()), int(len(hard) - fh.sum())], [int(fc.sum()), int(len(ctrl) - fc.sum())]]
        )
        self.assertGreater(odds, 2.2)
        self.assertLess(p_fish, 0.05)

        # Verify CONSTATAÇÃO_ÁTONA holds the global minimum of Arousal
        arousal_means = self.df_v2.groupby("B_Categoria_Tensiva")["B_Arousal"].mean()
        self.assertEqual(arousal_means.idxmin(), "CONSTATAÇÃO_ÁTONA")

    def test_diagnostic_suite_and_figures_present(self) -> None:
        items = json.loads(DIAG_JSON.read_text(encoding="utf-8"))
        self.assertEqual(len(items), 18)
        expected_figs = [
            "fig1_parabola_tensiva_zilberberg.png",
            "fig1_parabola_tensiva_zilberberg.pdf",
            "fig2_ablacao_controlada_ic95.png",
            "fig2_ablacao_controlada_ic95.pdf",
            "fig3_evolucao_v1_macro_vs_v2_micro.png",
            "fig3_evolucao_v1_macro_vs_v2_micro.pdf",
            "fig4_deslocamento_vetorial_casos.png",
            "fig4_deslocamento_vetorial_casos.pdf",
        ]
        for fname in expected_figs:
            self.assertTrue((ASSETS_DIR / fname).exists(), f"Missing figure asset: {fname}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
