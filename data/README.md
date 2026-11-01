# Data Card & Anti-Leakage Provenance (`data/`)

This directory documents the data splits used in **SemioABSA (DSAA v2)** and hosts our open-access figurative/tensive diagnostic suite.

## 1. Benchmark Composition (`N = 178` Aspects across `146` Unique Sentences)

| Source Split | ID Prefix | Aspects (`N`) | Provenance & Role |
| :--- | :--- | :---: | :--- |
| **SemEval-2026 Blind Dev (Restaurant)** | `rest26_aspect_va_dev_*` | `80` | Official DimABSA Track A Subtask 1 development set (stratified 50 hardest tensive/concessive + 30 literal control) |
| **SemEval-2026 Blind Dev (Laptop)** | `lap26_aspect_va_dev_*` | `80` | Official DimABSA Track A Subtask 1 development set (stratified 50 hardest tensive/concessive + 30 literal control) |
| **Canonical Tensive Diagnostic Suite** | `diag_01` – `diag_15` | `18` | Open-access controlled suite covering irony, sarcasm, metaphor, multi-aspect concession, extinction, and atonic neutrality (`diagnostic_suite_18.json`) |

## 2. Strict Zero Data-Leakage Guarantee

During Stage 3 exploratory audits, we discovered that legacy local subsets (`first_200_records.csv`, using `rest16_quad_dev_*` IDs) overlapped with `eng_restaurant_train_alltasks`.

To ensure **100% blind evaluation integrity**:
1. Zero examples from `train_alltasks` or `first_200_records.csv` are ever loaded or evaluated in `src/semioabsa_experiment.py`.
2. Automated guards in `tests/test_reproducibility.py` verify `0%` ID and text overlap on every CI run.
3. All LLM runs operate strictly **zero-shot** (no in-context dataset examples leaked into prompts).

## 3. Fetching Raw SemEval-2026 Datasets (Optional for New Runs)

`src/semioabsa_experiment.py` automatically fetches the official `DimABSA2026` public development JSONL files from GitHub if they are not present locally, requiring **zero manual file uploads** when executed in `notebooks/SemioABSA_DSAA_Colab_1Click.ipynb`.
