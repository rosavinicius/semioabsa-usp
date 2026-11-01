# Methodology, Tensive Scope Spillover & Forensic Audit

## 1. Theoretical Isomorphism: Tensive Semiotics × `(Valence, Arousal)`

Classical Aspect-Based Sentiment Analysis (Hu & Liu, 2004; Pontiki et al., 2014) collapses affect into `{positive, negative, neutral}`. Even when reformulated as continuous regression in Russell's (1980) `(Valence, Arousal)` space (`[1.00, 9.00] × [1.00, 9.00]`, SemEval-2026 Task 3 DimABSA), autoregressive LLMs suffer from **Arousal Collapse**: without an explicit theory of discursive intensity, zero-shot models predict accurate Valence (`PCC_V > 0.76`) while clustering Arousal onto flat constants (`A ≈ 4.0`, `PCC_A ≈ 0.08–0.37`).

Drawing on **Claude Zilberberg's Tensive Semiotics** (*Éléments de sémiotique tensive*, 2006; *La structure tensive*, 2012), **Luiz Tatit** (2011), and **Silva's (2021, 2023) Subjective Quantifications**, we model discursive sanction through the interaction of:
- **Intensity (*Intensidade*):** tempo (*andamento*) × tonicity (*tonicidade*) — homologous to **Arousal (`A`)**;
- **Extensity (*Extensidade*):** concentration × diffusion / value orientation — homologous to **Valence (`V`)**.

Crucially, human affective intensity follows a **non-linear parabolic U-curve** ($A = c_2 V^2 + c_1 V + c_0$): both extreme dysphoria (`EXTINÇÃO`, `MINIMIZAÇÃO`) and extreme euphoria (`SATURAÇÃO`, `REDOBRAMENTO`) exhibit elevated tonicity, while factual neutrality (`CONSTATAÇÃO_ÁTONA`) occupies the global minimum of Arousal.

---

## 2. Discovering & Solving *Tensive Scope Spillover* (`v1` → `v2` DSAA)

### Generation `v1` (Macro-Sentence Scope, `N = 78`)
In our initial controlled experiment (`results/v1_macro_sentence/`), the LLM diagnosed a single tensive category for the **entire review sentence** before scoring the target aspect:
- `RMSE_A` immediately improved by **-41.6%** (`2.2345 → 1.3039`) and `PCC_A` rose from `-0.0521` to `+0.3980`.
- However, `RMSE_V` degraded (`1.0067 → 1.6822`). Forensic inspection revealed that **73.6% of Condition B's squared Valence error came from just 6 concessive multi-aspect sentences**.
- **Mechanism (*Tensive Scope Spillover*):** In *"Although the aluminum chassis feels premium, the motherboard fried in two weeks"*, assigning a sentence-level label of `MINIMIZAÇÃO` bled dysphoric valence onto the conceded subordinate aspect (`aluminum chassis`).

### Generation `v2` (Aspect-Anchored Semiotic Decomposition — DSAA, `N = 178`)
To eliminate scope spillover, **DSAA v2** enforces a strict three-stage causal chain inside a single deterministic JSON generation (`do_sample=False`, `SEED=2026`):
1. **`evidencia_micro_aspecto`:** Extract *only* the local syntactic clause or modifier governing the target `Aspect`, explicitly severing adversative/concessive clauses (`but`, `although`, `even though`, `despite`) targeting other aspects.
2. **`categoria_tensiva`:** Classify that isolated micro-span into one of Silva's (2021) six tensive directions (`EXTINÇÃO`, `MINIMIZAÇÃO`, `CONSTATAÇÃO_ÁTONA`, `ATENUAÇÃO`, `REDOBRAMENTO`, `SATURAÇÃO`).
3. **`polaridade`, `valence`, `arousal`:** Project `(V, A)` within the calibrated tensive anchor of that micro-span.

**Impact:** `RMSE_V` dropped **-46.2%** from `v1` (`1.6822 → 0.9054`), `RMSE_A` dropped to `0.9226`, and joint `RMSE_VA` reached **`1.2926`** (`1.3033` on blind official SemEval-2026 dev).

---

## 3. Triple-Controlled Causal Isolation (`A` vs. `A2` vs. `B`)

A common methodological flaw in LLM prompting papers is comparing a Direct Baseline (`A`) against a multi-step Domain CoT prompt (`B`), confounding **domain theory** with **extra reasoning tokens**.

We rule out this confound via **Condition `A2` (Matched-Budget Lexical CoT Control)**:
- `A2` uses the exact same JSON two-step structure and token budget as `B` (extracting `nota_lexical` about the aspect before outputting polarity, `valence`, and `arousal`), but **without any semiotic concepts or tensive categories**.
- While `A2` yields a modest error reduction (`RMSE_VA: 2.1132 → 1.7571`), its Arousal correlation barely moves (`PCC_A: 0.3731 → 0.3986`).
- Adding Tensive Semiotics in `B` slashes `RMSE_VA` by an additional **`-0.4663` points below `A2`** (`95% Cluster Bootstrap CI: [-0.5889, -0.3434]`, Wilcoxon `p = 6.36e-18`) and boosts `PCC_A` to **`0.5541`** (`+0.1555` over `A2`), isolating the specific causal contribution of Tensive Semiotics.
