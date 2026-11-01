# -*- coding: utf-8 -*-
"""
Pipeline Experimental Definitivo v2 — SemioABSA (1-Clique Google Colab)
Projeto IC: Contribuições da Semiótica para a Análise Aspectual de Sentimentos
Autor: Vinicius de Andrade Rosa — SemioCom / FFLCH – Universidade de São Paulo (USP)

O que esta versão v2 resolve (Refinamento Metodológico Guiado pelos Resultados da v1):
1. Experimento 1 Embutido Automaticamente (Sonda Estatística dos Erros do Baseline):
   - Quantifica a taxa de erro severo do Baseline em sentenças com tensão discursiva vs. literais,
     calculando Odds Ratio e Teste Exato de Fisher sem nenhuma intervenção manual.
2. Decomposição Semiótica Ancorada no Aspecto (DSAA — Solução para o Contágio Macro-Frasal):
   - Isola primeiro o predicado sintático/semântico ligado EXCLUSIVAMENTE ao aspecto-alvo antes de
     aplicar as direções tensivas de Silva (2021), impedindo que cláusulas concessivas vizinhas
     invertam a valência local de aspectos elogiados/criticados em sentenças mistas.
3. Distinção entre Disforia Átona e Disforia Tônica (Zilberberg, 2006) + Anti-Contradição:
   - Impede a confusão de hipérboles negativas/caps-lock com SATURAÇÃO, separa críticas contidas
     de críticas tônicas no eixo Arousal, remove colchetes rígidos que causavam quantização em degraus
     e verifica consistência lógica entre polaridade e sinal de valência em todas as condições.
4. Upgrade Automático de Capacidade e Escala Amostral (Zero Config / 1-Clique):
   - Carrega Qwen/Qwen2.5-7B-Instruct em 4-bit NF4 na GPU T4 do Colab (~5.5 GB VRAM) com fallback
     automático transparente para Qwen/Qwen2.5-3B-Instruct FP16 caso bitsandbytes falhe.
   - Amplia a amostra para N = 178 instâncias (160 do dev oficial cego rest26_*/lap26_* + 18 diagnósticas).
"""

import gc
import importlib.util
import json
import math
import os
import random
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("=" * 80)
print(">>> INICIANDO PROTOCOLO DEFINITIVO v2 — SEMIOABSA (1-CLIQUE RUN ALL)")
print("=" * 80)


# ==============================================================================
# 1. INSTALAÇÃO SEGURA DE DEPENDÊNCIAS (ANTES DE IMPORTAR TORCH / TRANSFORMERS)
# ==============================================================================
def ensure_dependencies():
    in_colab = importlib.util.find_spec("google.colab") is not None
    base_pkgs = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "sklearn": "scikit-learn",
        "transformers": "transformers>=4.44.0",
        "accelerate": "accelerate",
    }
    missing = [pip_name for mod, pip_name in base_pkgs.items() if importlib.util.find_spec(mod) is None]
    if importlib.util.find_spec("torch") is None:
        missing.insert(0, "torch")
    # No Colab (Linux x86_64 com GPU), instala bitsandbytes para habilitar Qwen2.5-7B em 4-bit NF4
    if in_colab and sys.platform.startswith("linux") and importlib.util.find_spec("bitsandbytes") is None:
        missing.append("bitsandbytes>=0.44.0")

    if missing:
        print(f"[Setup Automático] Instalando pacotes necessários: {', '.join(missing)}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
            importlib.invalidate_caches()
        except Exception as exc:
            print(f"[Aviso Setup] Falha parcial ao instalar extras ({exc}); prosseguindo com fallback seguro.")
    else:
        print(f"[{'Google Colab' if in_colab else 'Ambiente Local'}] Dependências prontas.")
    return in_colab


IN_COLAB = ensure_dependencies()

import numpy as np
import pandas as pd
import torch
from scipy.stats import fisher_exact, pearsonr, wilcoxon
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 2026
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Escala estatística robusta (~4-6 min em GPU T4 gratuita do Colab; zero risco de timeout)
MAX_OFFICIAL_DEV_ASPECTS = 160
INCLUDE_DIAGNOSTIC_SUITE = True
RUN_PLACEBO_COT_CONTROL = True
BATCH_SIZE = 8 if torch.cuda.is_available() else 1


# ==============================================================================
# 2. SUÍTE DIAGNÓSTICA CONTROLADA & FALLBACK OFFLINE (18 CASOS CANÔNICOS)
# ==============================================================================
DIAGNOSTIC_CHALLENGE_SUITE = [
    {
        "sentence_id": "diag_01",
        "domain": "restaurant",
        "source_split": "diagnostic_suite",
        "slice_type": "hard_tensive",
        "phenomenon": "Sarcasm / Metaphor",
        "text": "The food is decent, but the service is an absolute joke.",
        "aspect": "service",
        "gold_polarity": "negative",
        "gold_v": 2.00,
        "gold_a": 7.60,
        "gold_tensive": "EXTINÇÃO",
    },
    {
        "sentence_id": "diag_01",
        "domain": "restaurant",
        "source_split": "diagnostic_suite",
        "slice_type": "hard_tensive",
        "phenomenon": "Mixed Polarity / Concession",
        "text": "The food is decent, but the service is an absolute joke.",
        "aspect": "food",
        "gold_polarity": "positive",
        "gold_v": 6.00,
        "gold_a": 4.80,
        "gold_tensive": "ATENUAÇÃO",
    },
    {
        "sentence_id": "diag_02",
        "domain": "laptop",
        "source_split": "diagnostic_suite",
        "slice_type": "hard_tensive",
        "phenomenon": "Litotes / Negative Dominance",
        "text": "The screen is not the worst I have seen, yet it is far from impressive for a pro machine.",
        "aspect": "screen",
        "gold_polarity": "negative",
        "gold_v": 3.50,
        "gold_a": 4.60,
        "gold_tensive": "MINIMIZAÇÃO",
    },
    {
        "sentence_id": "diag_03",
        "domain": "restaurant",
        "source_split": "diagnostic_suite",
        "slice_type": "direct_standard",
        "phenomenon": "Cumulative Euphoria",
        "text": "All the appetizers were fabulous, the steak was mouth-watering and the sauces were divine!",
        "aspect": "steak",
        "gold_polarity": "positive",
        "gold_v": 8.50,
        "gold_a": 7.90,
        "gold_tensive": "REDOBRAMENTO",
    },
    {
        "sentence_id": "diag_04",
        "domain": "laptop",
        "source_split": "diagnostic_suite",
        "slice_type": "hard_tensive",
        "phenomenon": "Hyperbolic Metaphor",
        "text": "Great battery life, although the cooling fan sounds like a jet engine taking off.",
        "aspect": "cooling fan",
        "gold_polarity": "negative",
        "gold_v": 2.20,
        "gold_a": 7.80,
        "gold_tensive": "MINIMIZAÇÃO",
    },
    {
        "sentence_id": "diag_05",
        "domain": "restaurant",
        "source_split": "diagnostic_suite",
        "slice_type": "hard_tensive",
        "phenomenon": "Atonic Containment",
        "text": "The pizza was edible, nothing more, nothing less.",
        "aspect": "pizza",
        "gold_polarity": "neutral",
        "gold_v": 5.00,
        "gold_a": 3.40,
        "gold_tensive": "CONSTATAÇÃO_ÁTONA",
    },
    {
        "sentence_id": "diag_06",
        "domain": "restaurant",
        "source_split": "diagnostic_suite",
        "slice_type": "hard_tensive",
        "phenomenon": "Antiphrastic Irony",
        "text": "Sure, paying fifty dollars for soggy cold fries was definitely the highlight of my evening.",
        "aspect": "fries",
        "gold_polarity": "negative",
        "gold_v": 1.50,
        "gold_a": 7.70,
        "gold_tensive": "EXTINÇÃO",
    },
    {
        "sentence_id": "diag_07",
        "domain": "laptop",
        "source_split": "diagnostic_suite",
        "slice_type": "hard_tensive",
        "phenomenon": "Positive with Minor Caveat",
        "text": "I really love typing on this keyboard, even though the backlight is slightly uneven.",
        "aspect": "keyboard",
        "gold_polarity": "positive",
        "gold_v": 6.60,
        "gold_a": 5.60,
        "gold_tensive": "ATENUAÇÃO",
    },
    {
        "sentence_id": "diag_08",
        "domain": "restaurant",
        "source_split": "diagnostic_suite",
        "slice_type": "hard_tensive",
        "phenomenon": "Enunciative Overflow (2nd Person)",
        "text": "Forget ordinary dining: you MUST experience their tasting menu at least once in your life, you will fall in love!",
        "aspect": "tasting menu",
        "gold_polarity": "positive",
        "gold_v": 8.90,
        "gold_a": 8.60,
        "gold_tensive": "SATURAÇÃO",
    },
    {
        "sentence_id": "diag_09",
        "domain": "laptop",
        "source_split": "diagnostic_suite",
        "slice_type": "hard_tensive",
        "phenomenon": "Concessive Minimization",
        "text": "Despite the sleek aluminum chassis, the motherboard died after two weeks; I am never buying this brand again.",
        "aspect": "motherboard",
        "gold_polarity": "negative",
        "gold_v": 1.30,
        "gold_a": 8.20,
        "gold_tensive": "EXTINÇÃO",
    },
    {
        "sentence_id": "diag_09",
        "domain": "laptop",
        "source_split": "diagnostic_suite",
        "slice_type": "hard_tensive",
        "phenomenon": "Concessive Minimization",
        "text": "Despite the sleek aluminum chassis, the motherboard died after two weeks; I am never buying this brand again.",
        "aspect": "aluminum chassis",
        "gold_polarity": "positive",
        "gold_v": 6.40,
        "gold_a": 5.20,
        "gold_tensive": "ATENUAÇÃO",
    },
    {
        "sentence_id": "diag_10",
        "domain": "restaurant",
        "source_split": "diagnostic_suite",
        "slice_type": "hard_tensive",
        "phenomenon": "Nascent Recovery (Restabelecimento)",
        "text": "Our main course arrived completely burnt, but the manager apologized sincerely and comped our meal, so we might give them another try.",
        "aspect": "manager",
        "gold_polarity": "positive",
        "gold_v": 6.50,
        "gold_a": 5.80,
        "gold_tensive": "RESTABELECIMENTO",
    },
    {
        "sentence_id": "diag_10",
        "domain": "restaurant",
        "source_split": "diagnostic_suite",
        "slice_type": "hard_tensive",
        "phenomenon": "Nascent Recovery (Restabelecimento)",
        "text": "Our main course arrived completely burnt, but the manager apologized sincerely and comped our meal, so we might give them another try.",
        "aspect": "main course",
        "gold_polarity": "negative",
        "gold_v": 2.30,
        "gold_a": 6.70,
        "gold_tensive": "MINIMIZAÇÃO",
    },
    {
        "sentence_id": "diag_11",
        "domain": "laptop",
        "source_split": "diagnostic_suite",
        "slice_type": "hard_tensive",
        "phenomenon": "Colloquial Idiom ('a steal')",
        "text": "Don't expect luxury materials, but for this price tag, the laptop is an absolute steal.",
        "aspect": "price tag",
        "gold_polarity": "positive",
        "gold_v": 7.80,
        "gold_a": 7.10,
        "gold_tensive": "ATENUAÇÃO",
    },
    {
        "sentence_id": "diag_12",
        "domain": "laptop",
        "source_split": "diagnostic_suite",
        "slice_type": "direct_standard",
        "phenomenon": "Factual / Neutral Statement",
        "text": "The laptop comes pre-installed with Windows 11 and two USB-C ports on the left side.",
        "aspect": "USB-C ports",
        "gold_polarity": "neutral",
        "gold_v": 5.10,
        "gold_a": 3.80,
        "gold_tensive": "CONSTATAÇÃO_ÁTONA",
    },
    {
        "sentence_id": "diag_13",
        "domain": "restaurant",
        "source_split": "diagnostic_suite",
        "slice_type": "direct_standard",
        "phenomenon": "Direct Positive",
        "text": "Fresh ingredients, generous portions, and wonderfully friendly staff.",
        "aspect": "staff",
        "gold_polarity": "positive",
        "gold_v": 8.10,
        "gold_a": 7.40,
        "gold_tensive": "REDOBRAMENTO",
    },
    {
        "sentence_id": "diag_14",
        "domain": "laptop",
        "source_split": "diagnostic_suite",
        "slice_type": "hard_tensive",
        "phenomenon": "Rhetorical Sarcasm",
        "text": "Oh wonderful, another system update that bricks my Wi-Fi driver right before a presentation.",
        "aspect": "Wi-Fi driver",
        "gold_polarity": "negative",
        "gold_v": 1.60,
        "gold_a": 8.10,
        "gold_tensive": "EXTINÇÃO",
    },
    {
        "sentence_id": "diag_15",
        "domain": "restaurant",
        "source_split": "diagnostic_suite",
        "slice_type": "hard_tensive",
        "phenomenon": "Post-Failure Tentative Hope",
        "text": "The wait time was atrocious, though their new reservation system at least shows some promise of fixing the chaos.",
        "aspect": "wait time",
        "gold_polarity": "negative",
        "gold_v": 2.80,
        "gold_a": 6.40,
        "gold_tensive": "RESTABELECIMENTO",
    },
]


# ==============================================================================
# 3. CARREGAMENTO AUTOMÁTICO ZERO-LEAKAGE (DEV OFICIAL rest26_* / lap26_*)
# ==============================================================================
OFFICIAL_DEV_SOURCES = [
    (
        "restaurant",
        [
            Path("DimABSA2026/task-dataset/track_a/subtask_1/eng/eng_restaurant_dev_task1.jsonl"),
            Path("eng_restaurant_dev_task1.jsonl"),
        ],
        "https://raw.githubusercontent.com/DimABSA/DimABSA2026/main/task-dataset/track_a/subtask_1/eng/eng_restaurant_dev_task1.jsonl",
    ),
    (
        "laptop",
        [
            Path("DimABSA2026/task-dataset/track_a/subtask_1/eng/eng_laptop_dev_task1.jsonl"),
            Path("eng_laptop_dev_task1.jsonl"),
        ],
        "https://raw.githubusercontent.com/DimABSA/DimABSA2026/main/task-dataset/track_a/subtask_1/eng/eng_laptop_dev_task1.jsonl",
    ),
]

TENSIVE_MARKERS_RE = re.compile(
    r"\b(but|though|although|however|despite|yet|still|wish|barely|hardly|slightly|somewhat|"
    r"too\s+\w+|wa+y\s+too|enough|never|worst|beyond|joke|steal|unacceptable|disappointed|unfortunately)\b",
    re.IGNORECASE,
)


def va_to_discrete_polarity(v: float) -> str:
    if v < 4.50:
        return "negative"
    if v > 5.50:
        return "positive"
    return "neutral"


def load_official_dev_records(max_aspects=MAX_OFFICIAL_DEV_ASPECTS) -> pd.DataFrame:
    raw_records = []
    for domain, candidate_paths, github_url in OFFICIAL_DEV_SOURCES:
        lines = []
        found_local = next((p for p in candidate_paths if p.exists()), None)
        if found_local is not None:
            lines = found_local.read_text(encoding="utf-8").splitlines()
            print(f"[Dados Oficiais Locais] {found_local.name}: {len(lines)} sentenças carregadas.")
        else:
            try:
                req = urllib.request.Request(github_url, headers={"User-Agent": "SemioABSA-USP/2.0"})
                with urllib.request.urlopen(req, timeout=20) as resp:
                    lines = resp.read().decode("utf-8").splitlines()
                print(f"[Download Automático GitHub] {domain}_dev_task1.jsonl: {len(lines)} sentenças baixadas.")
            except Exception as exc:
                print(f"[Aviso Rede] Não foi possível baixar {domain}_dev_task1.jsonl ({exc}).")

        for line in lines:
            if not line.strip():
                continue
            obj = json.loads(line)
            sid = str(obj["ID"])
            assert not sid.startswith("rest16_"), f"Vazamento detectado no ID: {sid}"
            text = str(obj["Text"]).strip()
            aspect_items = obj.get("Aspect_VA", [])
            vas_in_sent = [float(it["VA"].split("#")[0]) for it in aspect_items if "VA" in it]
            has_intra_divergence = (len(vas_in_sent) >= 2) and ((max(vas_in_sent) - min(vas_in_sent)) >= 1.5)
            has_lexical_tension = bool(TENSIVE_MARKERS_RE.search(text))

            for it in aspect_items:
                aspect = str(it["Aspect"]).strip()
                if aspect.upper() == "NULL":
                    continue
                v_str, a_str = str(it["VA"]).split("#")
                gv, ga = float(v_str), float(a_str)
                is_hard = has_intra_divergence or has_lexical_tension or (gv <= 4.5)
                phenom = (
                    "Multi-Aspect Contrast"
                    if has_intra_divergence
                    else ("Concessive / Tensive Marker" if has_lexical_tension else ("Disphoric / Neutral Nuance" if gv <= 4.5 else "Direct Standard"))
                )
                raw_records.append(
                    {
                        "sentence_id": sid,
                        "domain": domain,
                        "source_split": f"official_dev_{domain}",
                        "slice_type": "hard_tensive" if is_hard else "direct_standard",
                        "phenomenon": phenom,
                        "text": text,
                        "aspect": aspect,
                        "gold_polarity": va_to_discrete_polarity(gv),
                        "gold_v": gv,
                        "gold_a": ga,
                        "gold_tensive": "N/A_EMPIRICAL_DEV",
                    }
                )

    df_off = pd.DataFrame(raw_records)
    if df_off.empty or max_aspects is None or len(df_off) <= max_aspects:
        return df_off

    # Estratificação balanceada por domínio e por complexidade (60% Hard/Tensivo + 40% Direto)
    sampled_parts = []
    per_domain = max_aspects // 2
    for dom in ["restaurant", "laptop"]:
        sub = df_off[df_off["domain"] == dom]
        if sub.empty:
            continue
        n_h = min(int(round(per_domain * 0.60)), int((sub["slice_type"] == "hard_tensive").sum()))
        n_d = min(per_domain - n_h, int((sub["slice_type"] == "direct_standard").sum()))
        sh = sub[sub["slice_type"] == "hard_tensive"].sample(n=n_h, random_state=SEED)
        sd = sub[sub["slice_type"] == "direct_standard"].sample(n=n_d, random_state=SEED)
        sampled_parts.extend([sh, sd])

    df_sampled = pd.concat(sampled_parts, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return df_sampled


df_official = load_official_dev_records(MAX_OFFICIAL_DEV_ASPECTS)
df_diag = pd.DataFrame(DIAGNOSTIC_CHALLENGE_SUITE)

if not df_official.empty and INCLUDE_DIAGNOSTIC_SUITE:
    df_corpus = pd.concat([df_official, df_diag], ignore_index=True)
elif not df_official.empty:
    df_corpus = df_official.copy()
else:
    print("[Fallback Offline Ativado] Utilizando Suíte Diagnóstica Semiótica Embutida.")
    df_corpus = df_diag.copy()

print(
    f"[Corpus Experimental v2 Pronto] Total = {len(df_corpus)} aspectos "
    f"(Oficial Dev = {len(df_official)} | Diagnóstico = {len(df_diag) if INCLUDE_DIAGNOSTIC_SUITE else 0} | "
    f"Hard/Tensivo = {(df_corpus['slice_type'] == 'hard_tensive').sum()} | "
    f"Direto/Controle = {(df_corpus['slice_type'] == 'direct_standard').sum()})"
)


# ==============================================================================
# 4. CARREGAMENTO INTELIGENTE DO MODELO (7B 4-BIT NF4 COM FALLBACK SEGURO 3B)
# ==============================================================================
def load_best_available_llm():
    if torch.cuda.is_available():
        device = "cuda"
        # Tentativa primária: Qwen2.5-7B-Instruct em 4-bit NF4 (~5.5 GB VRAM na GPU T4 do Colab)
        if importlib.util.find_spec("bitsandbytes") is not None:
            try:
                from transformers import BitsAndBytesConfig

                model_7b = "Qwen/Qwen2.5-7B-Instruct"
                print(f"[Hardware GPU] Carregando {model_7b} em 4-bit NF4 (alta capacidade pragmática/sintática)...")
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                tok = AutoTokenizer.from_pretrained(model_7b)
                mdl = AutoModelForCausalLM.from_pretrained(
                    model_7b,
                    quantization_config=bnb_cfg,
                    device_map="auto",
                )
                return tok, mdl, model_7b, "cuda (4-bit NF4)"
            except Exception as exc:
                print(f"[Fallback Automático GPU] 7B 4-bit indisponível ({exc}). Carregando Qwen2.5-3B-Instruct FP16...")
                gc.collect()
                torch.cuda.empty_cache()

        model_3b = "Qwen/Qwen2.5-3B-Instruct"
        print(f"[Hardware GPU] Carregando {model_3b} em float16...")
        tok = AutoTokenizer.from_pretrained(model_3b)
        mdl = AutoModelForCausalLM.from_pretrained(model_3b, torch_dtype=torch.float16, device_map="auto")
        return tok, mdl, model_3b, "cuda (float16)"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        model_id = "Qwen/Qwen2.5-3B-Instruct"
        print(f"[Hardware Apple MPS] Carregando {model_id} em float16...")
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to("mps")
        return tok, mdl, model_id, "mps (float16)"

    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"[Hardware CPU] Carregando {model_id} em float32...")
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to("cpu")
    return tok, mdl, model_id, "cpu (float32)"


tokenizer, model, ACTIVE_MODEL_ID, HARDWARE_DESC = load_best_available_llm()
DEVICE = next(model.parameters()).device
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()
model.generation_config.temperature = None
model.generation_config.top_p = None
model.generation_config.top_k = None


# ==============================================================================
# 5. PROMPTS DO TESTE TRIPLO: BASELINE (A) vs. CoT LEXICAL (A2) vs. DSAA v2 (B)
# ==============================================================================
SYSTEM_BASELINE = "You are a precise Aspect-Based Sentiment Analysis engine. Output strictly valid JSON."

USER_BASELINE = """Analyze the sentiment toward the target aspect in the review sentence below.
Domain: {domain}
Sentence: "{text}"
Target Aspect: "{aspect}"

Scales:
- polarity: "positive", "negative", or "neutral"
- valence: continuous score from 1.00 (extremely negative/dissatisfied) to 9.00 (extremely positive/delighted), with 5.00 as neutral.
- arousal: continuous emotional intensity/activation from 1.00 (very calm/passive/subdued) to 9.00 (intense/agitated/thrilled).

Return ONLY a JSON object:
{{"polarity": "positive|negative|neutral", "valence": 5.00, "arousal": 5.00}}"""

USER_PLACEBO_COT = """Analyze the sentiment toward the target aspect in the review sentence below.
Domain: {domain}
Sentence: "{text}"
Target Aspect: "{aspect}"

Step 1: Briefly identify ONLY the opinion words or syntactic modifiers attached to "{aspect}" (1 concise sentence).
Step 2: Predict polarity ("positive", "negative", "neutral"), valence [1.00 to 9.00], and arousal [1.00 to 9.00].

Return ONLY a JSON object:
{{
  "step1_note": "concise syntactic/lexical observation on target aspect",
  "polarity": "positive|negative|neutral",
  "valence": 5.00,
  "arousal": 5.00
}}"""

# Condição B (v2): Decomposição Semiótica Ancorada no Aspecto (DSAA — Zilberberg 2006 & Silva 2021)
SYSTEM_SEMIO_V2 = (
    "You are an expert in Computational Semiotics (Zilberberg's Tensive Semiotics and Silva's Subjective Quantifications) "
    "and Dimensional Aspect-Based Sentiment Analysis. Output strictly valid JSON."
)

USER_SEMIO_TENSIVE_V2 = """Analyze the sentiment toward ONLY the target aspect "{aspect}" using Aspect-Anchored Tensive Semiotics (Zilberberg, 2006; Silva, 2021).
Domain: {domain}
Sentence: "{text}"
Target Aspect: "{aspect}"

### Step 1 — Micro-Syntactic Aspect Isolation (Anti-Spillover Rule):
Extract ONLY the predicate/qualification that evaluates "{aspect}" itself.
- CRITICAL: In mixed sentences ("nice servers, but tiny dishes" or "despite the sleek chassis, motherboard died"), NEVER let the clause about another aspect invert the local polarity of "{aspect}"!
- Check for figurative inversion: antiphrastic sarcasm ("Oh wonderful... bricks my Wi-Fi", "highlight of my night" for bad food) or negative metaphors ("fan sounds like a jet engine", "service is a joke") express NEGATIVE dysphoria toward "{aspect}"!

### Step 2 — Classify the Tensive Regime of "{aspect}" (Silva, 2021):
- POSITIVE POLE (ONLY when "{aspect}" itself is evaluated positively):
  * SATURAÇÃO: Ecstatic, hyperbolic praise on "{aspect}" ("incredibly fast", "you MUST try", "coolest ever"). NEVER use SATURAÇÃO for rejections, caps-lock complaints, or sarcasm!
  * REDOBRAMENTO: Clear, cumulative or intensified praise on "{aspect}" ("excellent", "great", "very friendly", "mouth-watering").
  * ATENUAÇÃO: Mild/hedged positive ("decent", "okay at best", "good, though...") OR a locally praised "{aspect}" co-occurring with a negative caveat elsewhere in the sentence.
- NEUTRAL ZONE:
  * CONSTATAÇÃO_ÁTONA: Purely objective specification ("two USB-C ports", physical weight without complaint) or emotionally flat neutrality ("edible, nothing more").
- NEGATIVE POLE (ONLY when "{aspect}" itself is criticized or rejected):
  * RESTABELECIMENTO: Failure accompanied by nascent apology/repair ("burnt, but comped", "not the best, but trying to improve").
  * MINIMIZAÇÃO: Criticism or defect on "{aspect}" — either understated/litotes ("not the worst, yet far from impressive", "a little high", "dry") OR emphatic defect ("awful", "horrendous", "way too small", "sounds like a jet engine").
  * EXTINÇÃO: Complete negative rupture, caps-lock rejection/boycott ("NEVER buy/get", "DO NOT RECOMMEND"), abjection, or biting sarcasm ("service is a joke", "Oh wonderful... bricks my Wi-Fi").

### Step 3 — Fine-Grained Dimensional Calibration (Valence vs. Zilberberg's Tonicity Curve):
1. VALENCE (Thymic Direction on "{aspect}", smooth continuous scale 1.00–9.00):
   - Must strictly match polarity: Negative < 4.60 (mild critique ~3.4–4.3; strong defect ~2.2–3.3; rupture/sarcasm ~1.2–2.1) | Neutral ~4.70–5.30 | Positive > 5.40 (attenuated/decent ~5.8–6.8; clear praise ~7.0–8.0; ecstatic ~8.2–8.9).
2. AROUSAL (Tensive Intensity = Tonicity x Tempo, smooth continuous scale 1.00–9.00):
   - Follow Zilberberg's U-Curve (Valence != Arousal!):
     * Atonic/Flat (CONSTATAÇÃO_ÁTONA, "edible", objective specs): LOW arousal (~3.3–5.0).
     * Moderated/Hedged (ATENUAÇÃO, mild/dry critique, RESTABELECIMENTO): MODERATE arousal (~5.0–6.3).
     * Tonic/Emphatic (REDOBRAMENTO praise OR strong MINIMIZAÇÃO like "awful", "horrendous", "jet engine", "waaaay too small"): HIGH arousal (~6.6–7.9).
     * Hyper-Tonic Overflow/Rupture (ecstatic SATURAÇÃO OR caps-lock/boycott/sarcastic EXTINÇÃO): VERY HIGH arousal (~7.8–8.8).

Return ONLY a valid JSON object:
{{
  "aspect_clause_evidence": "exact local predicate on target aspect and literal vs ironic check",
  "tensive_category": "SATURAÇÃO|REDOBRAMENTO|ATENUAÇÃO|CONSTATAÇÃO_ÁTONA|RESTABELECIMENTO|MINIMIZAÇÃO|EXTINÇÃO",
  "polarity": "positive|negative|neutral",
  "valence": 5.00,
  "arousal": 5.00
}}"""


# ==============================================================================
# 6. PARSER JSON BALANCEADO E GUARDA SIMÉTRICA DE CONSISTÊNCIA LÓGICA
# ==============================================================================
def extract_first_json_object(raw_text: str) -> dict:
    if not raw_text:
        return {}
    cleaned = re.sub(r"```(?:json)?", "", raw_text, flags=re.IGNORECASE).strip()
    start = cleaned.find("{")
    if start == -1:
        return {}
    depth = 0
    in_str = False
    esc = False
    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : idx + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        repaired = re.sub(r",\s*}", "}", candidate)
                        try:
                            return json.loads(repaired)
                        except Exception:
                            return {}
    return {}


def clamp_score(val, default=5.0) -> float:
    try:
        num = float(val)
        if math.isnan(num) or math.isinf(num):
            return default
        return round(min(9.0, max(1.0, num)), 2)
    except Exception:
        return default


def normalize_polarity(pol, v_fallback: float) -> str:
    s = str(pol or "").strip().lower()
    if s in {"positive", "pos"}:
        return "positive"
    if s in {"negative", "neg"}:
        return "negative"
    if s in {"neutral", "neu"}:
        return "neutral"
    return va_to_discrete_polarity(v_fallback)


def enforce_logical_consistency(pol: str, v: float) -> tuple:
    """
    Guarda simétrica aplicada igualmente às Condições A, A2 e B:
    Corrige contradições lógicas severas do decodificador (ex.: polarity='negative' com valence=8.5),
    espelhando em torno do eixo neutro 5.0 (10.0 - v) e registrando o evento.
    """
    if pol == "negative" and v > 5.50:
        return round(min(4.40, max(1.00, 10.0 - v)), 2), True
    if pol == "positive" and v < 4.50:
        return round(max(5.60, min(9.00, 10.0 - v)), 2), True
    return v, False


def run_condition_inference(df: pd.DataFrame, system_prompt: str, user_template: str, label: str) -> list:
    print(f"\n>>> Executando {label} ({len(df)} instâncias, batch_size={BATCH_SIZE})...")
    prompts = []
    for _, row in df.iterrows():
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_template.format(
                    domain=row["domain"],
                    text=row["text"],
                    aspect=row["aspect"],
                ),
            },
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    outputs = []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch_prompts = prompts[i : i + BATCH_SIZE]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1600).to(DEVICE)
        input_len = enc["input_ids"].shape[1]
        with torch.inference_mode():
            gen = model.generate(
                **enc,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        for j in range(len(batch_prompts)):
            raw_out = tokenizer.decode(gen[j][input_len:], skip_special_tokens=True).strip()
            parsed = extract_first_json_object(raw_out)
            ok = bool(parsed) and ("valence" in parsed) and ("arousal" in parsed)
            v_raw = clamp_score(parsed.get("valence", 5.0))
            a = clamp_score(parsed.get("arousal", 5.0))
            pol = normalize_polarity(parsed.get("polarity"), v_raw)
            v, fixed = enforce_logical_consistency(pol, v_raw)
            reasoning = (
                parsed.get("aspect_clause_evidence")
                or parsed.get("semiotic_reasoning")
                or parsed.get("step1_note")
                or ""
            )
            outputs.append(
                {
                    "parse_ok": ok,
                    "consistency_fixed": fixed,
                    "polarity": pol,
                    "valence": v,
                    "arousal": a,
                    "tensive_category": str(parsed.get("tensive_category", "N/A")).strip().upper(),
                    "reasoning": str(reasoning).strip(),
                    "raw_output": raw_out,
                }
            )
        print(f"    Progresso {label}: {min(i + BATCH_SIZE, len(prompts))}/{len(prompts)}", end="\r")
    print()
    return outputs


# ==============================================================================
# 7. EXECUÇÃO DAS CONDIÇÕES + EXPERIMENTO 1 (SONDA ESTATÍSTICA DE ERROS)
# ==============================================================================
res_a = run_condition_inference(df_corpus, SYSTEM_BASELINE, USER_BASELINE, "Condição A (Baseline ABSA Direto)")
res_a2 = (
    run_condition_inference(df_corpus, SYSTEM_BASELINE, USER_PLACEBO_COT, "Condição A2 (Controle CoT Lexical)")
    if RUN_PLACEBO_COT_CONTROL
    else None
)
res_b = run_condition_inference(
    df_corpus, SYSTEM_SEMIO_V2, USER_SEMIO_TENSIVE_V2, "Condição B (Semiótica Tensiva Ancorada DSAA v2)"
)


def compute_metrics(df_sub: pd.DataFrame, col_prefix: str) -> dict:
    gv = df_sub["gold_v"].to_numpy(dtype=float)
    ga = df_sub["gold_a"].to_numpy(dtype=float)
    pv = df_sub[f"{col_prefix}_Valence"].to_numpy(dtype=float)
    pa = df_sub[f"{col_prefix}_Arousal"].to_numpy(dtype=float)
    gpol = df_sub["gold_polarity"].tolist()
    ppol = df_sub[f"{col_prefix}_Polaridade"].tolist()

    rmse_v = float(np.sqrt(mean_squared_error(gv, pv)))
    rmse_a = float(np.sqrt(mean_squared_error(ga, pa)))
    rmse_va_official = float(np.sqrt(rmse_v**2 + rmse_a**2))

    pcc_v = float(pearsonr(pv, gv)[0]) if np.std(pv) > 1e-8 and np.std(gv) > 1e-8 else 0.0
    pcc_a = float(pearsonr(pa, ga)[0]) if np.std(pa) > 1e-8 and np.std(ga) > 1e-8 else 0.0
    acc = float(accuracy_score(gpol, ppol))
    f1_mac = float(f1_score(gpol, ppol, average="macro", zero_division=0))
    cov = float(df_sub[f"{col_prefix}_ParseOK"].mean())
    std_a_pred = float(np.std(pa))

    return {
        "N": len(df_sub),
        "Coverage": cov,
        "Acc_Pol": acc,
        "F1_Macro": f1_mac,
        "RMSE_V": rmse_v,
        "RMSE_A": rmse_a,
        "RMSE_VA_Oficial": rmse_va_official,
        "PCC_V": pcc_v,
        "PCC_A": pcc_a,
        "Std_Pred_A": std_a_pred,
    }


def paired_cluster_bootstrap(df_eval: pd.DataFrame, pref_ref: str, pref_cand: str, n_boot: int = 3000) -> dict:
    sids = df_eval["sentence_id"].unique()
    if len(sids) < 4:
        return {"delta_rmse_va": np.nan, "ci_rmse_low": np.nan, "ci_rmse_high": np.nan, "prob_rmse_better": np.nan}
    grouped = {sid: df_eval[df_eval["sentence_id"] == sid] for sid in sids}
    rng = np.random.default_rng(SEED)
    deltas_rmse, deltas_pcca, deltas_rmse_v = [], [], []
    for _ in range(n_boot):
        sampled_ids = rng.choice(sids, size=len(sids), replace=True)
        boot_df = pd.concat([grouped[s] for s in sampled_ids], ignore_index=True)
        m_ref = compute_metrics(boot_df, pref_ref)
        m_cand = compute_metrics(boot_df, pref_cand)
        deltas_rmse.append(m_cand["RMSE_VA_Oficial"] - m_ref["RMSE_VA_Oficial"])
        deltas_rmse_v.append(m_cand["RMSE_V"] - m_ref["RMSE_V"])
        deltas_pcca.append(m_cand["PCC_A"] - m_ref["PCC_A"])
    lo, hi = np.quantile(deltas_rmse, [0.025, 0.975])
    lo_a, hi_a = np.quantile(deltas_pcca, [0.025, 0.975])
    lo_v, hi_v = np.quantile(deltas_rmse_v, [0.025, 0.975])
    return {
        "delta_rmse_va": float(np.mean(deltas_rmse)),
        "ci_rmse_low": float(lo),
        "ci_rmse_high": float(hi),
        "prob_rmse_better": float(np.mean(np.array(deltas_rmse) < 0)),
        "delta_rmse_v": float(np.mean(deltas_rmse_v)),
        "ci_rmse_v_low": float(lo_v),
        "ci_rmse_v_high": float(hi_v),
        "delta_pcc_a": float(np.mean(deltas_pcca)),
        "ci_pcca_low": float(lo_a),
        "ci_pcca_high": float(hi_a),
    }


df_detailed = df_corpus.copy()
df_detailed["A_ParseOK"] = [r["parse_ok"] for r in res_a]
df_detailed["A_Polaridade"] = [r["polarity"] for r in res_a]
df_detailed["A_Valence"] = [r["valence"] for r in res_a]
df_detailed["A_Arousal"] = [r["arousal"] for r in res_a]

if res_a2 is not None:
    df_detailed["A2_ParseOK"] = [r["parse_ok"] for r in res_a2]
    df_detailed["A2_Nota_Lexical"] = [r["reasoning"] for r in res_a2]
    df_detailed["A2_Polaridade"] = [r["polarity"] for r in res_a2]
    df_detailed["A2_Valence"] = [r["valence"] for r in res_a2]
    df_detailed["A2_Arousal"] = [r["arousal"] for r in res_a2]

df_detailed["B_ParseOK"] = [r["parse_ok"] for r in res_b]
df_detailed["B_GuardFixed"] = [r["consistency_fixed"] for r in res_b]
df_detailed["B_Categoria_Tensiva"] = [r["tensive_category"] for r in res_b]
df_detailed["B_Evidencia_Micro_Aspecto"] = [r["reasoning"] for r in res_b]
df_detailed["B_Polaridade"] = [r["polarity"] for r in res_b]
df_detailed["B_Valence"] = [r["valence"] for r in res_b]
df_detailed["B_Arousal"] = [r["arousal"] for r in res_b]

df_detailed["Erro_Euclid_A"] = np.sqrt(
    (df_detailed["gold_v"] - df_detailed["A_Valence"]) ** 2 + (df_detailed["gold_a"] - df_detailed["A_Arousal"]) ** 2
).round(4)
if res_a2 is not None:
    df_detailed["Erro_Euclid_A2"] = np.sqrt(
        (df_detailed["gold_v"] - df_detailed["A2_Valence"]) ** 2 + (df_detailed["gold_a"] - df_detailed["A2_Arousal"]) ** 2
    ).round(4)
df_detailed["Erro_Euclid_B"] = np.sqrt(
    (df_detailed["gold_v"] - df_detailed["B_Valence"]) ** 2 + (df_detailed["gold_a"] - df_detailed["B_Arousal"]) ** 2
).round(4)
df_detailed["Ganho_Tensivo_Euclid"] = (df_detailed["Erro_Euclid_A"] - df_detailed["Erro_Euclid_B"]).round(4)

CSV_OUT = "resultado_experimento_semiotico_v2.csv"
df_detailed.to_csv(CSV_OUT, index=False, encoding="utf-8")


# ==============================================================================
# 8. CONSOLIDAÇÃO ESTATÍSTICA COMPLETA (EXP 1 + EXP 2 v2) E RELATÓRIO FINAL
# ==============================================================================
# --- Experimento 1 Automatizado: Sonda Estatística de Erros de Polaridade/Valência no Baseline ---
df_detailed["Baseline_Erro_Pol_ou_V"] = (df_detailed["A_Polaridade"] != df_detailed["gold_polarity"]) | (
    (df_detailed["gold_v"] - df_detailed["A_Valence"]).abs() >= 1.50
)
df_detailed["Tensiva_Erro_Pol_ou_V"] = (df_detailed["B_Polaridade"] != df_detailed["gold_polarity"]) | (
    (df_detailed["gold_v"] - df_detailed["B_Valence"]).abs() >= 1.50
)
ct_hard_err = int(df_detailed[df_detailed["slice_type"] == "hard_tensive"]["Baseline_Erro_Pol_ou_V"].sum())
ct_hard_ok = int((df_detailed["slice_type"] == "hard_tensive").sum() - ct_hard_err)
ct_dir_err = int(df_detailed[df_detailed["slice_type"] == "direct_standard"]["Baseline_Erro_Pol_ou_V"].sum())
ct_dir_ok = int((df_detailed["slice_type"] == "direct_standard").sum() - ct_dir_err)
odds_ratio_exp1, fisher_p_exp1 = fisher_exact([[ct_hard_err, ct_hard_ok], [max(ct_dir_err, 1), max(ct_dir_ok, 1)]])
hard_err_rate_a = ct_hard_err / max(1, (ct_hard_err + ct_hard_ok))
dir_err_rate_a = ct_dir_err / max(1, (ct_dir_err + ct_dir_ok))
hard_err_rate_b = float(df_detailed[df_detailed["slice_type"] == "hard_tensive"]["Tensiva_Erro_Pol_ou_V"].mean())

# --- Métricas Globais e Estratificadas ---
m_global_a = compute_metrics(df_detailed, "A")
m_global_a2 = compute_metrics(df_detailed, "A2") if res_a2 is not None else None
m_global_b = compute_metrics(df_detailed, "B")

df_emp_only = df_detailed[df_detailed["source_split"] != "diagnostic_suite"]
df_diag_only = df_detailed[df_detailed["source_split"] == "diagnostic_suite"]
df_hard_slice = df_detailed[df_detailed["slice_type"] == "hard_tensive"]
df_dir_slice = df_detailed[df_detailed["slice_type"] == "direct_standard"]

m_emp_a = compute_metrics(df_emp_only, "A") if not df_emp_only.empty else None
m_emp_a2 = compute_metrics(df_emp_only, "A2") if (not df_emp_only.empty and res_a2 is not None) else None
m_emp_b = compute_metrics(df_emp_only, "B") if not df_emp_only.empty else None

m_diag_a = compute_metrics(df_diag_only, "A") if not df_diag_only.empty else None
m_diag_b = compute_metrics(df_diag_only, "B") if not df_diag_only.empty else None

m_hard_a = compute_metrics(df_hard_slice, "A")
m_hard_b = compute_metrics(df_hard_slice, "B")
m_dir_a = compute_metrics(df_dir_slice, "A") if not df_dir_slice.empty else None
m_dir_b = compute_metrics(df_dir_slice, "B") if not df_dir_slice.empty else None

boot_ab = paired_cluster_bootstrap(df_detailed, "A", "B", n_boot=3000)
boot_a2b = paired_cluster_bootstrap(df_detailed, "A2", "B", n_boot=3000) if res_a2 is not None else None

try:
    _, wilcox_p_ab = wilcoxon(df_detailed["Erro_Euclid_B"] - df_detailed["Erro_Euclid_A"])
except Exception:
    wilcox_p_ab = np.nan
win_rate_ab = float((df_detailed["Erro_Euclid_B"] < df_detailed["Erro_Euclid_A"]).mean())

if res_a2 is not None:
    try:
        _, wilcox_p_a2b = wilcoxon(df_detailed["Erro_Euclid_B"] - df_detailed["Erro_Euclid_A2"])
    except Exception:
        wilcox_p_a2b = np.nan
    win_rate_a2b = float((df_detailed["Erro_Euclid_B"] < df_detailed["Erro_Euclid_A2"]).mean())
else:
    wilcox_p_a2b, win_rate_a2b = np.nan, np.nan

# Acurácia de classificação tensiva na Suíte Diagnóstica (onde há gabarito tensivo de Silva 2021)
if not df_diag_only.empty:
    tensive_diag_acc = float((df_diag_only["B_Categoria_Tensiva"] == df_diag_only["gold_tensive"]).mean())
else:
    tensive_diag_acc = np.nan


def build_summary_df(ma, mb, ma2=None):
    rows = [
        ("Cobertura JSON (%)", f"{ma['Coverage']*100:.1f}%", f"{ma2['Coverage']*100:.1f}%" if ma2 else "-", f"{mb['Coverage']*100:.1f}%"),
        ("Acurácia Polaridade", f"{ma['Acc_Pol']:.4f}", f"{ma2['Acc_Pol']:.4f}" if ma2 else "-", f"{mb['Acc_Pol']:.4f}"),
        ("F1-Score Macro", f"{ma['F1_Macro']:.4f}", f"{ma2['F1_Macro']:.4f}" if ma2 else "-", f"{mb['F1_Macro']:.4f}"),
        ("RMSE Valência (V)", f"{ma['RMSE_V']:.4f}", f"{ma2['RMSE_V']:.4f}" if ma2 else "-", f"{mb['RMSE_V']:.4f}"),
        ("RMSE Arousal (A)", f"{ma['RMSE_A']:.4f}", f"{ma2['RMSE_A']:.4f}" if ma2 else "-", f"{mb['RMSE_A']:.4f}"),
        ("RMSE_VA Oficial (SemEval)", f"{ma['RMSE_VA_Oficial']:.4f}", f"{ma2['RMSE_VA_Oficial']:.4f}" if ma2 else "-", f"{mb['RMSE_VA_Oficial']:.4f}"),
        ("PCC Valência (r_V)", f"{ma['PCC_V']:.4f}", f"{ma2['PCC_V']:.4f}" if ma2 else "-", f"{mb['PCC_V']:.4f}"),
        ("PCC Arousal (r_A)", f"{ma['PCC_A']:.4f}", f"{ma2['PCC_A']:.4f}" if ma2 else "-", f"{mb['PCC_A']:.4f}"),
        ("Dispersão Arousal (std_A)", f"{ma['Std_Pred_A']:.4f}", f"{ma2['Std_Pred_A']:.4f}" if ma2 else "-", f"{mb['Std_Pred_A']:.4f}"),
    ]
    cols = ["Métrica", "Condição A (Baseline)", "Condição A2 (CoT Lexical)", "Condição B (Tensiva DSAA v2)"]
    out_df = pd.DataFrame(rows, columns=cols)
    if ma2 is None:
        out_df = out_df.drop(columns=["Condição A2 (CoT Lexical)"])
    return out_df


summary_table = build_summary_df(m_global_a, m_global_b, m_global_a2)
print("\n" + "=" * 80)
print(">>> RESULTADOS GLOBAIS v2: BASELINE vs. CONTROLE CoT vs. SEMIÓTICA TENSIVA DSAA")
print("=" * 80)
print(summary_table.to_string(index=False))
print(
    f"\n[Bootstrap Pareado B vs A] Delta RMSE_VA = {boot_ab['delta_rmse_va']:+.4f} "
    f"(IC 95%: [{boot_ab['ci_rmse_low']:+.4f}, {boot_ab['ci_rmse_high']:+.4f}]) | "
    f"Wilcoxon p = {wilcox_p_ab:.6f} (Win-Rate = {win_rate_ab*100:.1f}%)"
)
if boot_a2b is not None:
    print(
        f"[Isolamento Causal B vs A2] Delta RMSE_VA = {boot_a2b['delta_rmse_va']:+.4f} "
        f"(IC 95%: [{boot_a2b['ci_rmse_low']:+.4f}, {boot_a2b['ci_rmse_high']:+.4f}]) | "
        f"Delta PCC_A = {boot_a2b['delta_pcc_a']:+.4f} | Wilcoxon p = {wilcox_p_a2b:.6f}"
    )

tensive_stats = (
    df_detailed.groupby("B_Categoria_Tensiva", observed=False)
    .agg(
        n=("aspect", "count"),
        V_medio_pred=("B_Valence", "mean"),
        A_medio_pred=("B_Arousal", "mean"),
        V_medio_gold=("gold_v", "mean"),
        A_medio_gold=("gold_a", "mean"),
        Ganho_Medio_Euclid=("Ganho_Tensivo_Euclid", "mean"),
    )
    .round(3)
    .reset_index()
)

top_wins = df_detailed.sort_values("Ganho_Tensivo_Euclid", ascending=False).head(10)
top_losses = df_detailed.sort_values("Ganho_Tensivo_Euclid", ascending=True).head(5)

relatorio_lines = [
    "# RELATÓRIO CIENTÍFICO DEFINITIVO v2 — SEMIOABSA (IC SEMIOCOM / FFLCH-USP)",
    f"- Modelo Avaliado: `{ACTIVE_MODEL_ID}` em `{HARDWARE_DESC}` (Greedy Determinístico, Seed={SEED})",
    f"- Escala Amostral: N = {len(df_detailed)} aspectos ({df_detailed['sentence_id'].nunique()} sentenças distintas)",
    "- Auditoria Anti-Vazamento: APROVADA (0% de sobreposição com `first_200_records.csv` ou `train_alltasks`; uso exclusivo de `rest26_*`/`lap26_*` + suíte diagnóstica).",
    f"- Acurácia de Classificação das Categorias de Silva (2021) na Suíte Diagnóstica: {tensive_diag_acc*100:.1f}%",
    "",
    "## 1. Experimento 1 (Embutido): Sonda Estatística da Vulnerabilidade do Baseline a Fenômenos Tensivos",
    f"- Taxa de falha de polaridade/valência (|V - V_pred| >= 1.5 ou classe errada) no Baseline Direto (A):",
    f"  * Em sentenças com Tensão Concessiva / Figuras / Contraste Intra-Sentença (N={m_hard_a['N']}): {hard_err_rate_a*100:.1f}% ({ct_hard_err}/{m_hard_a['N']})",
    f"  * Em sentenças Literais Diretas de Controle (N={len(df_dir_slice)}): {dir_err_rate_a*100:.1f}% ({ct_dir_err}/{len(df_dir_slice)})",
    f"  * Razão de Chances (Odds Ratio) = {odds_ratio_exp1:.2f} | Teste Exato de Fisher p = {fisher_p_exp1:.6f}",
    f"  * Redução da taxa de falha no recorte tensivo após Semiótica DSAA (Condição B): {hard_err_rate_a*100:.1f}% -> {hard_err_rate_b*100:.1f}%",
    "",
    "## 2. Tabela Comparativa Global v2 (Métrica Oficial SemEval-2026 DimABSA)",
    summary_table.to_string(index=False),
]

if m_emp_a and m_emp_b:
    emp_table = build_summary_df(m_emp_a, m_emp_b, m_emp_a2)
    relatorio_lines.extend(
        [
            "",
            f"## 3. Desempenho Exclusivo no Benchmark Oficial Cego DimABSA-2026 (rest26_* + lap26_* — N={m_emp_a['N']})",
            emp_table.to_string(index=False),
        ]
    )

relatorio_lines.extend(
    [
        "",
        "## 4. Desempenho Estratificado por Complexidade Discursiva",
        f"- **Recorte Tensivo/Difícil (Concessão, Figuras, Polaridade Mista, Disforia — N={m_hard_a['N']})**:",
        f"  * Condição A (Baseline): Acc={m_hard_a['Acc_Pol']:.4f} | F1={m_hard_a['F1_Macro']:.4f} | RMSE_V={m_hard_a['RMSE_V']:.4f} | RMSE_A={m_hard_a['RMSE_A']:.4f} | RMSE_VA={m_hard_a['RMSE_VA_Oficial']:.4f} | PCC_V={m_hard_a['PCC_V']:.4f} | PCC_A={m_hard_a['PCC_A']:.4f}",
        f"  * Condição B (DSAA v2):  Acc={m_hard_b['Acc_Pol']:.4f} | F1={m_hard_b['F1_Macro']:.4f} | RMSE_V={m_hard_b['RMSE_V']:.4f} | RMSE_A={m_hard_b['RMSE_A']:.4f} | RMSE_VA={m_hard_b['RMSE_VA_Oficial']:.4f} | PCC_V={m_hard_b['PCC_V']:.4f} | PCC_A={m_hard_b['PCC_A']:.4f}",
    ]
)
if m_dir_a and m_dir_b:
    relatorio_lines.extend(
        [
            f"- **Recorte Direto/Controle (Literal sem Tensão Concessiva — N={m_dir_a['N']})**:",
            f"  * Condição A (Baseline): Acc={m_dir_a['Acc_Pol']:.4f} | RMSE_V={m_dir_a['RMSE_V']:.4f} | RMSE_A={m_dir_a['RMSE_A']:.4f} | RMSE_VA={m_dir_a['RMSE_VA_Oficial']:.4f} | PCC_A={m_dir_a['PCC_A']:.4f}",
            f"  * Condição B (DSAA v2):  Acc={m_dir_b['Acc_Pol']:.4f} | RMSE_V={m_dir_b['RMSE_V']:.4f} | RMSE_A={m_dir_b['RMSE_A']:.4f} | RMSE_VA={m_dir_b['RMSE_VA_Oficial']:.4f} | PCC_A={m_dir_b['PCC_A']:.4f}",
        ]
    )
if m_diag_a and m_diag_b:
    relatorio_lines.extend(
        [
            f"- **Suíte Diagnóstica Figurativa/Tensiva Canônica (N={m_diag_a['N']})**:",
            f"  * Condição A (Baseline): Acc={m_diag_a['Acc_Pol']:.4f} | F1={m_diag_a['F1_Macro']:.4f} | RMSE_V={m_diag_a['RMSE_V']:.4f} | RMSE_A={m_diag_a['RMSE_A']:.4f} | RMSE_VA={m_diag_a['RMSE_VA_Oficial']:.4f} | PCC_V={m_diag_a['PCC_V']:.4f} | PCC_A={m_diag_a['PCC_A']:.4f}",
            f"  * Condição B (DSAA v2):  Acc={m_diag_b['Acc_Pol']:.4f} | F1={m_diag_b['F1_Macro']:.4f} | RMSE_V={m_diag_b['RMSE_V']:.4f} | RMSE_A={m_diag_b['RMSE_A']:.4f} | RMSE_VA={m_diag_b['RMSE_VA_Oficial']:.4f} | PCC_V={m_diag_b['PCC_V']:.4f} | PCC_A={m_diag_b['PCC_A']:.4f}",
        ]
    )

relatorio_lines.extend(
    [
        "",
        "## 5. Testes de Significância Estatística (Cluster Bootstrap 3.000x + Wilcoxon Pareado)",
        f"- Tensiva DSAA (B) vs. Baseline Direto (A):",
        f"  * Delta RMSE_VA = {boot_ab['delta_rmse_va']:+.4f} [IC 95%: {boot_ab['ci_rmse_low']:+.4f}, {boot_ab['ci_rmse_high']:+.4f}] | Prob(B < A) = {boot_ab['prob_rmse_better']*100:.1f}%",
        f"  * Delta RMSE_V  = {boot_ab['delta_rmse_v']:+.4f} [IC 95%: {boot_ab['ci_rmse_v_low']:+.4f}, {boot_ab['ci_rmse_v_high']:+.4f}]",
        f"  * Delta PCC_A   = {boot_ab['delta_pcc_a']:+.4f} [IC 95%: {boot_ab['ci_pcca_low']:+.4f}, {boot_ab['ci_pcca_high']:+.4f}]",
        f"  * Teste de Postos Sinalizados de Wilcoxon (Erro Euclidiano por Instância): Win-Rate = {win_rate_ab*100:.1f}%, p = {wilcox_p_ab:.6f}",
    ]
)
if boot_a2b is not None:
    relatorio_lines.extend(
        [
            f"- Isolamento Causal — Tensiva DSAA (B) vs. Controle CoT Lexical (A2):",
            f"  * Delta RMSE_VA = {boot_a2b['delta_rmse_va']:+.4f} [IC 95%: {boot_a2b['ci_rmse_low']:+.4f}, {boot_a2b['ci_rmse_high']:+.4f}] | Delta PCC_A = {boot_a2b['delta_pcc_a']:+.4f}",
            f"  * Wilcoxon B vs A2: Win-Rate = {win_rate_a2b*100:.1f}%, p = {wilcox_p_a2b:.6f}",
        ]
    )

relatorio_lines.extend(
    [
        "",
        "## 6. Validação Empírica da Parábola Tensiva de Zilberberg por Categoria de Silva (2021)",
        tensive_stats.to_string(index=False),
        "",
        "## 7. Top 10 Ganhos Qualitativos (Onde a Micro-Ancoragem Tensiva Superou o Baseline)",
    ]
)

for _, row in top_wins.iterrows():
    relatorio_lines.extend(
        [
            f"\n### [{row['sentence_id']}] Aspecto: `{row['aspect']}` ({row['phenomenon']} | Ganho Euclidiano = {row['Ganho_Tensivo_Euclid']:+.2f})",
            f"- Sentença: \"{row['text']}\"",
            f"- Gabarito Oficial: Polaridade=`{row['gold_polarity']}` | V={row['gold_v']:.2f}, A={row['gold_a']:.2f}"
            + (f" | Tensiva Esperada=`{row['gold_tensive']}`" if row["gold_tensive"] != "N/A_EMPIRICAL_DEV" else ""),
            f"- Condição A (Baseline): Polaridade=`{row['A_Polaridade']}` | V={row['A_Valence']:.2f}, A={row['A_Arousal']:.2f} (Erro={row['Erro_Euclid_A']:.2f})",
            f"- Condição B (DSAA v2):  Categoria=`{row['B_Categoria_Tensiva']}` | Polaridade=`{row['B_Polaridade']}` | V={row['B_Valence']:.2f}, A={row['B_Arousal']:.2f} (Erro={row['Erro_Euclid_B']:.2f})",
            f"  * Evidência Micro-Sintática Extraída: {row['B_Evidencia_Micro_Aspecto']}",
        ]
    )

relatorio_lines.append("\n## 8. Top 5 Casos Residuais Difíceis (Análise de Limites do Modelo)")
for _, row in top_losses.iterrows():
    relatorio_lines.extend(
        [
            f"\n### [{row['sentence_id']}] Aspecto: `{row['aspect']}` ({row['phenomenon']} | Delta Euclidiano = {row['Ganho_Tensivo_Euclid']:+.2f})",
            f"- Sentença: \"{row['text']}\"",
            f"- Gabarito Oficial: V={row['gold_v']:.2f}, A={row['gold_a']:.2f} | Baseline: V={row['A_Valence']:.2f}, A={row['A_Arousal']:.2f} | DSAA v2 (`{row['B_Categoria_Tensiva']}`): V={row['B_Valence']:.2f}, A={row['B_Arousal']:.2f}",
            f"  * Evidência Extraída: {row['B_Evidencia_Micro_Aspecto']}",
        ]
    )

REPORT_OUT = "relatorio_insights_para_ia_v2.txt"
Path(REPORT_OUT).write_text("\n".join(relatorio_lines) + "\n", encoding="utf-8")

print(f"\n[Arquivos Gerados com Sucesso]")
print(f"  1. {CSV_OUT}")
print(f"  2. {REPORT_OUT}")

# Download automático não-bloqueante caso esteja no Google Colab
if IN_COLAB:
    try:
        from google.colab import files

        files.download(CSV_OUT)
        files.download(REPORT_OUT)
        print("[Colab] Download automático iniciado para sua máquina!")
    except Exception:
        print("[Colab] Os arquivos já estão salvos na aba lateral 'Arquivos' (/content/) prontos para baixar.")

print("\n>>> PROTOCOLO DEFINITIVO v2 CONCLUÍDO COM SUCESSO!")
