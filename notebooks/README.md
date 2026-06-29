# 📔 Notebooks Directory

This folder contains the core experimental pipeline for **CLECK** (CLaim–Evidence Conflict of Knowledge), a benchmark for studying how LLMs behave under *latent* claim–evidence conflict. The notebooks are organized sequentially, following the lifecycle of the dataset from generation through evaluation.

## 🚀 Pipeline Overview

### 1. `create_data.ipynb`

**Purpose:** Entry point for dataset generation.

* **Key Tasks:** Runs the construction pipeline that converts raw Wikipedia "List of…" tables into conflicting claim/evidence instances — table extraction and filtering, strategy-conditioned question generation, counterfactual answer construction, and paragraph synthesis.
* **Outputs:** Produces the final CLECK dataset of **5,922 instances** spanning eleven reasoning strategies, along with the **2,669-instance balanced evaluation subset** used for the main experiments.
* **Analytics:** Contains the dataset-statistics analyses — strategy-distribution plots (full vs. balanced subset) and the lexical/semantic parity histograms (cosine, ROUGE-L, BM25) that verify query-relevance balance between the claim and evidence paragraphs.

### 2. `dataset_annotation.ipynb`

**Purpose:** Human validation of the generated dataset.

* **Metrics:** Computes pairwise inter-annotator agreement (raw agreement and Gwet's AC1) across the four annotators over the 200-sample overlap set.
* **Outcome:** Reports the construction-pipeline success rate and the resulting gold-standard subset of **477** manually verified conflict instances, with per-strategy validation rates.

### 3. `llm_judge_annotation.ipynb`

**Purpose:** Validates the reliability of the LLM-as-a-judge used for response classification.

* **Function:** Tests the judge (Gemini 3.1 Flash Lite) on its two tasks — classifying the model's final answer (claim / evidence / memory / abstain / other) and detecting **dual-target conflict awareness**.
* **Validation:** Audits the judge against 400 human-annotated samples, reporting per-class accuracy and the awareness confusion matrix.

### 4. `evaluation.ipynb`

**Purpose:** Main benchmarking suite.

* **Experiments:** Runs the evaluation loop across the five instruction-tuned models (Gemini 3 Flash, Mistral 3.1 Medium, Llama 4 Maverick, GPT-5.4 Mini, DeepSeek V4 Flash) via the OpenRouter API, plus the CoT, hedging, and scaling ablations.
* **Metrics:** Computes the core conflict-behavior metrics — **AWARE / AWARE\*** (conflict awareness) and **PREF / PREF\*** and **PREF(x | aw)** (answer preference, including conditional-on-awareness) — that drive the analysis.