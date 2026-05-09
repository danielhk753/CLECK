# 📔 Notebooks Directory

This folder contains the core experimental pipeline for the research on **Knowledge Conflict Resolution in LLMs**. The notebooks are organized sequentially, following the lifecycle of the dataset from generation to evaluation.

## 🚀 Pipeline Overview

### 1. [create_data.ipynb](https://www.google.com/search?q=./create_data.ipynb)

**Purpose:** The entry point for the dataset generation.

* **Key Tasks:** Handles the primary data synthesis pipeline, converting raw Wikipedia tables into conflicting Claim/Evidence pairs.
* **Outputs:** Generates the final dataset of 913 samples.
* **Analytics:** Contains the "Dataset Statistics" section used in Appendix C, including the double-nested donut plots for strategy distribution and parity histograms for lexical bias.

### 2. [dataset_annotation.ipynb]()

**Purpose:** Documents the human-in-the-loop validation of the generated data.

* **Metrics:** Calculation of pairwise agreement (IAAs) between the four human annotators.
* **Resolution:** Detailed breakdown of the expert-led disagreement resolution process to ensure high-fidelity ground truth labels.

### 3. [llm_judge_annotation.ipynb]()

**Purpose:** Validates the reliability of using LLMs as automated judges.

* **Function:** Tests the `CoTKnowledgeConflictJudge` logic (Gemini-3.1-Flash-Lite) for answer classification and "Conflict Acknowledgement" detection.
* **Validation:** Measures how well the model identifies when it is actively reasoning through a conflict versus simply reciting pre-trained knowledge.

### 4. [evaluation.ipynb]()

**Purpose:** The main benchmarking suite for the research.

* **Experiments:** Runs the evaluation loop across various models (e.g., GPT-5-mini, Gemini-3-flash, DeepSeek-V4) using the OpenRouter API.
* **Analysis:** Computes the primary metrics (CAR, CER, CON) to determine model performance in "knowledge conflict" scenarios.