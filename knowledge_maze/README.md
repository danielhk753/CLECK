# knowledge_maze/

Python package implementing the CLECK (CLaim–Evidence Conflict of Knowledge)
dataset construction and evaluation pipeline. This is a library consumed by the
notebooks in [notebooks/](../notebooks) — see [notebooks/README.md](../notebooks/README.md)
for the actual end-to-end execution order and narrative. This README is a map of
the code itself.

Most modules expect a `REPO_PATH` environment variable (`.env`, loaded via
`python-dotenv`) pointing at the repo root, and several call out to LLMs via
OpenRouter (`tools/api_tools.py`) or directly via Gemini.

## Layout

### `create_dataset/` — dataset construction pipeline

Driven by `notebooks/create_data.ipynb`. Stages run roughly top-to-bottom:

| Subpackage | Key classes/files | Role |
|---|---|---|
| `create_questions/` | `wikipedia_mining.py` (`WikiMiner`) | Discovers and de-duplicates Wikipedia "List of…" pages. |
| | `table_processing.py` (`AsyncWikiExtractor`, `TablePostProcessor`) | Extracts and cleans structured tables from the mined pages. |
| | `grouping_entities.py` (`TableGroupConstructor`, `GroupValidator`) | LLM-validates tables into coherent "entity groups". |
| | `question_generation.py` (`TableSanitizer`, `StrategySelector`, `QuestionGenerator`, `QuestionFilter`) | Selects a reasoning strategy per group and generates+dedupes natural-language questions. `question_strategies.json` defines the 11 strategies. |
| | `question_validator.py` (`ManualTagger`) | Jupyter widget UI for human auditing of generated questions. |
| `create_answers/` | `answer_orchestrator.py` (`AnswerGenerator`) | Builds counterfactual Memory/Claim/Evidence answer triples via the `strategies/` and `utils/` op libraries (aggregation, extremes, filters, relational, math/table/text ops). |
| | `model_filter.py` (`LLMFilter`) | LLM gatekeeper rejecting ill-formed or unanswerable QA pairs. |
| | `answer_qa_tagger.py` (`AnswerValidityTagger`) | Jupyter widget UI for human auditing of generated answers. |
| `create_context/` | `evidence_generator.py` (`EvidenceParagraph`), `claim_generator.py` (`ClaimParagraph`) | Synthesize the authoritative "Evidence" prose and the filler-padded "Claim" haystack paragraph for each sample. |
| | `verifier.py` (`LLMValidator`) | LLM-as-judge gate: discards samples not solvable from their own Evidence/Claim paragraph. |
| | `distractors.py` (`ParallelDistractorValidator`) | TF-IDF + LLM sweep to find and validate adversarial distractor paragraphs. |
| | `paragraph_pipeline.py` (`IncrementalContextPipeline`, `IncrementalDistractorPipeline`, `create_dataset_ids`, `update_dataset_with_annotations`) | Orchestrates the above into the incremental, checkpointed final build; writes/reads `data_files/{export_prefix}_questions.json` etc. (`export_prefix` defaults to `"cleck"`). |
| `data_annotation/` | `get_annotation_samples.py` (`AnnotationSplitter`) | Strategy-balanced sampling for human annotation batches. |
| | `human_validator.py` (`ParagraphAuditTagger`) | Jupyter HITL audit UI for generated paragraphs. |
| | `annotation_results.py` | Hierarchical merge of annotator/expert/expert-resolution audits into the final gold labels (see `annotated_data/`). |
| | `agreement_score.py` | Inter-annotator reliability (raw agreement, Gwet's AC1). |
| | `data_statistics.py` (`KnowledgeConflictDatasetStats`) | Final-dataset diagnostics/visualizations (strategy distribution, lexical parity, etc). |

`annotated_data/` and `intermediate_stages_annotation_files/` hold the actual
audit results and tagger-progress checkpoints consumed by the modules above.

### `evaluation/` — model benchmarking pipeline

Driven by `notebooks/evaluation.ipynb` (and `notebooks/llm_judge_annotation.ipynb`
for judge validation).

| Subpackage | Key classes/files | Role |
|---|---|---|
| `utils/` | `preprocess.py` (`KnowledgeMazeDatasetDistiller`) | Builds the frozen, pre-shuffled evaluation subsets in `results/preprocessed_data/`. |
| | `data_loader.py` (`KnowledgeMazePromptFactory`) | Reconstructs full-text prompts from frozen paragraph-ID shuffles; handles the hedged-paragraph swap. |
| | `prompt_templates.py` | `BASELINE_PROMPT` / `COT_PROMPT` system prompts. |
| `models/` | `choose_models.py` (`UnifiedValidationSweep`, `KnowledgeMazeRunner`) | Screens candidate models for baseline reliability before the main study. |
| | `model_inference.py` (`KnowledgeMazeIncrementalRunner`) | Runs the actual self-consistency evaluation loop against `models_config.yaml`. |
| | `judge_model.py` (`CoTKnowledgeConflictJudge`, `KnowledgeMazeIncrementalJudge`) | LLM-as-judge: classifies each response's answer source and dual-awareness. |
| `llm_judge_annotation/` | `judge_annotation.py` (`JudgeDatasetManager`, `JudgeAuditTagger`, `AnnotatorReport`) | Curates a stratified sample of judge decisions and runs the HITL audit of judge accuracy (`data_to_annotate/`, `annotation_results/`). |
| `metrics/tables/` | `main_tables.py`, `ablation_tables.py`, `exhaustive_recall.py` | Publication-ready AWARE/AWARE\*, PREF/PREF\*, and ablation delta tables. |
| `metrics/plots/` | `awareness_pivot.py`, `evidence_volume.py`, `model_agreement.py`, `strategy.py` | Figures: PREF(x\|aw), evidence-volume scaling, cross-model agreement, per-strategy breakdowns. |

### `tools/`

- `api_tools.py` — async `OpenRouterAPI` client with concurrency control and persistent cost/usage tracking.
- `utils.py` — `load_json` / `save_json` / `load_jsonl` helpers. `save_json(name, obj)` always writes to `data_files/{name}.json`.
