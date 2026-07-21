# data_files/

Checkpoint outputs from the dataset construction pipeline (`knowledge_maze/create_dataset/`,
driven by `notebooks/create_data.ipynb`). They let the pipeline resume from a mid-stage
point instead of re-running Wikipedia mining and LLM generation from scratch:

| File | Produced by |
|------|-------------|
| `valid_titles.json` | `wikipedia_mining.py` |
| `valid_groups.json` | `grouping_entities.py` |
| `question_dataset.json` | `question_generation.py` |
| `question_answer_dataset.json` | `answer_orchestrator.py` / `paragraph_pipeline.py` |

(`candidate_tables.jsonl`, the raw pre-filter table dump, is regenerated locally and
excluded via `.gitignore`.)

## Licensing

This directory is **not** covered by the repository's MIT license (see [`LICENSE`](../LICENSE)
at the repo root, which applies to the code only). Its content comes from two sources:

- **Wikipedia-derived data** — table rows, entity names, and other facts mined directly
  from Wikipedia "List of…" articles, licensed under
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), same as the dataset
  published on Hugging Face.
- **Model-generated content** — questions, entity groupings, and filtering decisions
  produced by Google's Gemini models (`gemini-3-flash-preview`, `gemini-3.1-pro-preview`)
  as part of the construction pipeline, and therefore subject to
  [Google's Gemini API terms of use](https://ai.google.dev/gemini-api/terms) in addition
  to the above.

Use of this data should comply with both sets of terms.
