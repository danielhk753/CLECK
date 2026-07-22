# results/

Raw experimental outputs for CLECK, provided so others can re-analyze model
behavior directly. The folder separates the **frozen inputs** used for every
run from the **raw outputs** of each experimental condition.

> Two output formats live here. The retrieval condition (`extract/`) stores
> bare answer lists; all QA conditions (`baseline/`, `cot/`, `hedged/`,
> `scaling/`) store free-text generations plus LLM-judge labels. See
> **Schemas** below before parsing.

## Layout

### `preprocessed_data/` — frozen evaluation inputs
The exact instances fed to the models, with self-consistency orderings already
fixed so results are reproducible. One file per condition:

| File | Feeds |
|------|-------|
| `full_study.json` | main QA evaluation (`baseline/`) |
| `cot_subset.json` | chain-of-thought ablation (`cot/`) |
| `hedged_subset.json` | claim-hedging ablation (`hedged/`) |
| `scaling_subset.json` | within-family scaling comparison (`scaling/`) |

### Output folders — one JSON per model (`<provider>_<model>.json`)

| Folder | Condition | Format |
|--------|-----------|--------|
| `baseline/` | Main QA setting, no intervention. The primary results. | QA |
| `extract/` | **Retrieval setting**: the model is asked to surface both candidate answers directly. Establishes whether it *can* recover each answer in isolation — the capability ceiling against which the spontaneous-awareness "Detection Gap" is measured. | Retrieval |
| `cot/` | Chain-of-thought prompting ablation. | QA |
| `hedged/` | Hedged-claim ablation (the claim is softened from a firm assertion). | QA |
| `scaling/` | Larger within-family model variants, for the scaling comparison. | QA |

## Shared instance fields

Every file (both formats) is a JSON object keyed by **instance UUID**, and each
instance shares these fields:

```json
{
  "<instance_uuid>": {
    "question": "<question posed to the model>",
    "strategy": "<one of the 11 reasoning strategies, e.g. FREQUENCY_MODE>",
    "gt_answers": {
      "Memory":   "<real-world answer; counterfactually made incorrect, so it
                    is NOT recoverable from parametric memory>",
      "Claim":    "<answer asserted by the claim paragraph>",
      "Evidence": "<answer derivable from the evidence paragraph>"
    },
    "metadata": { "claim_id": "<uuid>", "evidence_id": "<uuid>",
                  "is_hedged_run": false },
    "responses": [ ... ],          // shape differs by format, see below
    "is_fully_answered": true
  }
}
```

## Schemas

### Retrieval format (`extract/`)
A single pass; `raw_response` is an unparsed JSON string with an `answers` array:

```json
"responses": [
  { "iteration_id": 1, "raw_response": "{\"answers\": [\"Copesa\", \"Grupo Claro\"]}" }
]
```
The array length is the raw retrieval signal: **one** element = the model
surfaced a single answer; **two** = it surfaced both the claim and evidence
answers when prompted directly. Types are not normalized (numbers may be
strings or ints; `TOP_K_RANKING` nests a list per answer) — parse defensively.

### QA format (`baseline/`, `cot/`, `hedged/`, `scaling/`)
Five self-consistency generations per instance. Each carries the raw text and
the LLM-judge's labels:

```json
"responses": [
  {
    "iteration_id": 1,
    "raw_response": "Explanation: ... \nFinal answer: 1",
    "judge_results": {
      "classification": "claim",        // claim | evidence | memory | abstain | other
      "dual_awareness": true,           // did the model verbalize the conflict?
      "explanation": "<judge's rationale>"
    }
  }
]
```

- `classification` is the final-answer source, used for the **PREF** metrics.
- `dual_awareness` is the per-generation **AWARE** signal, aggregated to
  **AWARE\*** (stable awareness ≥4/5).
- **Important:** `dual_awareness` tracks whether the model *explicitly
  verbalized* the claim–evidence conflict — not whether it engaged the
  evidence. A generation can derive the evidence answer correctly yet be
  `dual_awareness: false` if it never names the conflict (see the lower-bound
  discussion in the paper/thesis). Treat it as a lower bound on awareness.

The normalization and metric computation that turn these raw files into the
`AWARE` / `PREF` / `PREF(x | aw)` figures live in
`notebooks/evaluation.ipynb`.

## License & attribution

The code that produced these files is MIT-licensed (repo root [`LICENSE`](../LICENSE)). The
`raw_response` outputs themselves come from third-party models and remain subject to each
provider's own terms:

* **GPT-5.4, GPT-5.4 Mini, o3** (OpenAI) — subject to
  [OpenAI's Terms of Use](https://openai.com/policies/row-terms-of-use/).
* **Gemini 3 Flash Preview** (Google) — subject to
  [Google's Gemini API terms of use](https://ai.google.dev/gemini-api/terms). Gemini 3 Flash
  Preview (`gemini-3.1-flash-lite-preview`, the judge variant) is also used as the **LLM judge**
  that produces the `judge_results` field on *every* record in `baseline/`, `cot/`, `hedged/`,
  and `scaling/` — so this term applies across all of those files, not only
  `google_gemini-3-flash-preview.json`.
* **Mistral Medium 3.1** (Mistral AI) — subject to
  [Mistral AI's Terms of Service](https://mistral.ai/en/terms).
* **Llama 4 Maverick** (Meta) — subject to the
  [Llama 4 Community License Agreement](https://www.llama.com/llama4/license/).
* **DeepSeek V4 Flash, DeepSeek V4 Pro** (DeepSeek) — subject to the
  [DeepSeek Model License](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-MODEL).
* **DeepSeek R1-0528** (DeepSeek) — released under the
  [MIT License](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/LICENSE), distinct from the
  DeepSeek Model License that covers the V-series models above.

If you redistribute or build on these outputs, check the linked terms for the specific
model(s) involved.