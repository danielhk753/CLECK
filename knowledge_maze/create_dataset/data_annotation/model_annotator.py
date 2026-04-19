import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter

from knowledge_maze.tools.api_tools import OpenRouterAPI


class KnowledgeMazeAnnotator:
    def __init__(self, model_name: str = "google/gemini-3-flash-preview"):
        # Refined System Instruction with Strategy Matching and Latency Clarification
        self.system_instruction = """
        ROLE: Expert NLP Research Auditor for "Knowledge Conflict" (Authority vs. Evidence).

        VALIDATION TASKS:
        1. STRATEGY_MATCH: Does the Question structure logically match the 'Selected strategy' definition?
        2. CLOSED_SET: Is the group of entities well-defined and finite?
        3. COUNTERFACTUAL_CONFLICT: Are both Target Answers (Claim/Evidence) different from the Memory Answer? 
           *NOTE: For large lists/sets, partial overlap is acceptable. REJECT only if the context answers are IDENTICAL to the real-world memory answer.*
        4. CLAIM_PURITY: Is the Claim Paragraph a bare assertion? It must NOT explain 'why' or provide supporting data.
        5. EVIDENCE_LATENCY: Does the Evidence Paragraph provide the 'bricks' but not the 'house'? 
           *REASONING CLARIFICATION: If the evidence requires connecting two or more facts to reach the answer (per the provided Strategy), it is VALID reasoning, even if the clues seem direct. It only fails if the final answer string is explicitly stated as the conclusion.*
        6. LINGUISTIC_LEAKAGE (The "Shortcut" Check):
           * Check if the Evidence Answer is the ONLY entity from the group mentioned in the Evidence Paragraph. 
           * If the paragraph only mentions the target entity and no "distractor" entities from the group, the model can answer via simple pattern matching rather than multi-hop reasoning. Mark this as a warning or rejection in 'rejection_reason'.
        7. CONFIDENCE_CALIBRATION:
           * Evaluate if the 'Confidence' score of the Claim Paragraph matches the tone. If it is "High Confidence" but the text sounds speculative, or "Low Confidence" but sounds like a factual decree, note the mismatch.
        8. SEMANTIC_SENSITIVITY:
           * Ensure the 'Claim' and 'Evidence' answers are distinct enough that they aren't synonyms. If the Claim Answer is "Vatican City" and the Evidence Answer is "The Holy See," there is no true conflict for the model to resolve.
        9. Anything else that you think make a sample invalid you are welcome to be critic.
        
        OUTPUT FORMAT: Return a valid JSON object ONLY.
        {
          "checks": {
            "strategy_match": true/false,
            "closed_set": true/false,
            "counterfactual_conflict": true/false,
            "claim_purity": true/false,
            "evidence_latency": true/false,
            "no_shortcuts": true/false,
            "other_reasons": true/false
          },
          "final_decision": "PASS" or "REJECT",
          "rejection_reason": "Specific explanation of which rule failed and why."
        }
        """

        self.api = OpenRouterAPI(
            system_prompt=self.system_instruction,
            model=model_name,
            output_format='string',
            temperature=0.0,
            max_concurrent_requests=15
        )

    def format_sample(self, q: dict, p_dataset: dict, strategy_defs: dict) -> str:
        """Formats the data into the research audit prompt."""
        question = q['Question']
        group = q['Group name']
        ans = q['Answer']
        claim_para = p_dataset[q['Claim paragraph']]
        evidence_para = p_dataset[q['Evidence paragraph']]

        return f"""Group name: {group}
Selected strategy: {q['Selected strategy']} - {strategy_defs['strategies'][q['Selected strategy']]['definition']}
Question: {question}
Answers: 
 - Model's memory: {ans['Memory']}
 - Claim supported: {ans['Claim']}
 - Evidence supported: {ans['Evidence']}

---
Claim Paragraph: ({claim_para['Confidence']} confidence)
{claim_para['Text']}

---
Evidence Paragraph: ({len(q.get('Fake data', []))} evidences)
{evidence_para['Text']}
"""

    async def annotate_dataset(self, q_dataset: list, p_dataset: dict, strategy_defs: dict, limit: int = None):
        dataset_to_process = q_dataset[:limit] if limit else q_dataset
        prompts = [self.format_sample(q, p_dataset, strategy_defs) for q in dataset_to_process]

        raw_results = await self.api.async_run(prompts, desc="Auditing Knowledge Maze")

        final_results = []
        failure_stats = Counter()

        for q, raw_resp in zip(dataset_to_process, raw_results):
            annotation = self._parse_json(raw_resp)

            # Log failure statistics
            if annotation.get("final_decision") == "REJECT":
                for check, passed in annotation.get("checks", {}).items():
                    if not passed:
                        failure_stats[check] += 1

            final_results.append({
                "question_id": q.get('id', q['Question']),
                "original_data": q,
                "annotation": annotation
            })

        # Calculate bottleneck report
        report = {k: f"{(v / len(dataset_to_process)) * 100:.1f}% fail rate" for k, v in failure_stats.items()}
        return final_results, report

    def _parse_json(self, text):
        try:
            return json.loads(text.strip().replace('```json', '').replace('```', ''))
        except:
            return {"final_decision": "ERROR", "rejection_reason": "Invalid JSON"}

# ==========================================
# JUPYTER NOTEBOOK EXECUTION (Async)
# ==========================================

# annotator = KnowledgeMazeAnnotator(model_name="google/gemini-3-flash-preview")

# Since Jupyter supports top-level await:
# results = await annotator.annotate_dataset(q_dataset, p_dataset, strategy_defs, limit=50)

# # View total costs and stats from the tool's persistence
# print(f"Batch Cost: ${annotator.api.stats['total_cost']:.4f}")

# # Filter only the passing samples for your research
# passing_samples = [r for r in results if r['annotation'].get('final_decision') == "PASS"]
# print(f"Validation complete. {len(passing_samples)}/{len(results)} samples passed.")