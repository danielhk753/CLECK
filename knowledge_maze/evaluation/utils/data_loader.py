"""
KnowledgeMaze Prompt Factory Module

This module defines the template-level logic for constructing LLM prompts from
pre-shuffled, frozen context registries. It is designed to support controlled
experiments involving knowledge conflict resolution and linguistic uncertainty.

Key Functional Features:
------------------------
1.  Deterministic Reconstruction: Converts 'frozen' paragraph ID sequences back into
    full-text context blocks. This ensures that every model evaluated in the
    KnowledgeMaze study sees identical paragraph permutations for each iteration.

2.  Targeted Hedging: Supports an experimental toggle (`use_hedged_version`) that
    selectively replaces standard paragraph text with a "Hedged" version
    (containing uncertainty markers). This replacement is strictly isolated to
    'Target' paragraphs (Claim and Evidence) while leaving distractors untouched.

3.  Resilient Extraction: Implements robust text retrieval logic that handles
    missing keys or 'None' values in the paragraph database without crashing
    the batch inference pipeline.

4.  Multi-Iteration Support: Generates a list of 5 standardized prompts per
    sample, enabling the calculation of positional variance and model stability
    metrics as required for the thesis research.

"""

from typing import List, Dict, Any


class KnowledgeMazePromptFactory:
    def __init__(self, paragraph_db: Dict[str, Any]):
        """
        Args:
            paragraph_db: Dictionary mapping IDs to entries containing 'Text'
                         and potentially 'Hedged text'.
        """
        self.paragraphs = paragraph_db

    def _get_text(self, doc_id: str, use_hedged: bool = False) -> str:
        """
        Helper to safely extract text. Uses 'or' logic to handle keys
        that exist in the dict but have None as a value.
        """
        entry = self.paragraphs.get(doc_id, {})

        if not isinstance(entry, dict):
            return str(entry) if entry is not None else "[Missing ID]"

        text_key = 'Hedged text' if use_hedged else 'Text'

        # The 'or' chain handles both missing keys and None values
        content = entry.get(text_key) or entry.get('Text') or f"[Missing Content: {doc_id}]"

        return str(content)

    def create_prompts(self, sample: Dict, use_hedged_version: bool = False) -> List[str]:
        """
        Assembles a list of 5 prompt strings based on the frozen context shuffles.

        Args:
            sample: The preprocessed sample containing Claim_ID and Evidence_ID.
            use_hedged_version: If True, applies 'Hedged text' only to target paragraphs.
        """
        question = sample.get('Question', '')
        frozen_contexts = sample.get('Context', [])

        # Identify the target IDs to be modified if the flag is active
        target_ids = {sample.get('Claim_ID'), sample.get('Evidence_ID')}

        prompts = []

        for iteration in frozen_contexts:
            p_ids = iteration.get('paragraphs', [])

            text_items = []
            for p_id in p_ids:
                # Switch key only if the flag is active AND paragraph is a target
                is_hedged_target = use_hedged_version and (p_id in target_ids)
                text_items.append(self._get_text(p_id, use_hedged=is_hedged_target))

            # Reconstruct the context block
            context_block = "\n\n---\n\n".join(text_items)

            # Format the final prompt string
            prompt = (
                f"CONTEXT:\n{context_block}\n\n"
                f"QUESTION: {question}\n\n"
                "ANSWER:"
            )
            prompts.append(prompt)

        return prompts