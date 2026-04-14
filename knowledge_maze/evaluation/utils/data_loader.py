import random
from typing import List, Dict, Optional


class KnowledgeMazePromptFactory:
    def __init__(self, paragraph_db: Dict[str, Dict[str, str]]):
        """
        Args:
            paragraph_db: Dictionary mapping IDs to {'Text': '...', 'Confidence': '...'}
        """
        self.paragraphs = paragraph_db

    def _get_text(self, doc_id: str) -> str:
        """Helper to safely extract text from the database."""
        entry = self.paragraphs.get(doc_id, {})
        if isinstance(entry, dict):
            return entry.get('Text', "[Missing Context]")
        return str(entry)

    def create_prompt(self, sample: Dict) -> str:
        """
        Assembles a single prompt string with shuffled context.
        """
        question = sample.get('Question', '')

        # 1. Collect all constituent IDs
        evidence_id = sample.get('Evidence paragraph')
        claim_id = sample.get('Claim paragraph')
        distractor_ids = sample.get('Distractor_IDs', [])

        # 2. Gather text and wrap in IDs for the model to reference
        context_items = [f"{self._get_text(evidence_id)}",
                         f"{self._get_text(claim_id)}"]

        # Add Distractors
        for d_id in distractor_ids:
            context_items.append(f"{self._get_text(d_id)}")

        # 3. SHUFFLE: This is critical for scientific validity
        # It prevents the model from assuming the first paragraph is always the 'Evidence'
        random.shuffle(context_items)

        # 4. Final String Construction
        context_block = "\n\n---\n\n".join(context_items)

        prompt = (
            f"CONTEXT:\n{context_block}\n\n"
            f"QUESTION: {question}\n\n"
            "ANSWER:"
        )

        return prompt


class KnowledgeMazeDataLoader:
    def __init__(self, questions_path: str, paragraphs_path: str):
        import json
        with open(questions_path, 'r') as f:
            self.dataset = json.load(f)
        with open(paragraphs_path, 'r') as f:
            self.paragraphs = json.load(f)

        self.factory = KnowledgeMazePromptFactory(self.paragraphs)

    def get_eval_samples(self) -> List[Dict]:
        """
        Returns the dataset with an additional 'formatted_prompt' key for each row.
        """
        for sample in self.dataset:
            sample['formatted_prompt'] = self.factory.create_prompt(sample)
        return self.dataset