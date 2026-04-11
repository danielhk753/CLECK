"""
Knowledge Maze Dataset Generation Pipeline
------------------------------------------
This module provides an end-to-end asynchronous pipeline for generating
synthetic, verified QA datasets. It orchestrates the transformation of
structured tabular data into complex narrative contexts.

The pipeline operates in four distinct stages:
1. **Evidence Synthesis**: Converts raw tabular data into authoritative,
   prose-based ground truth paragraphs using 'The Chronicler' personas.
2. **Claim Construction**: Generates abstract "haystack" paragraphs containing
   specific, varying-certain claims that point to the evidence.
3. **Identity & Merging**: Synchronizes the question-answer pairs with
   their newly generated contexts and assigns global unique identifiers (UUIDs).
4. **Adversarial Verification**: Runs a multiphase LLM-as-a-Judge loop to
   discard any samples where the context fails to support the answer or
   the claim is un-extractable.

Final output consists of a pruned dataset of "Valid Data" and an "Approved
Paragraph" map, ensuring 100% answer-ability within the synthetic environment.
"""

from uuid import uuid4
from typing import List, Dict, Any

from knowledge_maze.tools.utils import save_json
from knowledge_maze.create_dataset.create_context.verifier import LLMValidator
from knowledge_maze.create_dataset.create_context.claim_generator import ClaimParagraph
from knowledge_maze.create_dataset.create_context.evidence_generator import EvidenceParagraph


class ContextPipeline:
    def __init__(self, model_name: str = "google/gemini-3-flash-preview"):
        """
        Initializes the pipeline with the necessary generation and validation engines.
        """
        self.evidence_generator = EvidenceParagraph()
        self.claim_generator = ClaimParagraph()
        self.validator = LLMValidator(model=model_name)

        self.final_question_dataset = []
        self.final_paragraph_dataset = {}

    async def run(self, input_samples: List[Dict[str, Any]], export_prefix: str = "knowledge_maze"):
        """
        Executes the full generation and verification flow.

        Args:
            input_samples: List of dictionaries containing questions and tabular data.
            export_prefix: The prefix used for the saved JSON files.
        """

        # --- Stage 1: Generate Evidence Paragraphs ---
        # Transforms the 'Fake Data' tables into authoritative prose.
        print(f"[*] Starting Stage 1: Evidence Generation for {len(input_samples)} samples...")
        await self.evidence_generator.generate_evidence_paragraphs(input_samples)

        # --- Stage 2: Generate Claim Paragraphs ---
        # Houses the specific answers within abstract fillers.
        print("[*] Starting Stage 2: Claim Generation...")
        await self.claim_generator.generate_claim_paragraphs(self.evidence_generator.question_dataset)

        # --- Stage 3: Merge and Identify ---
        print("[*] Starting Stage 3: Merging datasets and assigning IDs...")
        self.final_paragraph_dataset = (
                self.claim_generator.paragraph_dataset | self.evidence_generator.paragraph_dataset
        )
        self.final_question_dataset = self.claim_generator.question_dataset

        for sample in self.final_question_dataset:
            sample['ID'] = str(uuid4())

        # --- Stage 4: Validation Loop ---
        # Filters out any samples that the LLM cannot consistently solve.
        print("[*] Starting Stage 4: Adversarial Validation (LLM-in-the-loop)...")
        await self.validator.review_data(self.final_question_dataset, self.final_paragraph_dataset)

        # --- Stage 5: Export Approved Data ---
        print(f"[*] Finalizing: Saving approved samples to {export_prefix}_*.json")
        save_json(f"{export_prefix}_questions", self.validator.valid_data)
        save_json(f"{export_prefix}_paragraphs", self.validator.approved_paragraphs)
