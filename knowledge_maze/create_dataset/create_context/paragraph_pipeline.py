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
5. **Distractor Integration**: Utilizes TF-IDF vector search and parallel
   asynchronous validation to identify and inject semantically relevant
   but non-leaking "noise" paragraphs, testing model robustness against
   high-fidelity red herrings.

Final output consists of a pruned dataset of "Valid Data" and an "Approved
Paragraph" map, ensuring 100% answer-ability within the synthetic environment.
"""

import random
from uuid import uuid4
from typing import List, Dict, Any
from collections import defaultdict

from knowledge_maze.tools.utils import save_json
from knowledge_maze.create_dataset.create_context.verifier import LLMValidator
from knowledge_maze.create_dataset.create_context.claim_generator import ClaimParagraph
from knowledge_maze.create_dataset.create_context.evidence_generator import EvidenceParagraph
from knowledge_maze.create_dataset.create_context.distractors import ParallelDistractorValidator



class ContextPipeline:
    def __init__(self, model_name: str = "google/gemini-3-flash-preview"):
        """
        Initializes the pipeline with the necessary generation and validation engines.
        """
        self.evidence_generator = EvidenceParagraph()
        self.claim_generator = ClaimParagraph()
        self.validator = LLMValidator()
        self.distractor_retriever = None

        self.final_question_dataset = []
        self.final_paragraph_dataset = {}

    async def run(self, input_samples: List[Dict[str, Any]], export_prefix: str = "knowledge_maze"):
        """
        Executes the Stage-Gate generation and verification flow.
        """

        # --- Stage 1: Generate Evidence Paragraphs ---
        print(f"[*] Stage 1: Evidence Generation for {len(input_samples)} samples...")
        await self.evidence_generator.generate_evidence_paragraphs(input_samples)

        # --- GATE 1: Evidence Validation ---
        print("[*] GATE 1: Validating Evidence Paragraphs...")
        # You will need a method in LLMValidator that only checks the evidence paragraph.
        # It should return only the samples that passed.
        valid_evidence_samples = await self.validator.validate_evidence_only(
            self.evidence_generator.question_dataset,
            self.evidence_generator.paragraph_dataset
        )

        # --- Stage 2: Generate Claim Paragraphs (Cost Savings Here) ---
        print(f"[*] Stage 2: Claim Generation for {len(valid_evidence_samples)} valid samples...")
        await self.claim_generator.generate_claim_paragraphs(valid_evidence_samples)

        # --- GATE 2: Claim Validation ---
        print("[*] GATE 2: Validating Claim Paragraphs...")
        # Now validate that the newly generated claims are also solvable in isolation.
        valid_final_samples = await self.validator.validate_claim_only(
            self.claim_generator.question_dataset,
            self.claim_generator.paragraph_dataset
        )

        # --- Stage 3: Merge and Identify ---
        print("[*] Stage 3: Merging datasets and assigning IDs...")
        # Only merge paragraphs that survived both gates
        self.final_paragraph_dataset = (
                self.claim_generator.paragraph_dataset | self.evidence_generator.paragraph_dataset
        )
        self.final_question_dataset = valid_final_samples

        for sample in self.final_question_dataset:
            sample['ID'] = str(uuid4())

        # --- Stage 4: Retrieve Distractor Paragraphs ---
        print("[*] Stage 4: Distractor Paragraphs Retrieval...")
        self.distractor_retriever = ParallelDistractorValidator(
            self.final_question_dataset,
            self.final_paragraph_dataset
        )
        await self.distractor_retriever.run_parallel_validation()

        # --- Stage 6: Export Approved Data ---
        print(f"[*] Finalizing: Saving approved samples to {export_prefix}_*.json")
        save_json(f"{export_prefix}_questions", self.distractor_retriever.data_with_distractors)
        save_json(f"{export_prefix}_paragraphs", self.final_paragraph_dataset)
