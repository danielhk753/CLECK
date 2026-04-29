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

import re
import copy
import random
from uuid import uuid4
from typing import List, Dict, Any
from collections import defaultdict
from IPython.display import display, HTML

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

    def balance_dataset_flexible(self, data: List[Dict[str, Any]], strategy_key: str = 'Selected strategy',
                                 ratio: float = 1.2, seed: int = 42) -> List[Dict[str, Any]]:
        """
        Balances the dataset with a 'soft' cap.
        Larger categories are limited to (min_size * ratio).
        """
        if not data:
            return []

        # 1. Group the dictionaries by the strategy key
        grouped_data = defaultdict(list)
        for row in data:
            key_val = row.get(strategy_key)
            if key_val is not None:
                grouped_data[key_val].append(row)

        # 2. Calculate the balancing limits
        group_sizes = {k: len(v) for k, v in grouped_data.items()}
        min_size = min(group_sizes.values())
        max_allowed = int(min_size * ratio)

        print(f"[*] Stage 6: Balancing categories for small dataset variation.")
        print(f"📊 Smallest category: {min_size}")
        print(f"⚖️ Soft-balancing: Categories capped at {max_allowed} (Ratio: {ratio}x)")

        # Isolated random number generator to match Pandas random_state=42
        rng = random.Random(seed)

        # 3. Sample from each group
        balanced_data = []
        for group in grouped_data.values():
            n = min(len(group), max_allowed)
            sampled_group = rng.sample(group, k=n)
            balanced_data.extend(sampled_group)

        # 4. Final shuffle
        rng.shuffle(balanced_data)

        print(f"Remained dataset size {len(balanced_data)} out of initial {len(data)} samples.")
        return balanced_data

    def process_and_subset_hedged_data(self, question_dataset, paragraphs_dict):
        """
        1. Updates the paragraph dictionary:
           - 'Text' becomes the cleaned, confident version.
           - 'Hedged text' keeps the words but removes the [[ ]].
        2. Returns a subset of questions that link to these hedged paragraphs.
        """
        updated_paragraphs = copy.deepcopy(paragraphs_dict)

        # Regex Patterns for your specific example:
        # 1. Matches everything inside and including the [[ ]]
        # Used to create the Confident version
        strip_all_pattern = r"\[\[.*?\]\]"

        # 2. Matches only the brackets [[ or ]]
        # Used to create the Hedged version (keeps the text inside)
        strip_brackets_only_pattern = r"\[\[|\]\]"

        hedged_ids = set()

        for p_id, p_data in updated_paragraphs.items():
            # Check if the paragraph is marked as 'Hedged'
            if p_data.get('Confidence') == 'Hedged':
                original_text = p_data.get('Text', "")

                # Create Confident Version: "[[ It is claimed ]] Exactly 6..." -> "Exactly 6..."
                conf_text = re.sub(strip_all_pattern, "", original_text)
                conf_text = re.sub(r'\s+', ' ', conf_text).strip()

                # Create Hedged Version: "[[ It is claimed ]] Exactly 6..." -> "It is claimed Exactly 6..."
                hedge_text = re.sub(strip_brackets_only_pattern, "", original_text)
                hedge_text = re.sub(r'\s+', ' ', hedge_text).strip()

                # Store both in the paragraph dictionary
                p_data['Text'] = conf_text
                p_data['Hedged text'] = hedge_text

                hedged_ids.add(p_id)
            else:
                # For 'General' or 'Confident' paragraphs, 'Hedged text' is just None
                p_data['Hedged text'] = None

        # Filter the question dataset to only include those referencing a hedged paragraph
        hedged_question_subset = [
            copy.deepcopy(q) for q in question_dataset
            if q.get('Claim paragraph') in hedged_ids or q.get('Evidence paragraph') in hedged_ids
        ]

        print(f"✨ Processed {len(hedged_ids)} hedged paragraphs.")
        print(f"📦 Created subset of {len(hedged_question_subset)} questions.")

        return hedged_question_subset, updated_paragraphs

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

        # --- Stage 5: Separate Hedged from non-hedged paragraphs ---
        print("[*] Stage 5: Separate Hedged from Non-Hedged Paragraphs...")
        hedged_dataset, updated_paragraphs = self.process_and_subset_hedged_data(self.distractor_retriever.data_with_distractors,
                                                                                 self.final_paragraph_dataset)
        self.final_paragraph_dataset = updated_paragraphs

        # --- Stage 6: Export Approved Data ---
        print(f"[*] Finalizing: Saving approved samples to {export_prefix}_*.json")
        save_json(f"{export_prefix}_questions_full", self.distractor_retriever.data_with_distractors)
        save_json(f"{export_prefix}_questions_hedged", hedged_dataset)
        save_json(f"{export_prefix}_paragraphs", self.final_paragraph_dataset)

    def create_small_dataset(self, ratio, export_prefix: str = "knowledge_maze"):
        save_json(f"{export_prefix}_questions_small",
                  self.balance_dataset_flexible(self.distractor_retriever.data_with_distractors, ratio=ratio))

    def display_filtered_samples(self, num_to_show: int = 5, is_wrong: bool = False):
        # 1. Select 5 random samples from each attribute
        if is_wrong:
            wrong_subset = random.sample(self.validator.wrong_answers, min(len(self.validator.wrong_answers), num_to_show))

        abstained_subset = random.sample(self.validator.abstained_answers,
                                         min(len(self.validator.abstained_answers), num_to_show))

        css = """
        <style>
            .audit-container { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; margin: 20px 0; }
            .sample-card { 
                display: flex; 
                background: #ffffff; 
                border-radius: 10px; 
                box-shadow: 0 4px 15px rgba(0,0,0,0.06); 
                margin-bottom: 25px; 
                overflow: hidden;
                border: 1px solid #eef2f7;
            }
            .status-sidebar { width: 10px; }
            .status-wrong { background: #e74c3c; }
            .status-abstained { background: #95a5a6; }

            .card-content { padding: 20px; flex: 1; }
            .header { display: flex; justify-content: space-between; margin-bottom: 15px; }

            .badge { 
                padding: 5px 12px; border-radius: 6px; font-size: 0.7em; 
                font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px;
            }
            .badge-wrong { background: #fdeaea; color: #e74c3c; }
            .badge-abstained { background: #f0f2f5; color: #7f8c8d; }

            .section-label { font-weight: 700; color: #34495e; font-size: 0.85em; margin-bottom: 6px; display: block; }

            /* Side-by-Side Container */
            .answer-comparison { display: flex; gap: 15px; margin: 15px 0; }
            .answer-column { flex: 1; }

            .text-box { 
                background: #f8f9fa; padding: 12px; border-radius: 6px; 
                font-size: 0.95em; line-height: 1.5; color: #2c3e50; border: 1px solid #edf2f7;
            }
            .intended-box { border-left: 4px solid #27ae60; background: #f0fff4; }
            .model-box-wrong { border-left: 4px solid #e74c3c; }
            .model-box-abstain { border-left: 4px solid #95a5a6; }

            .explanation-box { background: #fffcf0; border-left: 4px solid #f1c40f; font-style: italic; }
            .context-box { font-size: 0.85em; color: #636e72; max-height: 150px; overflow-y: auto; }
        </style>
        """

        def build_card(sample, sample_type):
            is_wrong = sample_type == "Wrong Answer"
            status_class = "status-wrong" if is_wrong else "status-abstained"
            badge_class = "badge-wrong" if is_wrong else "badge-abstained"
            model_border = "model-box-wrong" if is_wrong else "model-box-abstain"

            return f"""
            <div class="sample-card">
                <div class="status-sidebar {status_class}"></div>
                <div class="card-content">
                    <div class="header">
                        <span class="badge {badge_class}">{sample_type}</span>
                    </div>

                    <span class="section-label">Question</span>
                    <div class="text-box" style="font-weight: 600; background: white;">{sample.get('Question')}</div>

                    <div class="answer-comparison">
                        <div class="answer-column">
                            <span class="section-label">Intended Answer (Target)</span>
                            <div class="text-box intended-box">{sample.get('Intended Answer')}</div>
                        </div>
                        <div class="answer-column">
                            <span class="section-label">Model Output</span>
                            <div class="text-box {model_border}">{sample.get('Model Answer')}</div>
                        </div>
                    </div>

                    <span class="section-label">Reasoning / Explanation</span>
                    <div class="text-box explanation-box">{sample.get('Explanation')}</div>

                    <div style="margin-top: 15px;">
                        <span class="section-label">Context Reference</span>
                        <div class="text-box context-box">{sample.get('Context')}</div>
                    </div>
                </div>
            </div>
            """

        html_output = f"{css}<div class='audit-container'>"

        if is_wrong:
            for s in wrong_subset: html_output += build_card(s, "Wrong Answer")

        for s in abstained_subset: html_output += build_card(s, "Abstained")

        html_output += "</div>"
        display(HTML(html_output))
