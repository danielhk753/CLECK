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
import os
import copy
import random
from uuid import uuid4
from typing import List, Dict, Any
from IPython.display import display, HTML
from dotenv.main import load_dotenv

from knowledge_maze.tools.utils import save_json, load_json
from knowledge_maze.create_dataset.create_context.verifier import LLMValidator
from knowledge_maze.create_dataset.create_context.claim_generator import ClaimParagraph
from knowledge_maze.create_dataset.create_context.evidence_generator import EvidenceParagraph
from knowledge_maze.create_dataset.create_context.distractors import ParallelDistractorValidator

load_dotenv()
REPO_PATH = os.getenv('REPO_PATH')

def create_dataset_ids(dataset):
    for sample in dataset:
        sample['ID'] = str(uuid4())
    save_json("intermediate_files/question_answer_dataset", dataset)


class IncrementalContextPipeline:
    def __init__(self,
                 storage_dir: str = "data_files",
                 export_prefix: str = "knowledge_maze"):
        self.storage_dir = storage_dir
        self.export_prefix = export_prefix

        # Paths for persistent storage
        self.paths = {
            "full": os.path.join(REPO_PATH, f"{storage_dir}/{export_prefix}_questions"),
            "paragraphs": os.path.join(REPO_PATH, f"{storage_dir}/{export_prefix}_paragraphs"),
            "hedged": os.path.join(REPO_PATH, f"{storage_dir}/{export_prefix}_questions_hedged")
        }

        # Initialize Engines
        self.evidence_generator = EvidenceParagraph()
        self.claim_generator = ClaimParagraph()
        self.validator = LLMValidator()

        # In-memory datasets
        self.existing_questions = []
        self.existing_paragraphs = {}
        self.existing_hedged_questions = []
        self.processed_ids = set()

        # Load state
        self._load_state()

    def _load_state(self):
        """Loads all three datasets to ensure incremental updates."""
        if os.path.exists(f"{self.paths['full']}.json"):
            self.existing_questions = load_json(f"{self.paths['full']}.json")
            self.processed_ids = {q['ID'] for q in self.existing_questions}

        if os.path.exists(f"{self.paths['paragraphs']}.json"):
            self.existing_paragraphs = load_json(f"{self.paths['paragraphs']}.json")

        if os.path.exists(f"{self.paths['hedged']}.json"):
            self.existing_hedged_questions = load_json(f"{self.paths['hedged']}.json")

        print(f"📂 State: {len(self.processed_ids)} full dataset / {len(self.existing_hedged_questions)} hedged samples loaded.")

    async def run_incremental(self, new_input_samples: List[Dict[str, Any]]):
        deltas = [s for s in new_input_samples if s.get('ID') not in self.processed_ids]
        if not deltas: return

        # 1. Run Generation (Assuming your logic from previous steps)
        # --- Stage 1 & Gate 1: Evidence ---
        await self.evidence_generator.generate_evidence_paragraphs(deltas)
        valid_evidence_samples = await self.validator.validate_evidence_only(
            self.evidence_generator.question_dataset,
            self.evidence_generator.paragraph_dataset
        )

        # --- Stage 2 & Gate 2: Claims ---
        await self.claim_generator.generate_claim_paragraphs(valid_evidence_samples)
        newly_valid = await self.validator.validate_claim_only(
            self.claim_generator.question_dataset,
            self.claim_generator.paragraph_dataset
        )

        # 2. Process Hedged Text for NEW Paragraphs ONLY
        new_paragraphs = self.claim_generator.paragraph_dataset | self.evidence_generator.paragraph_dataset

        strip_all = r"\[\[.*?\]\]"
        strip_brackets = r"\[\[|\]\]"

        for p_id, p_data in new_paragraphs.items():
            if p_data.get('Confidence') == 'Hedged':
                raw_text = p_data.get('Text', "")

                # 1. Strip the bracketed parts
                clean_conf = re.sub(strip_all, "", raw_text)
                clean_hedge = re.sub(strip_brackets, "", raw_text)

                # 2. Fix whitespace: replace multiple spaces with one and strip edges
                p_data['Text'] = re.sub(r'\s+', ' ', clean_conf).strip()
                p_data['Hedged text'] = re.sub(r'\s+', ' ', clean_hedge).strip()

            else:
                p_data['Hedged text'] = None

        # 3. Identify and Create NEW Hedged Question subset
        new_hedged_subset = []
        for q in newly_valid:
            q['Human Validated'] = False
            # Check if this question refers to any of the new hedged paragraphs
            p_ids = [q.get('Claim paragraph'), q.get('Evidence paragraph')]
            if any(new_paragraphs.get(pid, {}).get('Confidence') == 'Hedged' for pid in p_ids):
                new_hedged_subset.append(copy.deepcopy(q))

        # 4. Merge into Master Lists
        self.existing_questions.extend(newly_valid)
        self.existing_paragraphs.update(new_paragraphs)
        self.existing_hedged_questions.extend(new_hedged_subset)

        # --- GARBAGE COLLECTION: Clean Paragraph Dataset ---

        # 1. Collect all paragraph IDs referenced in the current question set
        # This includes Claim, Evidence, and any Distractor paragraphs if already assigned
        referenced_p_ids = set()
        for q in self.existing_questions:
            if q.get('Claim paragraph'):
                referenced_p_ids.add(q['Claim paragraph'])
            if q.get('Evidence paragraph'):
                referenced_p_ids.add(q['Evidence paragraph'])
            # Also check distractors list if Stage 4 has been run previously
            for d_id in q.get('distractor_paragraphs', []):
                referenced_p_ids.add(d_id)

        # 2. Filter the existing_paragraphs dictionary to keep only referenced keys
        self.existing_paragraphs = {
            p_id: p_data for p_id, p_data in self.existing_paragraphs.items()
            if p_id in referenced_p_ids
        }

        # We save all three every time a batch finishes to prevent data loss
        save_json(self.paths["full"], self.existing_questions)
        save_json(self.paths["paragraphs"], self.existing_paragraphs)
        save_json(self.paths["hedged"], self.existing_hedged_questions)

        print(f"✅ Batch Saved: +{len(newly_valid)} Full Dataset, +{len(new_hedged_subset)} Hedged.")
        print(f"Total dataset size:")
        full_path = f"{self.paths['full']}.json"
        hedged_path = f"{self.paths['hedged']}.json"
        print(f" - Full: {len(load_json(full_path))} samples.")
        print(f" - Hedged: {len(load_json(hedged_path))} samples.")

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


class IncrementalDistractorPipeline:
    def __init__(self,
                 storage_dir: str = "data_files",
                 export_prefix: str = "knowledge_maze"):
        self.storage_dir = storage_dir
        self.export_prefix = export_prefix

        # Paths for persistent storage (Mirrors IncrementalContextPipeline)
        self.paths = {
            "full": os.path.join(REPO_PATH, f"{storage_dir}/{export_prefix}_questions"),
            "paragraphs": os.path.join(REPO_PATH, f"{storage_dir}/{export_prefix}_paragraphs"),
            "hedged": os.path.join(REPO_PATH, f"{storage_dir}/{export_prefix}_questions_hedged")
        }

        # In-memory datasets
        self.existing_questions = []
        self.existing_paragraphs = {}
        self.existing_hedged_questions = []

        # Load state
        self._load_state()

    def _load_state(self):
        """Loads datasets to ensure incremental updates."""
        if os.path.exists(f"{self.paths['full']}.json"):
            self.existing_questions = load_json(f"{self.paths['full']}.json")

        if os.path.exists(f"{self.paths['paragraphs']}.json"):
            self.existing_paragraphs = load_json(f"{self.paths['paragraphs']}.json")

        if os.path.exists(f"{self.paths['hedged']}.json"):
            self.existing_hedged_questions = load_json(f"{self.paths['hedged']}.json")

        print(
            f"📂 Distractor State: {len(self.existing_questions)} full / {len(self.existing_hedged_questions)} hedged samples loaded.")

    async def run_incremental(self, num_samples: int = None):
        """
        Runs distractor retrieval only on questions that lack them,
        merges the results, and synchronizes to the Hedged dataset.
        """
        if not self.existing_questions:
            print("⚠️ Full dataset is empty. Cannot generate distractors.")
            return

        print("\n[*] Incremental Distractor Retrieval...")

        # 1. Filter out questions that already have distractors
        questions_needing_distractors = [
            q for q in self.existing_questions
            if not q.get('Distractor_IDs')  # Checks for missing key or empty list
        ]

        if num_samples:
            if len(questions_needing_distractors) > num_samples:
                questions_needing_distractors = random.sample(questions_needing_distractors, num_samples)

        if not questions_needing_distractors:
            print("✅ All questions already have distractors. Skipping generation.")
        else:
            print(f"[*] Found {len(questions_needing_distractors)} questions missing distractors. Generating...")

            # 2. Run validator ONLY on the subset that needs them
            distractor_retriever = ParallelDistractorValidator(
                questions_needing_distractors,
                self.existing_paragraphs
            )
            await distractor_retriever.run_parallel_validation()

            # 3. Create a temporary lookup for the NEWLY generated distractors
            new_distractors_map = {
                q['ID']: q.get('Distractor_IDs', [])
                for q in distractor_retriever.data_with_distractors
            }

            # 4. Merge the new distractors back into the main existing_questions list
            for q in self.existing_questions:
                if q['ID'] in new_distractors_map:
                    q['Distractor_IDs'] = new_distractors_map[q['ID']]

        # 5. Create a master lookup map {Sample_ID: [Distractor_Para_IDs]} from the Full dataset
        # This ensures Hedged always mirrors the Full dataset
        master_distractor_lookup = {
            q['ID']: q.get('Distractor_IDs', [])
            for q in self.existing_questions
        }

        # 6. Synchronize to the Hedged dataset
        print("[*] Syncing distractors to Hedged dataset...")
        for q in self.existing_hedged_questions:
            q['Distractor_IDs'] = master_distractor_lookup.get(q['ID'], [])

        # 7. Save to disk
        # Note: We don't save paragraphs here because distractor generation doesn't alter them.
        save_json(self.paths["full"], self.existing_questions)
        save_json(self.paths["hedged"], self.existing_hedged_questions)

        print(f"✅ Distractor Sync Complete.")
        print(f" - Full: {len(self.existing_questions)} total samples.")
        print(f" - Hedged: {len(self.existing_hedged_questions)} total samples.")