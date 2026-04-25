"""
Data Verification and LLM Validation Module
------------------------------------------
This module provides a robust validation pipeline for Question-Answering datasets.
It utilizes a multiphase verification process to ensure that generated questions
are answerable by an LLM using provided context paragraphs.

Key Features:
- **Cost-Optimized Pipeline**: Validates 'Evidence' paragraphs first. If a sample fails
  this phase, it is discarded before the 'Claim' validation, significantly reducing
  token consumption and API costs.
- **Strict Verification Logic**: Employs distinct specialized system prompts for
  verbatim extraction and factual reasoning.
- **Semantic Judgment**: Uses an LLM-based judge to compare model outputs against
  ground truth, accounting for synonyms while enforcing strict numerical accuracy.
- **Data Pruning**: Automatically filters and stores valid question-answer pairs
  and maintains a mapping of only the paragraphs required for successful samples.
- **Visual Analytics**: Includes a high-fidelity, dark-mode dashboard for real-time
  monitoring of pass rates and failure distributions.

Classes:
    LLMValidator: The primary engine for reviewing, validating, and auditing datasets.
"""
import re
import unicodedata
from typing import List, Dict
from difflib import SequenceMatcher
from IPython.display import display, HTML
from sentence_transformers import SentenceTransformer, util

from knowledge_maze.tools.api_tools import OpenRouterAPI
from knowledge_maze.evaluation.utils.prompt_templates import VERIFIER_PROMPT


class LLMValidator:
    """
    Uses OpenRouterAPI to validate questions against context and ground truth.
    """

    def __init__(self, model: str = "google/gemini-3-flash-preview", verbose: bool = True):
        # We set the output_format to 'bool' so the API returns True/False directly

        self.unified_system_prompt = VERIFIER_PROMPT

        self.model = model
        self.api = None
        self.api = OpenRouterAPI(
            system_prompt=self.unified_system_prompt,
            model=self.model,
            temperature=0.0
        )

        self.verbose = verbose

        self.total_samples = 0

        self.typo_threshold = 0.9
        self.iou_threshold = 1.0
        self.embedding_threshold = 0.8
        self.validate_model = SentenceTransformer('all-MiniLM-L6-v2')

        self.failed_claim = []
        self.failed_evidence = []
        self.valid_data = []  # Stores fully validated samples
        self.approved_paragraphs = {}  # Stores only the paragraphs used by valid_data

    @staticmethod
    def _prepare_prompt(sample: Dict[str, str]) -> str:
        """Formats the data sample into a clear evaluation prompt."""
        return (
            f"CONTEXT: {sample.get('context')}\n"
            f"QUESTION: {sample.get('question')}\n"
            f"ANSWER: "
        )

    async def validate_samples(self, samples: List[Dict[str, str]]) -> List[bool]:
        """
        Processes a list of samples and returns a list of booleans.
        """
        prompts = [self._prepare_prompt(s) for s in samples]
        true_answers = [s.get('answer') for s in samples]

        # async_run handles the batching, progress bar, and concurrency
        results = await self.api.async_run(prompts, desc="Validating Samples")

        results_pairs = [(true_ans, model_ans) for s, true_ans, model_ans in zip(samples, true_answers, results)]
        final_summary = [self.are_equal(*pair) for pair in results_pairs]

        return final_summary

    async def validate_evidence_only(self, question_data: List[Dict], paragraphs: Dict) -> List[Dict]:
        """
        GATE 1: Tests if the LLM can derive the answer from the Evidence paragraph alone.
        """
        self.total_samples = len(question_data)
        self.failed_evidence = []
        passed_evidence_samples = []

        if not question_data:
            return passed_evidence_samples

        evidence_input = [{'question': s.get('Question'),
                           'answer': s.get('Answer').get('Evidence'),
                           'context': paragraphs[s.get('Evidence paragraph')]} for s in question_data]

        evidence_results = await self.validate_samples(evidence_input)

        # Filter out failed evidence samples
        for s, evidence_ans in zip(question_data, evidence_results):
            if evidence_ans:
                passed_evidence_samples.append(s)
            else:
                self.failed_evidence.append(s)

        return passed_evidence_samples

    async def validate_claim_only(self, question_data: List[Dict], paragraphs: Dict) -> List[Dict]:
        """
        GATE 2: Tests if the LLM can extract the answer from the newly generated Claim paragraph.
        Runs final cleanup and generates the dashboard.
        """
        self.failed_claim = []
        self.valid_data = []

        if not question_data:
            self.display_summary()
            return self.valid_data

        claim_input = [{'question': s.get('Question'),
                        'answer': s.get('Answer').get('Claim'),
                        'context': paragraphs[s.get('Claim paragraph')]} for s in question_data]

        claim_results = await self.validate_samples(claim_input)

        # Filter final valid samples
        for s, claim_ans in zip(question_data, claim_results):
            if claim_ans:
                self.valid_data.append(s)
            else:
                self.failed_claim.append(s)

        # ---------------------------------------------------------
        # FINAL PHASE: SAVE APPROVED PARAGRAPHS
        # ---------------------------------------------------------
        if self.valid_data:
            # Get all unique paragraph indices used by the approved questions
            valid_indices = set()
            for s in self.valid_data:
                valid_indices.add(s.get('Claim paragraph'))
                valid_indices.add(s.get('Evidence paragraph'))

            # Save only the relevant paragraphs
            self.approved_paragraphs = {idx: paragraphs[idx] for idx in valid_indices if idx in paragraphs}

        if self.verbose:
            self.display_summary()
        return self.valid_data

    def clean(self, text: str):
        """Minimalist cleaning: lowercase, no accents, no list markers."""
        text = str(text).lower()
        text = "".join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
        # Remove common list symbols and markers
        text = re.sub(r'(?:\d+[\.\)]|\(\d+\)|[†\-\*])\s*', ' ', text)
        # Replace punctuation with spaces to avoid merging words
        text = re.sub(r'[,;."]', ' ', text)
        return " ".join(text.split())

    def get_set(self, text: str):
        """Splits into a set of unique words/terms."""
        cleaned = self.clean(text)
        return set(cleaned.split())

    def _is_numeric(self, s: str):
        """
        Returns a float only if the string is primarily a number.
        Prevents "1. Luanda" from being treated as the number 1.0.
        """
        # CRITICAL: Force cast to string to handle int/float inputs
        s = str(s)

        # 1. Clean the string of commas and extra whitespace
        cleaned = s.replace(',', '').strip()

        # 2. Match the number
        match = re.search(r"[-+]?\d*\.\d+|\d+", cleaned)
        if not match:
            return None

        # 3. Validation: Is there too much 'text' left?
        # We remove the matched number and common units/signs
        remaining_text = cleaned.replace(match.group(), '', 1)
        # Remove units like 'MW', '%', 'm', or 'kg' and punctuation
        remaining_text = re.sub(r'[a-zA-Z%$\. ]', '', remaining_text)

        # If there's still significant text left, it's a list or a name, not a number
        if len(remaining_text) > 0:
            return None

        return float(match.group())

    def calculate_iou(self, str_a: str, str_b: str):
        # --- NUMERICAL OVERRIDE ---
        # If both contain numbers, they MUST be identical.
        num_a, num_b = self._is_numeric(str_a), self._is_numeric(str_b)
        if num_a is not None or num_b is not None:
            return 1.0 if num_a == num_b else 0.0

        set_a = self.get_set(str_a)
        set_b = self.get_set(str_b)

        if not set_a and not set_b: return 1.0
        if not set_a or not set_b: return 0.0

        # --- FUZZY INTERSECTION ---
        intersection_count = 0
        temp_set_b = list(set_b)

        for item_a in set_a:
            best_score = 0
            best_match_idx = -1

            for i, item_b in enumerate(temp_set_b):
                # Quick exact check
                if item_a == item_b:
                    score = 1.0
                else:
                    score = SequenceMatcher(None, item_a, item_b).ratio()

                if score > best_score:
                    best_score = score
                    best_match_idx = i

            if best_score >= self.typo_threshold:
                intersection_count += 1
                if best_match_idx != -1:
                    temp_set_b.pop(best_match_idx)  # Don't double-count matches

        union_count = len(set_a) + len(set_b) - intersection_count
        return intersection_count / union_count

    def get_semantic_similarity(self, a: str, b: str):
        """Tier 3: Check if the meaning is the same even if words differ."""
        try:
            embedding_a = self.validate_model.encode(a, convert_to_tensor=True)
            embedding_b = self.validate_model.encode(b, convert_to_tensor=True)
            return util.pytorch_cos_sim(embedding_a, embedding_b).item()
        except:
            return 0.0

    def are_equal(self, a: str, b: str) -> bool:
        # TIER 1: Numerical Check
        num_a, num_b = self._is_numeric(a), self._is_numeric(b)
        if num_a is not None or num_b is not None:
            return num_a == num_b

        # TIER 2: IoU Check
        iou_score = self.calculate_iou(a, b)
        if iou_score >= self.iou_threshold:
            return True

        # TIER 3: Semantic Fallback
        # We only run this if IoU fails, saving compute.
        semantic_score = self.get_semantic_similarity(a, b)
        return semantic_score >= self.embedding_threshold

    def display_summary(self):
        """Displays a stylized dark-mode dashboard of the validation results."""
        passed = len(self.valid_data)
        fail_ev = len(self.failed_evidence)
        fail_cl = len(self.failed_claim)
        rate = (passed / self.total_samples * 100) if self.total_samples > 0 else 0

        # Color logic for the progress bar and percentage
        accent_color = "#4caf50" if rate >= 80 else "#ff9800" if rate >= 50 else "#f44336"

        dashboard_html = f"""
        <div style="background-color: #1e1e1e; color: #e0e0e0; padding: 20px; border-radius: 12px; 
                    border: 1px solid #333; font-family: 'Segoe UI', sans-serif; width: 400px; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.5); margin: 10px 0;">

            <h3 style="color: #bb86fc; margin-top: 0; margin-bottom: 5px;">
                📊 Approved Data Summary:
            </h3>
            
            <div style="color: #9aa0a6; font-size: 14px; margin-bottom: 15px;">
                Model Could Answer Both paragraphs
            </div>


            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="font-size: 0.9em; color: #aaa;">Overall Pass Rate</span>
                <span style="font-weight: bold; color: {accent_color};">{rate:.1f}%</span>
            </div>

            <div style="background-color: #333; border-radius: 10px; height: 8px; width: 100%; margin-bottom: 20px;">
                <div style="background-color: {accent_color}; height: 8px; width: {rate}%; border-radius: 10px; transition: width 0.5s;"></div>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px;">
                <div style="background: #252525; padding: 10px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 0.8em; color: #aaa;">Total Samples</div>
                    <div style="font-size: 1.2em; font-weight: bold; color: #03dac6;">{self.total_samples}</div>
                </div>
                <div style="background: #252525; padding: 10px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 0.8em; color: #aaa;">Passed</div>
                    <div style="font-size: 1.2em; font-weight: bold; color: #4caf50;">{passed}</div>
                </div>
            </div>

            <div style="font-size: 0.9em; color: #aaa; margin-bottom: 10px;">Failures Breakdown</div>
            <div style="display: flex; flex-direction: column; gap: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center; background: #2d2d2d; padding: 6px 12px; border-radius: 6px;">
                    <span style="font-size: 0.85em;">❌ Failed Evidence Paragraphs</span>
                    <span style="color: #f44336; font-weight: bold;">{fail_ev}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; background: #2d2d2d; padding: 6px 12px; border-radius: 6px;">
                    <span style="font-size: 0.85em;">❌ Failed Claim Paragraphs</span>
                    <span style="color: #ff9800; font-weight: bold;">{fail_cl}</span>
                </div>
            </div>
        </div>
        """
        display(HTML(dashboard_html))
