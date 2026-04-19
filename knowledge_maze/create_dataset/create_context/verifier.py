"""
Data Verification and LLM Validation Module
------------------------------------------
This module provides a robust validation pipeline for Question-Answering datasets.
It utilizes a multi-phase verification process to ensure that generated questions
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
import torch
import unicodedata
from typing import List, Dict
from difflib import SequenceMatcher
from IPython.display import display, HTML
from sentence_transformers import SentenceTransformer, util

from knowledge_maze.tools.api_tools import OpenRouterAPI


class LLMValidator:
    """
    Uses OpenRouterAPI to validate questions against context and ground truth.
    """

    def __init__(self, model: str = "google/gemini-3-flash-preview"):
        # We set the output_format to 'bool' so the API returns True/False directly
        self.extract_system_prompt = (
            """
            You are a precise data extraction engine. Your sole purpose is to identify the specific segment of the provided context that answers the user's question.

            STRICT RULES:
            1. VERBATIM EXTRACTION: You must extract the answer as an exact quote from the context. Do not change a single character, tense, or word.
            2. MINIMALISM: Provide the shortest possible span of text that remains a complete answer. If the answer is "1994," do not return "It happened in 1994."
            3. NUMERICAL FIDELITY: Maintain the exact format of numbers and dates as they appear in the text (e.g., if the text says "22", do not write "twenty-two").
            4. ZERO EXTERNAL KNOWLEDGE: The provided context is your entire universe. Treat the context as absolute truth.
            5. NO WRAPPING: Do not use introductory phrases like "The answer is..." or quotes around the text unless the quotes are part of the original context.
            6. RAW VALUE: Output only the raw value (number in numerical form, full name, or comma-separated list) with no sentences, labels, or preamble.
            """
        )

        self.reasoning_system_prompt = (
            """
            You are a factual reasoning agent. Your task is to derive a concise answer by synthesizing information provided in the context.

            STRICT RULES:
            1. STEP-BY-STEP LOGIC: Analyze the evidence in the context to reach your conclusion. (If using a reasoning-enabled model, perform this in your thinking space).
            2. CONCISE OUTPUT: The final "Output" must be a focused answer - it can be a number, a name or a list of names if required. 
            3. FORMAT MATCHING: Ensure dates, currency, and numerical values match the style used in the context (e.g., "10.5M" vs "10,500,000").
            4. GROUND TRUTH ADHERENCE: Treat the context as absolute truth, even if it contradicts outside trivial facts - this is an alternate universe.
            5. NO CONVERSATION: Do not explain your answer in the final Output field. Only provide the result of the reasoning.
            6. RAW VALUE: Output only the raw value (number in numerical form, full name, or comma-separated list) with no sentences, labels, or preamble.
            """
        )

        self.llm_judge_prompt = (
            """
            You are a precise semantic judge. Given a question and two answers, determine if Answer A and Answer B convey the same factual meaning relative to the question.

            Rules:
            1. Ignore phrasing, casing, punctuation, and extra polite filler.
            2. Allow parenthetical synonyms (e.g., "NYC (New York City)" matches "NYC").
            3. STRICT: Numerical values must be identical (e.g., "10" is NOT the same as "11").
            4. STRICT: If Answer A is a list of multiple items, Answer B must contain all of them to be "True".
            5. Respond only with "True" or "False".
            """
        )

        self.model = model
        self.api = None
        self.llm_judge = OpenRouterAPI(
            system_prompt=self.llm_judge_prompt,
            model=self.model,
            output_format='bool',  # This triggers the BoolResponse Pydantic model
            temperature=0.0  # Minimal randomness for evaluation tasks
        )

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

        # results_pairs = [str({"Question": s.get('question'),
        #                       "Answer A": true_ans,
        #                       "Answer B": model_ans}) for s, true_ans, model_ans in zip(samples, true_answers, results)]

        # [print(f"- {true_ans}\n- {model_ans}\n --{self.are_equal(true_ans, model_ans)}--\n") for s, true_ans, model_ans in zip(samples, true_answers, results)]

        results_pairs = [(true_ans, model_ans) for s, true_ans, model_ans in zip(samples, true_answers, results)]
        final_summary = [self.are_equal(*pair) for pair in results_pairs]

        # final_summary = await self.llm_judge.async_run(results_pairs, desc="Validating Model Answers")

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

        self.api = OpenRouterAPI(
            system_prompt=self.reasoning_system_prompt,
            model=self.model,
            output_format='string',
            temperature=0.0
        )

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

        self.api = OpenRouterAPI(
            system_prompt=self.extract_system_prompt,
            model=self.model,
            output_format='string',
            temperature=0.0
        )

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
        embedding_a = self.validate_model.encode(a, convert_to_tensor=True)
        embedding_b = self.validate_model.encode(b, convert_to_tensor=True)
        return util.pytorch_cos_sim(embedding_a, embedding_b).item()

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

    # async def review_data(self, question_data, paragraphs):
    #     self.total_samples = len(question_data)
    #
    #     # ---------------------------------------------------------
    #     # PHASE 1: CHECK EVIDENCE FIRST (To save tokens/cost)
    #     # ---------------------------------------------------------
    #     evidence_input = [{'question': s.get('Question'),
    #                        'answer': s.get('Answer').get('Evidence'),
    #                        'context': paragraphs[s.get('Evidence paragraph')]} for s in question_data]
    #
    #     self.api = OpenRouterAPI(
    #         system_prompt=self.reasoning_system_prompt,
    #         model=self.model,
    #         output_format='string',
    #         temperature=0.0
    #     )
    #
    #     evidence_results = await self.validate_samples(evidence_input)
    #
    #     # Filter out failed evidence samples
    #     passed_evidence_samples = []
    #     for s, evidence_ans in zip(question_data, evidence_results):
    #         if evidence_ans:
    #             passed_evidence_samples.append(s)
    #         else:
    #             self.failed_evidence.append(s)
    #
    #     # ---------------------------------------------------------
    #     # PHASE 2: CHECK CLAIMS (Only on surviving samples)
    #     # ---------------------------------------------------------
    #     # Only run if there are samples that passed the evidence check
    #     if passed_evidence_samples:
    #         claim_input = [{'question': s.get('Question'),
    #                         'answer': s.get('Answer').get('Claim'),
    #                         'context': paragraphs[s.get('Claim paragraph')]} for s in passed_evidence_samples]
    #
    #         self.api = OpenRouterAPI(
    #             system_prompt=self.extract_system_prompt,
    #             model=self.model,
    #             output_format='string',
    #             temperature=0.0
    #         )
    #
    #         claim_results = await self.validate_samples(claim_input)
    #
    #         # Filter final valid samples based on claim results
    #         for s, claim_ans in zip(passed_evidence_samples, claim_results):
    #             if claim_ans:
    #                 self.valid_data.append(s)
    #             else:
    #                 self.failed_claim.append(s)
    #
    #     # ---------------------------------------------------------
    #     # PHASE 3: SAVE APPROVED PARAGRAPHS
    #     # ---------------------------------------------------------
    #     if self.valid_data:
    #         # Get all unique paragraph indices used by the approved questions
    #         valid_indices = set()
    #         for s in self.valid_data:
    #             valid_indices.add(s.get('Claim paragraph'))
    #             valid_indices.add(s.get('Evidence paragraph'))
    #
    #         # Save only the relevant paragraphs (Using a dict preserves original index keys)
    #         self.approved_paragraphs = {idx: paragraphs[idx] for idx in valid_indices if idx is not None}
    #
    #     self.display_summary()

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