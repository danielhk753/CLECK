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

from typing import List, Dict
from IPython.display import display, HTML

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
            4. ZERO EXTERNAL KNOWLEDGE: The provided context is your entire universe. If the answer is not present, output "NOT_FOUND".
            5. NO WRAPPING: Do not use introductory phrases like "The answer is..." or quotes around the text unless the quotes are part of the original context.
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

        results_pairs = [str({"Question": s.get('question'),
                              "Answer A": true_ans,
                              "Answer B": model_ans}) for s, true_ans, model_ans in zip(samples, true_answers, results)]

        final_summary = await self.llm_judge.async_run(results_pairs, desc="Validating Model Answers")

        return final_summary

    async def review_data(self, question_data, paragraphs):
        self.total_samples = len(question_data)

        # ---------------------------------------------------------
        # PHASE 1: CHECK EVIDENCE FIRST (To save tokens/cost)
        # ---------------------------------------------------------
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
        passed_evidence_samples = []
        for s, evidence_ans in zip(question_data, evidence_results):
            if evidence_ans:
                passed_evidence_samples.append(s)
            else:
                self.failed_evidence.append(s)

        # ---------------------------------------------------------
        # PHASE 2: CHECK CLAIMS (Only on surviving samples)
        # ---------------------------------------------------------
        # Only run if there are samples that passed the evidence check
        if passed_evidence_samples:
            claim_input = [{'question': s.get('Question'),
                            'answer': s.get('Answer').get('Claim'),
                            'context': paragraphs[s.get('Claim paragraph')]} for s in passed_evidence_samples]

            self.api = OpenRouterAPI(
                system_prompt=self.extract_system_prompt,
                model=self.model,
                output_format='string',
                temperature=0.0
            )

            claim_results = await self.validate_samples(claim_input)

            # Filter final valid samples based on claim results
            for s, claim_ans in zip(passed_evidence_samples, claim_results):
                if claim_ans:
                    self.valid_data.append(s)
                else:
                    self.failed_claim.append(s)

        # ---------------------------------------------------------
        # PHASE 3: SAVE APPROVED PARAGRAPHS
        # ---------------------------------------------------------
        if self.valid_data:
            # Get all unique paragraph indices used by the approved questions
            valid_indices = set()
            for s in self.valid_data:
                valid_indices.add(s.get('Claim paragraph'))
                valid_indices.add(s.get('Evidence paragraph'))

            # Save only the relevant paragraphs (Using a dict preserves original index keys)
            self.approved_paragraphs = {idx: paragraphs[idx] for idx in valid_indices if idx is not None}

        self.display_summary()

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