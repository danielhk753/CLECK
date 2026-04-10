import numpy as np
from typing import List, Dict

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

        self.failed_claim = []
        self.failed_evidence = []

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

        final_summary = await  self.llm_judge.async_run(results_pairs, desc="Validating Model Answers")

        return final_summary

    async def review_data(self, question_data, paragraphs):

        claim_input = [{'question': s.get('Question'),
                        'answer': s.get('Answer').get('Claim'),
                        'context': paragraphs[s.get('Claim paragraph')]} for s in question_data]
        evidence_input = [{'question': s.get('Question'),
                           'answer': s.get('Answer').get('Evidence'),
                           'context': paragraphs[s.get('Evidence paragraph')]} for s in question_data]

        self.api = OpenRouterAPI(
            system_prompt=self.extract_system_prompt,
            model=self.model,
            output_format='string',  # This triggers the BoolResponse Pydantic model
            temperature=0.0  # Minimal randomness for evaluation tasks
        )
        claim_results = await self.validate_samples(claim_input)


        self.api = OpenRouterAPI(
            system_prompt=self.reasoning_system_prompt,
            model=self.model,
            output_format='string',  # This triggers the BoolResponse Pydantic model
            temperature=0.0  # Minimal randomness for evaluation tasks
        )
        evidence_results = await self.validate_samples(evidence_input)

        final_summary = []
        for s, claim_ans, evidence_ans in zip(question_data, claim_results, evidence_results):
            final_summary.append(claim_ans and evidence_ans)
            if not claim_ans:
                self.failed_claim.append(s)
            if not evidence_ans:
                self.failed_evidence.append(s)


        print(f"Pass rate: {100 * np.mean(final_summary):.1f}%")
