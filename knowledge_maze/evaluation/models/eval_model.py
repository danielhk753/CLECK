import os
import random
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple

from knowledge_maze.tools.utils import load_json
from knowledge_maze.tools.api_tools import OpenRouterAPI
from knowledge_maze.evaluation.utils.data_loader import KnowledgeMazeDataLoader
from knowledge_maze.evaluation.models.judge_model import KnowledgeConflictJudge
from knowledge_maze.evaluation.utils.prompt_templates import DIRECT_PROMPT, CONFLICT_AWARE_PROMPT, COT_PROMPT

class KnowledgeMazeEvaluator:
    """
    Evaluates LLM epistemic priorities (Claim vs. Evidence vs. Memory)
    using the Knowledge Maze dataset, evaluated by an LLM-as-a-Judge.
    """

    def __init__(
            self,
            repo_path: str,
            model_name: str = "google/gemini-3-flash-preview",
            prompt_type: str = "direct",
            max_concurrent_requests: int = 15,
    ):
        self.repo_path = repo_path
        self.model_name = model_name
        self.max_concurrent_requests = max_concurrent_requests

        # Paths
        self.questions_path = os.path.join(self.repo_path, "data_files/knowledge_maze_questions_small.json")
        self.paragraphs_path = os.path.join(self.repo_path, "data_files/knowledge_maze_paragraphs.json")

        # Initialize Target API
        self.prompt_type = prompt_type
        if self.prompt_type == 'direct':
            prompt = DIRECT_PROMPT
        elif self.prompt_type == 'conflict':
            prompt = CONFLICT_AWARE_PROMPT
        elif self.prompt_type == 'cot':
            prompt = COT_PROMPT
        else:
            raise ValueError(f"Unknown prompt type: {self.prompt_type}")

        self.api = OpenRouterAPI(
            system_prompt=prompt,
            model=self.model_name,
            max_concurrent_requests=self.max_concurrent_requests
        )

        # Initialize the Evaluator Judge (Flash-Lite)
        self.judge = KnowledgeConflictJudge(
            max_concurrent_requests=self.max_concurrent_requests
        )

    def _prepare_data(self) -> Tuple[List[str], List[Dict], List[str], List[str], List[int], List[int]]:
        """Loads and processes the datasets into aligned lists for evaluation."""
        question_data = load_json(self.questions_path)
        paragraphs = load_json(self.paragraphs_path)

        loader = KnowledgeMazeDataLoader(question_data, paragraphs)
        eval_samples = loader.get_eval_samples()

        questions = [sample['Question'] for sample in eval_samples]
        prompts = [sample['formatted_prompt'] for sample in eval_samples]
        answers = [sample['Answer'] for sample in eval_samples]
        strategies = [sample['Selected strategy'] for sample in eval_samples]

        certainty = [paragraphs[sample['Claim paragraph']]['Confidence'] for sample in question_data]

        # Calculate number of evidences securely using .get() to prevent KeyErrors
        num_evidences = [
            len(sample.get('Fake data', [])) if sample.get('Selected strategy') != "COMPARATIVE_DIFFERENCE" else 2
            for sample in question_data
        ]

        return prompts, answers, certainty, strategies, num_evidences, questions

    async def run_evaluation(self) -> List[Dict[str, Any]]:
        """
        Executes the full evaluation pipeline: data loading, API inference, and LLM Judge evaluation.
        """
        print(f"Loading data from {self.repo_path}...")
        prompts, answers, certainty, strategies, num_evidences, questions = self._prepare_data()

        print(f"Running target inference with model: {self.model_name}...")
        model_ans = await self.api.async_run(prompts, desc="Generating Answers")

        # Normalize target answers
        model_ans_norm = [str(data) if data else 'None' for data in model_ans]

        print("Evaluating answers using KnowledgeConflictJudge...")
        # Pack the data for the Judge module
        judge_dataset = [
            {'answer': m, 'context': a, 'question': q}
            for m, a, q in zip(model_ans_norm, answers, questions)
        ]

        # Run the semantic batch evaluation
        judge_verdicts = await self.judge.evaluate_batch(judge_dataset, desc="Judging Outputs")

        print("Calculating reliance statistics...")
        statistics = []

        for verdict, c, s, l in zip(judge_verdicts, certainty, strategies, num_evidences):
            statistics.append({
                'Answer': verdict.title(),  # Formats 'memory' to 'Memory', 'no answer' to 'No Answer', etc.
                'Certainty': c,
                'Strategy': s,
                'Num_evidence': l
            })

        print("Evaluation complete.")
        return statistics


class ThinkingKnowledgeMazeEvaluator:
    """
    Evaluator specifically for Reasoning/Thinking models.
    Captures both the final answer and the internal reasoning chain.
    """

    DIRECT_SYSTEM_PROMPT = (
        "You are a QA expert. You will be provided with a context and a question. "
        "Analyze the context carefully and provide a direct answer based strictly "
        "on the information provided. Make sure the final answer is concise and as short as you can."
        "If the answer is a number, state the number and nothing else, if it is a name, than "
        "state the name and nothing else. If you cannot answer Just say 'No answer'."
    )

    def __init__(
            self,
            repo_path: str,
            model_name: str = "deepseek/deepseek-r1",  # or "openai/o1-preview"
            max_concurrent_requests: int = 10,
            similarity_threshold: float = 0.2
    ):
        self.repo_path = repo_path
        self.model_name = model_name
        self.max_concurrent_requests = max_concurrent_requests
        self.similarity_threshold = similarity_threshold

        # Paths
        self.questions_path = os.path.join(self.repo_path, "data_files/knowledge_maze_questions.json")
        self.paragraphs_path = os.path.join(self.repo_path, "data_files/knowledge_maze_paragraphs.json")

        # Initialize API with return_thinking enabled
        self.api = OpenRouterAPI(
            system_prompt=self.DIRECT_SYSTEM_PROMPT,
            model=self.model_name,
            output_format='string',  # We want the final answer as a string
            max_concurrent_requests=self.max_concurrent_requests,
            return_thinking=True  # Trigger the thinking-specific API logic
        )

    def _prepare_data(self) -> Tuple[List[str], List[Dict], List[str], List[str], List[int], List[int]]:
        """Identical to the original loader, ensuring alignment across lists."""
        question_data = random.sample(load_json(self.questions_path), 100)
        paragraphs = load_json(self.paragraphs_path)

        loader = KnowledgeMazeDataLoader(question_data, paragraphs)
        eval_samples = loader.get_eval_samples()

        prompts = [sample['formatted_prompt'] for sample in eval_samples]
        answers = [sample['Answer'] for sample in eval_samples]
        strategies = [sample['Selected strategy'] for sample in eval_samples]
        certainty = [paragraphs[sample['Claim paragraph']]['Confidence'] for sample in question_data]
        question = [sample['Question'] for sample in question_data]

        num_evidences = [
            len(sample.get('Fake data', [])) if sample.get('Selected strategy') != "COMPARATIVE_DIFFERENCE" else 2
            for sample in question_data
        ]

        return prompts, answers, certainty, strategies, num_evidences, question

    @staticmethod
    def get_most_similar_index(target: str, candidates: List[str], threshold: float) -> int:
        """Categorizes response into Memory (0), Claim (1), or Evidence (2)."""

        def similarity(a: str, b: str) -> float:
            return SequenceMatcher(None, str(a), str(b)).ratio()

        scores = [similarity(target, cand) for cand in candidates]
        max_score = max(scores)
        return scores.index(max_score) if max_score >= threshold else -1

    async def run_evaluation(self) -> List[Dict[str, Any]]:
        """
        Executes evaluation for thinking models.
        Extracts 'output' for scoring and saves 'thinking' for analysis.
        """
        print(f"Loading data for thinking model evaluation...")
        prompts, answers, certainty, strategies, num_evidences, questions = self._prepare_data()

        print(f"Requesting thinking traces from: {self.model_name}...")
        # API returns List[Dict] where each dict is {"output": str, "thinking": str}
        raw_responses = await self.api.async_run(prompts)

        print("Processing reasoning chains and calculating reliance...")
        statistics = []

        for resp, a, c, s, l, q in zip(raw_responses, answers, certainty, strategies, num_evidences, questions):
            # Extract fields from the thinking model response
            # Handle potential None/Failures from API
            model_output = resp.get('output', '') if resp else 'None'
            model_thinking = resp.get('thinking', '') if resp else ''

            candidates = [
                str(a.get('Memory', '')),
                str(a.get('Claim', '')),
                str(a.get('Evidence', ''))
            ]

            # Match the FINAL output against the expected answers
            win_idx = self.get_most_similar_index(model_output, candidates, self.similarity_threshold)

            winner = {0: 'Memory', 1: 'Claim', 2: 'Evidence'}.get(win_idx, 'Other')

            if model_thinking:

                statistics.append({
                    'Question': q,
                    'Model Answer': model_output,
                    'Answer Options': a,
                    'Answer verdict': winner,
                    'Certainty': c,
                    'Strategy': s,
                    'Num_evidence': l,
                    'Thinking': model_thinking,  # Critical for verifying epistemic logic
                })

        print(f"Evaluation of {self.model_name} complete.")
        return statistics


