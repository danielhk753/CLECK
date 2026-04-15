import os
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple

from knowledge_maze.tools.utils import load_json
from knowledge_maze.tools.api_tools import OpenRouterAPI
from knowledge_maze.evaluation.utils.data_loader import KnowledgeMazeDataLoader


class KnowledgeMazeEvaluator:
    """
    Evaluates LLM epistemic priorities (Claim vs. Evidence vs. Memory) 
    using the Knowledge Maze dataset.
    """

    GLOBAL_SYSTEM_PROMPT = (
        "You are a qa expert, you will get a context and you are required to "
        "answer the question based on the context. give only a direct answer "
        "and strictly be based on the context."
    )

    def __init__(
            self,
            repo_path: str,
            model_name: str = "google/gemini-3-flash-preview",
            max_concurrent_requests: int = 15,
            similarity_threshold: float = 0.2
    ):
        self.repo_path = repo_path
        self.model_name = model_name
        self.max_concurrent_requests = max_concurrent_requests
        self.similarity_threshold = similarity_threshold

        # Paths
        self.questions_path = os.path.join(self.repo_path, "data_files/knowledge_maze_questions.json")
        self.paragraphs_path = os.path.join(self.repo_path, "data_files/knowledge_maze_paragraphs.json")

        # Initialize API
        self.api = OpenRouterAPI(
            system_prompt=self.GLOBAL_SYSTEM_PROMPT,
            model=self.model_name,
            output_format="string",
            max_concurrent_requests=self.max_concurrent_requests
        )

    def _prepare_data(self) -> Tuple[List[str], List[Dict], List[str], List[str], List[int]]:
        """Loads and processes the datasets into aligned lists for evaluation."""
        question_data = load_json(self.questions_path)
        paragraphs = load_json(self.paragraphs_path)

        loader = KnowledgeMazeDataLoader(question_data, paragraphs)
        eval_samples = loader.get_eval_samples()

        prompts = [sample['formatted_prompt'] for sample in eval_samples]
        answers = [sample['Answer'] for sample in eval_samples]
        strategies = [sample['Selected strategy'] for sample in eval_samples]

        certainty = [paragraphs[sample['Claim paragraph']]['Confidence'] for sample in question_data]

        # Calculate number of evidences securely using .get() to prevent KeyErrors
        num_evidences = [
            len(sample.get('Fake data', [])) if sample.get('Selected strategy') != "COMPARATIVE_DIFFERENCE" else 2
            for sample in question_data
        ]

        return prompts, answers, certainty, strategies, num_evidences

    @staticmethod
    def get_most_similar_index(target: str, candidates: List[str], threshold: float) -> int:
        """
        Returns the index (0=Memory, 1=Claim, 2=Evidence) of the candidate string 
        most similar to the target string. 
        Returns -1 if no candidate meets the similarity threshold.
        """

        def similarity(a: str, b: str) -> float:
            return SequenceMatcher(None, a, b).ratio()

        scores = [similarity(target, cand) for cand in candidates]
        max_score = max(scores)

        # If the highest score is below our threshold, it's categorized as 'Other'
        if max_score < threshold:
            return -1

        return scores.index(max_score)

    async def run_evaluation(self) -> List[Dict[str, Any]]:
        """
        Executes the full evaluation pipeline: data loading, API inference, and statistical matching.
        """
        print(f"Loading data from {self.repo_path}...")
        prompts, answers, certainty, strategies, num_evidences = self._prepare_data()

        print(f"Running inference with model: {self.model_name}...")
        model_ans = await self.api.async_run(prompts)
        model_ans_norm = []
        for data in model_ans:
            if data:
                model_ans_norm.append(data)
            else:
                model_ans_norm.append('None')

        print("Calculating reliance statistics...")
        statistics = []

        for m, a, c, s, l in zip(model_ans_norm, answers, certainty, strategies, num_evidences):
            candidates = [str(a.get('Memory', '')), str(a.get('Claim', '')), str(a.get('Evidence', ''))]
            win_idx = self.get_most_similar_index(m, candidates, self.similarity_threshold)

            if win_idx == 0:
                winner = 'Memory'
            elif win_idx == 1:
                winner = 'Claim'
            elif win_idx == 2:
                winner = 'Evidence'
            else:
                winner = 'Other'

            statistics.append({
                'Answer': winner,
                'Certainty': c,
                'Strategy': s,
                'Num_evidence': l
            })

        print("Evaluation complete.")
        return statistics