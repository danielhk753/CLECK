import json
import yaml
import copy
import random
from typing import List, Dict, Any, Optional

from knowledge_maze.tools.utils import load_json
from knowledge_maze.tools.api_tools import OpenRouterAPI
from knowledge_maze.evaluation.utils.data_loader import KnowledgeMazeDataLoader
from knowledge_maze.evaluation.utils.prompt_templates import DIRECT_PROMPT, CONFLICT_AWARE_PROMPT, COT_PROMPT


class KnowledgeMazeMultiInference:
    """
    Handles data loading and multimodal inference, supporting Self-Consistency
    through multiple context permutations. Evaluates a list of models against
    the exact same data shuffles for fair comparison.
    """

    def __init__(
            self,
            questions_path: str,
            paragraphs_path: str,
            config_path: str,
            prompt_type: str = "direct",
            num_samples: Optional[int] = None,
    ):
        self.questions_path = questions_path
        self.paragraphs_path = paragraphs_path
        self.config_path = config_path
        self.prompt_type = prompt_type
        self.num_samples = num_samples

        # 1. Load Configurations from YAML
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.models = config.get('models_to_evaluate', [])
        params = config.get('parameters', {})
        self.sc_iterations = params.get('sc_iterations', 1)
        self.temperature = params.get('temperature', 0.0)
        self.max_concurrent = params.get('max_concurrent', 15)
        self.max_rpm = params.get('max_rpm', 1000)

        # 2. Setup System Prompt
        self.is_cot = (self.prompt_type == 'cot')
        if self.prompt_type == 'direct':
            self.sys_prompt = DIRECT_PROMPT
        elif self.prompt_type == 'conflict':
            self.sys_prompt = CONFLICT_AWARE_PROMPT
        elif self.prompt_type == 'cot':
            self.sys_prompt = COT_PROMPT
        else:
            raise ValueError(f"Unknown prompt type: {self.prompt_type}")

        # 3. Load and Sample Dataset
        all_questions = load_json(self.questions_path)
        self.paragraphs = load_json(self.paragraphs_path)

        if self.num_samples is not None and self.num_samples < len(all_questions):
            print(f"Randomly sampling {self.num_samples} questions from the dataset...")
            self.question_data = random.sample(all_questions, self.num_samples)
        else:
            print(f"Using full dataset ({len(all_questions)} questions)...")
            self.question_data = all_questions

        # Generate the shared dataset once so all models evaluate the exact same shuffles
        self.base_dataset = self._build_shared_sc_dataset()

    def _build_shared_sc_dataset(self) -> List[Dict[str, Any]]:
        """
        Generates `sc_iterations` context permutations for the sampled questions.
        This is generated ONCE to ensure fair model comparison.
        """
        eval_batches = []
        for _ in range(self.sc_iterations):
            loader = KnowledgeMazeDataLoader(self.question_data, self.paragraphs)
            eval_batches.append(loader.get_eval_samples())

        structured_dataset = []

        for q_idx in range(len(self.question_data)):
            base_q = self.question_data[q_idx]
            certainty = self.paragraphs[base_q['Claim paragraph']]['Confidence']

            # Handle Num Evidences
            if base_q.get('Selected strategy') == "COMPARATIVE_DIFFERENCE":
                num_evidences = 2
            else:
                num_evidences = len(base_q.get('Fake data', []))

            sample_record = {
                "question_id": base_q.get('ID', q_idx),  # Fallback to index if ID is missing
                "question": base_q.get('Question', ''),
                "strategy": base_q.get('Selected strategy', 'UNKNOWN'),
                "certainty": certainty,
                "num_evidences": num_evidences,
                "true_answers": eval_batches[0][q_idx]['Answer'],
                "iterations": []
            }

            for i in range(self.sc_iterations):
                sample_record["iterations"].append({
                    "iteration_id": i + 1,
                    "prompt": eval_batches[i][q_idx]['formatted_prompt'],
                    "raw_response": None
                })

            structured_dataset.append(sample_record)

        return structured_dataset

    async def run_all_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Loops through all configured models, runs inference, and parses outputs.
        Returns a dictionary mapping model names to their completed datasets.
        """
        final_results = {}

        for model_name in self.models:
            print(f"\n[{model_name}] Initializing Inference Pipeline...")

            # Create a fresh copy of the dataset for this specific model
            model_dataset = copy.deepcopy(self.base_dataset)

            # Flatten prompts for the API call
            flat_prompts = []
            for sample in model_dataset:
                for iter_data in sample["iterations"]:
                    flat_prompts.append(iter_data["prompt"])

            # Initialize API for the current model
            api = OpenRouterAPI(
                system_prompt=self.sys_prompt,
                model=model_name,
                temperature=self.temperature,
                max_concurrent_requests=self.max_concurrent,
                max_rpm=self.max_rpm
            )

            print(f"[{model_name}] Processing {len(flat_prompts)} requests ({self.sc_iterations}x SC)...")
            raw_answers = await api.async_run(flat_prompts, desc=f"Generating {model_name.split('/')[-1]}")

            # Re-fold answers back into the nested structure
            ans_idx = 0
            for sample in model_dataset:
                for iter_data in sample["iterations"]:
                    response = raw_answers[ans_idx]

                    # Remove the prompt to keep memory/JSON output clean
                    iter_data.pop("prompt", None)

                    if self.is_cot and response:
                        try:
                            clean_resp = response.strip().strip('```json').strip('```').strip()
                            iter_data["raw_response"] = json.loads(clean_resp)
                        except json.JSONDecodeError:
                            iter_data["raw_response"] = {"explanation": "JSON Error", "answer": str(response)}
                    else:
                        iter_data["raw_response"] = str(response) if response else "None"

                    ans_idx += 1

            # Store in the master dictionary
            final_results[model_name] = model_dataset
            print(f"[{model_name}] Inference complete.")

        return final_results