# import json
# import yaml
# import copy
# import random
# from typing import List, Dict, Any, Optional
#
# from knowledge_maze.tools.api_tools import OpenRouterAPI
# from knowledge_maze.evaluation.utils.data_loader import KnowledgeMazeDataLoader
# from knowledge_maze.evaluation.utils.prompt_templates import BASELINE_PROMPT, COT_PROMPT
#
#
# class KnowledgeMazeMultiInference:
#     """
#     Handles data loading and multimodal inference, supporting Self-Consistency
#     through multiple context permutations. Evaluates a list of models against
#     the exact same data shuffles for fair comparison.
#     """
#
#     def __init__(
#             self,
#             questions_data: list[dict],
#             paragraphs_data: dict,
#             config_path: str,
#             prompt_type: str = "direct",
#             num_samples: Optional[int] = None,
#     ):
#         self.questions = questions_data
#         self.paragraphs = paragraphs_data
#         self.config_path = config_path
#         self.prompt_type = prompt_type
#         self.num_samples = num_samples
#
#         # 1. Load Configurations from YAML
#         with open(self.config_path, 'r') as f:
#             config = yaml.safe_load(f)
#
#         self.models = config.get('models_to_evaluate', [])
#         params = config.get('parameters', {})
#         self.sc_iterations = params.get('sc_iterations', 1)
#         self.temperature = params.get('temperature', 0.0)
#         self.max_concurrent = params.get('max_concurrent', 15)
#         self.max_rpm = params.get('max_rpm', 1000)
#
#         # 2. Setup System Prompt
#         self.is_cot = (self.prompt_type == 'cot' or self.prompt_type == 'evidence_cot')
#         if self.prompt_type == 'baseline':
#             self.sys_prompt = BASELINE_PROMPT
#         elif self.prompt_type == 'cot':
#             self.sys_prompt = COT_PROMPT
#         # elif self.prompt_type == 'evidence_cot':
#         #     self.sys_prompt = RECONSTRUCTION_PROMPT
#         else:
#             raise ValueError(f"Unknown prompt type: {self.prompt_type}")
#
#         # 3. Load and Sample Dataset
#         all_questions = self.questions
#
#         if self.num_samples is not None and self.num_samples < len(all_questions):
#             print(f"Randomly sampling {self.num_samples} questions from the dataset...")
#             self.question_data = random.sample(all_questions, self.num_samples)
#         else:
#             print(f"Using full dataset ({len(all_questions)} questions)...")
#             self.question_data = all_questions
#
#         # Generate the shared dataset once so all models evaluate the exact same shuffles
#         self.base_dataset = self._build_shared_sc_dataset()
#
#     def _build_shared_sc_dataset(self) -> List[Dict[str, Any]]:
#         """
#         Generates `sc_iterations` context permutations for the sampled questions.
#         This is generated ONCE to ensure fair model comparison.
#         """
#         eval_batches = []
#         for _ in range(self.sc_iterations):
#             loader = KnowledgeMazeDataLoader(self.question_data, self.paragraphs)
#             eval_batches.append(loader.get_eval_samples())
#
#         structured_dataset = []
#
#         for q_idx in range(len(self.question_data)):
#             base_q = self.question_data[q_idx]
#             certainty = self.paragraphs[base_q['Claim paragraph']]['Confidence']
#
#             # Handle Num Evidences
#             if base_q.get('Selected strategy') == "COMPARATIVE_DIFFERENCE":
#                 num_evidences = 2
#             else:
#                 num_evidences = len(base_q.get('Fake data', []))
#
#             sample_record = {
#                 "question_id": base_q.get('ID', q_idx),  # Fallback to index if ID is missing
#                 "question": base_q.get('Question', ''),
#                 "strategy": base_q.get('Selected strategy', 'UNKNOWN'),
#                 "certainty": certainty,
#                 "num_evidences": num_evidences,
#                 "true_answers": eval_batches[0][q_idx]['Answer'],
#                 "iterations": []
#             }
#
#             for i in range(self.sc_iterations):
#                 sample_record["iterations"].append({
#                     "iteration_id": i + 1,
#                     "prompt": eval_batches[i][q_idx]['formatted_prompt'],
#                     "raw_response": None
#                 })
#
#             structured_dataset.append(sample_record)
#
#         return structured_dataset
#
#     async def run_all_models(self) -> Dict[str, List[Dict[str, Any]]]:
#         """
#         Loops through all configured models, runs inference, and parses outputs.
#         Returns a dictionary mapping model names to their completed datasets.
#         """
#         final_results = {}
#
#         for model_name in self.models:
#             print(f"\n[{model_name}] Initializing Inference Pipeline...")
#
#             # Create a fresh copy of the dataset for this specific model
#             model_dataset = copy.deepcopy(self.base_dataset)
#
#             # Flatten prompts for the API call
#             flat_prompts = []
#             for sample in model_dataset:
#                 for iter_data in sample["iterations"]:
#                     flat_prompts.append(iter_data["prompt"])
#
#             # Initialize API for the current model
#             api = OpenRouterAPI(
#                 system_prompt=self.sys_prompt,
#                 model=model_name,
#                 temperature=self.temperature,
#                 max_concurrent_requests=self.max_concurrent,
#                 max_rpm=self.max_rpm
#             )
#
#             print(f"[{model_name}] Processing {len(flat_prompts)} requests ({self.sc_iterations}x SC)...")
#             raw_answers = await api.async_run(flat_prompts, desc=f"Generating {model_name.split('/')[-1]}")
#
#             # Re-fold answers back into the nested structure
#             ans_idx = 0
#             for sample in model_dataset:
#                 for iter_data in sample["iterations"]:
#                     response = raw_answers[ans_idx]
#
#                     # Remove the prompt to keep memory/JSON output clean
#                     iter_data.pop("prompt", None)
#
#                     if self.is_cot and response:
#                         try:
#                             clean_resp = response.strip().strip('```json').strip('```').strip()
#                             iter_data["raw_response"] = json.loads(clean_resp)
#                         except json.JSONDecodeError:
#                             iter_data["raw_response"] = {"explanation": "JSON Error", "answer": str(response)}
#                     else:
#                         iter_data["raw_response"] = str(response) if response else "None"
#
#                     ans_idx += 1
#
#             # Store in the master dictionary
#             final_results[model_name] = model_dataset
#             print(f"[{model_name}] Inference complete.")
#
#         return final_results

import os
import json
import yaml
import random
from dotenv.main import load_dotenv
from typing import List, Dict, Any, Optional

from knowledge_maze.tools.api_tools import OpenRouterAPI
from knowledge_maze.evaluation.utils.data_loader import KnowledgeMazePromptFactory
from knowledge_maze.evaluation.utils.prompt_templates import BASELINE_PROMPT, COT_PROMPT, EXTRACT_PROMPT

load_dotenv()
REPO_PATH = os.getenv('REPO_PATH')
DATASET_DIR = os.getenv('DATASET_DIR')


class KnowledgeMazeIncrementalRunner:
    def __init__(self, config_path: str, experiment_type: str):
        """
        Args:
            config_path: Path to the YAML configuration file.
            experiment_type: One of 'baseline', 'cot', 'scaling', 'hedged', or 'extract'.
        """
        self.experiment_type = experiment_type.lower()
        repo_path = REPO_PATH
        dataset_dir = DATASET_DIR

        # 1. Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 2. Extract configuration specific to this experiment tier
        self.models = self.config.get('models', {}).get(self.experiment_type, [])
        self.params = self.config.get('parameters', {})
        self.use_hedged_text = (self.experiment_type == "hedged")

        # 3. Dynamic Gating rules for Prompts and Iterations
        if self.experiment_type == "cot":
            self.sys_prompt = COT_PROMPT
            self.num_iterations = 5
        elif self.experiment_type == "extract":
            self.sys_prompt = EXTRACT_PROMPT
            self.num_iterations = 1
        else:
            self.sys_prompt = BASELINE_PROMPT
            self.num_iterations = 5  # Default for baseline, hedged, scaling

        # 4. Handle input data loading dynamically relative to REPO_PATH
        dataset_rel_path = self.config.get('datasets', {}).get(self.experiment_type)
        if not dataset_rel_path:
            raise KeyError(f"Dataset path for '{self.experiment_type}' missing from config.")

        paragraph_rel_path = self.config.get('paragraph_path', os.path.join(DATASET_DIR, "cleck_paragraphs.json"))

        with open(os.path.join(repo_path, dataset_rel_path), 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)

        with open(os.path.join(dataset_dir, paragraph_rel_path), 'r', encoding='utf-8') as f:
            self.paragraphs = json.load(f)

        # 5. Initialize prompt factory
        self.factory = KnowledgeMazePromptFactory(self.paragraphs)

        # 6. Set up target results folder location
        results_rel_path = self.config.get('results_folder', {}).get(self.experiment_type)
        self.base_output_dir = os.path.join(repo_path, results_rel_path)
        os.makedirs(self.base_output_dir, exist_ok=True)

    def _get_save_path(self, model_name: str) -> str:
        safe_name = model_name.replace("/", "_")
        return os.path.join(self.base_output_dir, f"{safe_name}.json")

    def _load_model_state(self, model_name: str) -> Dict[str, Any]:
        path = self._get_save_path(model_name)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_model_state(self, model_name: str, state: Dict[str, Any]):
        path = self._get_save_path(model_name)
        temp_path = path + ".tmp"
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=4)
        os.replace(temp_path, path)

    def get_pending_samples(self, model_name: str, limit: Optional[int] = None) -> List[Dict]:
        """Filters out samples that already have the required iterations completed."""
        current_state = self._load_model_state(model_name)
        pending_ids = []

        for s_id, sample in self.dataset.items():
            is_done = current_state.get(str(s_id), {}).get('is_fully_answered', False)
            if not is_done:
                pending_ids.append((s_id, sample))

        if limit and limit < len(pending_ids):
            pending_ids = random.sample(pending_ids, limit)

        return pending_ids

    async def run_incremental(self, sample_limit: Optional[int] = None):
        """Orchestrates batch inference across models."""
        for model_name in self.models:
            print(f"\n>>> Model: {model_name} | Mode: {self.experiment_type.upper()}")

            to_process = self.get_pending_samples(model_name, limit=sample_limit)
            if not to_process:
                print(f"--- All samples processed. Skipping.")
                continue

            print(f"--- Found {len(to_process)} pending samples. Generating frozen prompts...")

            # 1. Initialize API for this model
            api = OpenRouterAPI(
                system_prompt=self.sys_prompt,
                model=model_name,
                temperature=self.params.get('temperature', 0.0),
                max_concurrent_requests=self.params.get('max_concurrent', 15)
            )

            # 2. Preparation: Flatten the required number of iterations for the API
            flattened_prompts = []
            for s_id, sample in to_process:
                # Use factory to get the iteration strings, slice to the needed amount
                prompts = self.factory.create_prompts(sample, use_hedged_version=self.use_hedged_text)
                flattened_prompts.extend(prompts[:self.num_iterations])

            # 3. Batch Call
            raw_results = await api.async_run(flattened_prompts, desc=f"Inference: {model_name}")

            # 4. State Update and Save
            current_state = self._load_model_state(model_name)

            for i, (s_id, sample) in enumerate(to_process):
                # Slice the batch of responses back out dynamically
                start_idx = i * self.num_iterations
                batch_responses = raw_results[start_idx: start_idx + self.num_iterations]

                # Map responses to iteration IDs for traceability
                iteration_results = []
                for idx, res in enumerate(batch_responses):
                    iteration_results.append({
                        "iteration_id": idx + 1,
                        "raw_response": str(res) if res else None
                    })

                current_state[str(s_id)] = {
                    "question": sample.get('Question'),
                    "strategy": sample.get('Selected strategy'),
                    "gt_answers": sample.get('Answer'),
                    "metadata": {
                        "claim_id": sample.get('Claim_ID'),
                        "evidence_id": sample.get('Evidence_ID'),
                        "is_hedged_run": self.use_hedged_text
                    },
                    "responses": iteration_results,
                    "is_fully_answered": len([r for r in batch_responses if r]) == self.num_iterations
                }

                # Save every 10 samples to checkpoint
                if i % 10 == 0:
                    self._save_model_state(model_name, current_state)

            self._save_model_state(model_name, current_state)
            print(f">>> Completed run for {model_name}.")