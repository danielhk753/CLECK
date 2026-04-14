"""
Knowledge Maze: Distractor Validation & Noise Injection
------------------------------------------------------
This module implements Stage 5 of the pipeline: the identification and
verification of adversarial distractors. It ensures that the final QA
context is populated with semantically relevant but factually inert "noise."

Key Mechanisms:
1. **Semantic Retrieval**: Uses a TF-IDF vector space model to surface
   paragraphs from the global pool that share significant keyword overlap
   with the target question, ensuring high-fidelity distractors.
2. **Parallel Adversarial Sweep**: Orchestrates a massive asynchronous
   validation pass using 'The Investigator' (LLM) to identify "leaking"
   paragraphs—distractors that accidentally contain the answer.
3. **Automated Filtering**: Reconstructs the dataset by purging identified
   culprits and selecting the top-N safest adversarial candidates.

The resulting 'Distractor_IDs' are injected into the sample metadata,
forcing the model to distinguish between the 'Claim', the 'Evidence',
and high-similarity red herrings.
"""

import json
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from knowledge_maze.tools.api_tools import OpenRouterAPI

# PROMPT: The Batch Sweep (Identifies ALL culprits in one go)
BATCH_VALIDATOR_PROMPT = """
You are a strict Reading Comprehension Evaluator.
You will be provided with a question and a list of numbered reference paragraphs.

YOUR DIRECTIVE:
1. Identify IF the answer to the question can be found or easily inferred from any of the paragraphs.
2. Ignore your internal knowledge. Only use the provided text.
3. Output ONLY a JSON object containing a list of the paragraph IDs that contain the answer.
4. If NO paragraphs contain the answer, output an empty list.

Output Format: {"leaking_ids": ["id_1", "id_4"]} OR {"leaking_ids": []}

"""

class ParallelDistractorValidator:
    def __init__(self, dataset: List[Dict], paragraph_dataset: Dict[str, Dict[str, str]],
                 api_model: str = "google/gemini-3-flash-preview"):
        self.dataset = dataset
        self.paragraph_dataset = paragraph_dataset
        self.doc_ids = list(paragraph_dataset.keys())
        self.doc_contents = [v['Text'] for v in paragraph_dataset.values()]

        # Standard TF-IDF setup
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_contents)
        self.cosine_similarity = cosine_similarity

        self.api_model = api_model
        # The tool already handles the async logic internally
        self.batch_api = OpenRouterAPI(system_prompt=BATCH_VALIDATOR_PROMPT, model=api_model)
        self.data_with_distractors = []

    def _get_tfidf_queue(self, question: str, exclude_ids: List[str], top_n: int = 50) -> List[Tuple[str, str]]:
        query_vec = self.vectorizer.transform([question])
        similarities = self.cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        sorted_indices = similarities.argsort()[::-1]

        queue = []
        for idx in sorted_indices:
            doc_id = self.doc_ids[idx]
            if doc_id not in exclude_ids:
                queue.append((doc_id, self.doc_contents[idx]))
            if len(queue) == top_n: break
        return queue

    async def run_parallel_validation(self):
        """Processes the entire dataset in a single massive async_run call."""
        print(f"🚀 Preparing batch for {len(self.dataset)} samples...")

        all_prompts = []
        sample_metadata = []  # To keep track of which prompt belongs to which sample

        # 1. Pre-calculate all candidate batches (e.g., first 10 for everyone)
        for i, sample in enumerate(self.dataset):
            exclude = [str(sample.get('Evidence paragraph')), str(sample.get('Claim paragraph'))]
            queue = self._get_tfidf_queue(sample['Question'], exclude, top_n=10)

            formatted_context = "\n".join([f"[ID: {d[0]}]\n{d[1]}\n---" for d in queue])
            prompt = f"REFERENCE TEXT:\n{formatted_context}\n\nQUESTION: {sample['Question']}"

            all_prompts.append(prompt)
            sample_metadata.append({"index": i, "candidates": [d[0] for d in queue]})

        # 2. Fire the massive async batch via your existing tool
        print(f"📡 Sending {len(all_prompts)} prompts to OpenRouter...")
        raw_outputs = await self.batch_api.async_run(all_prompts, desc="Global Distractor Sweep")

        # 3. Reconstruct the dataset
        for i, output in enumerate(raw_outputs):
            try:
                raw_content = str(output).strip().strip("`").replace("json", "", 1).strip()
                response = json.loads(raw_content)
                leaking_ids = response.get("leaking_ids", []) if isinstance(response, dict) else response
            except:
                leaking_ids = sample_metadata[i]["candidates"]  # Reject all on error

            normalized_leaks = [str(lid).strip() for lid in leaking_ids]

            # Filter the 10 candidates down to safe ones
            safe_ids = [
                cid for cid in sample_metadata[i]["candidates"]
                if str(cid).strip() not in normalized_leaks
            ]

            # Update the original dataset row
            self.dataset[i]['Distractor_IDs'] = safe_ids[:8]

        self.data_with_distractors = self.dataset
        print(f"✅ Sweep complete. Finalized distractors for {len(self.dataset)} samples.")