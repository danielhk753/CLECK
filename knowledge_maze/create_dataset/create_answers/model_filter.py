"""
llm_filter.py

Provides an automated filtering mechanism for dataset quality assurance using Large Language Models (LLMs).

This module defines the LLMFilter class, which evaluates counterfactual question-answer pairs
against a provided data table. It acts as a semantic and structural "gatekeeper" to ensure
that only logically sound, answerable, and well-formed samples proceed to the final evaluation
phase of the KnowledgeMaze pipeline.

The filter evaluates four primary dimensions:
    1. Structural Validity: Is the question linguistically sound and well-formed?
    2. Answerability: Is the question free of ambiguity and invalid operations (e.g.,
       impossible aggregations)?
    3. Answer Plausibility: Do the claim and evidence answers match the expected
       entity types (Who/Which/How many) and cardinality constraints?
    4. Retrievability: Can the evidence-based answer be deterministically derived
       from the provided counterfactual data table?

The module utilizes the OpenRouter API (defaulting to Gemini 3 Flash) to perform
zero-temperature inference, ensuring high-consistency verdicts across large datasets.
"""

import re
import json
from knowledge_maze.tools.api_tools import OpenRouterAPI


class LLMFilter:
    def __init__(self, model="google/gemini-3-flash-preview"):
        self.system_prompt = """You are an evaluation module for question quality, logical validity, and data-grounded answer plausibility.

You will receive:
1. A question
2. Two candidate answers
3. A table of data

Your task is to evaluate four properties:

1. QUESTION_VALID
Is the question linguistically and structurally well-formed?
- It must be understandable and not malformed.

2. QUESTION_ANSWERABLE
Can the question be answered in a well-defined way?
Reject (false) if ANY of the following hold:
- The question is ambiguous or underspecified
- It requires aggregation (e.g., "most common", "highest", "average") over an attribute that:
  - is unique per entity
  - is not comparable
  - is ill-defined in context
- The question applies an invalid operation (e.g., counting non-countable entities)

3. ANSWERS_PLAUSIBLE

Evaluate whether BOTH answers are structurally valid candidates.

Reject (false) if ANY of the following hold:

- The answer does not match the expected answer type:
  - "Who" → single person
  - "How many" → number
  - "Which" → single entity unless explicitly plural

- The answer violates cardinality:
  - multiple entities when a single one is required
  - concatenated names (e.g., "A B") instead of a single entity
  - lists, sets, or combined values when not requested

- The answer is not a clean span or identifiable entity from the data

Do NOT check correctness, only structural validity.


4. ANSWER_RETRIEVABLE
Can the SECOND answer be derived from the provided table?
- The answer must be supported by the data using valid operations (lookup, filtering, counting, comparison)
- If the table does not contain sufficient information, return false
- If the question itself is invalid or unanswerable, this must be false

Important rules:
- Do NOT use external knowledge
- Do NOT assume missing data
- Validate that the operation required by the question is compatible with the data schema

Output STRICT JSON only:

{
  "explanation": explain yourself in a sentence,
  "question_valid": true/false,
  "question_answerable": true/false,
  "answers_plausible": true/false,
  "answer_retrievable": true/false
}
"""

        self.api = OpenRouterAPI(
            system_prompt=self.system_prompt,
            model=model,
            temperature=0.0,
            max_rpm=1000
        )

    def _build_prompt(self, sample):
        context = sample['Answer']
        return f"""Question: {sample['Question']}
Answers:
- {context.get('Claim', 'N/A')}
- {context.get('Evidence', 'N/A')}
Table: {sample['Group name']}
{sample['Fake data']}
"""

    def _safe_parse(self, text):

        # Invalid type
        if not isinstance(text, str):
            return None

        text = re.sub(r"```(?:json)?", "", text)
        text = text.replace("```", "")

        candidates = []
        stack = []
        start = None

        for i, ch in enumerate(text):
            if ch == "{":
                if not stack:
                    start = i
                stack.append(ch)
            elif ch == "}":
                if stack:
                    stack.pop()
                    if not stack and start is not None:
                        candidates.append(text[start:i+1])

        required_keys = {
            "explanation",
            "question_valid",
            "question_answerable",
            "answers_plausible",
            "answer_retrievable"
        }

        valid_objs = []

        for candidate in candidates:
            try:
                obj = json.loads(candidate)

                if not isinstance(obj, dict):
                    continue

                if not required_keys.issubset(obj.keys()):
                    continue

                if not isinstance(obj["explanation"], str):
                    continue

                bool_keys = required_keys - {"explanation"}
                if not all(isinstance(obj[k], bool) for k in bool_keys):
                    continue

                valid_objs.append(obj)

            except json.JSONDecodeError:
                continue

        return valid_objs[-1] if valid_objs else None

    async def filter_dataset(self, dataset, return_explanations=False):
        print("Using LLM to filter bad samples...")

        prompts = [self._build_prompt(s) for s in dataset]

        raw_outputs = await self.api.async_run(prompts)

        filtered_dataset = []
        explanations = []

        for sample, output in zip(dataset, raw_outputs):
            parsed = self._safe_parse(output)

            if not parsed:
                continue

            verdict = all([
                parsed["question_valid"],
                parsed["question_answerable"],
                parsed["answers_plausible"],
                parsed["answer_retrievable"]
            ])

            if verdict:
                if return_explanations:
                    sample = dict(sample)  # avoid mutating original
                    sample["llm_filter_explanation"] = parsed["explanation"]
                filtered_dataset.append(sample)

            explanations.append(parsed)

        print(f"Discarded {len(dataset) - len(filtered_dataset)} bad samples.")

        if return_explanations:
            return filtered_dataset, explanations

        return filtered_dataset
