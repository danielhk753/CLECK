import json
import random
import pandas as pd
from collections import Counter
from rich.jupyter import display
from IPython.display import display


from knowledge_maze.tools.api_tools import gptAPI

VALIDATE_PROMPT = """
### ROLE
You are a Data Quality Auditor specializing in structured data and Natural Language Processing. 
Your goal is to ensure that a generated question is logical, non-trivial, and factually supported by the data.

### CRITERIA FOR VALIDITY
1. **ANSWERABILITY**: The question MUST be answerable using ONLY the provided 'Table Data'. No outside knowledge.
2. **TRIVIALITY**: The question is INVALID if it is too simple (e.g., asking for a value already mentioned in the question) 
or if the answer is explicitly stated in the 'Group Name' (e.g., asking "How many are there?" when the group is "Top 10 players").
3. **IRRATIONALITY**: The question must be grammatically and logically sound. Avoid "hallucinated" verbs that don't match 
the data (e.g., don't say "won" if the data only says "nominated").
4. **CONTEXT**: The question MUST use the 'Group Name' to define its scope so it is self-contained for a third party.
5. **STRATEGY ALIGNMENT**: The question must follow the specific logic of the assigned strategy.

### OUTPUT FORMAT
You must return a valid JSON object only. No prose.
{
  "is_valid": boolean,
  "reason": "Clear explanation of why it passed or failed"
}
"""


# class QuestionValidator:
#     def __init__(self, strategy_definitions):
#         self.model = gptAPI(
#             system_prompt=VALIDATE_PROMPT,
#             model="google/gemini-2.5-flash",
#             api_provider="openrouter",
#             max_rpm=500,
#             max_concurrent_requests=100
#         )
#         self.definitions = strategy_definitions['strategies']
#
#     async def validate_samples(self, samples):
#         """
#         Takes a list of samples and returns a list of booleans.
#         """
#         if not samples:
#             return []
#
#         # 1. Prepare all prompts in parallel
#         prompts = [self._prepare_validation_input(s) for s in samples]
#
#         # 2. Fire all requests concurrently using the model's async_run
#         raw_responses = await self.model.async_run(prompts)
#
#         # 3. Parse results
#         validation_results = []
#         for i, response in enumerate(raw_responses):
#             try:
#                 # Basic check for empty/null questions before parsing JSON
#                 if not samples[i].get('Question') or str(samples[i]['Question']).lower() in ['null', 'none']:
#                     validation_results.append(False)
#                     continue
#
#                 result = json.loads(response)
#                 validation_results.append(result.get("is_valid", False))
#             except (json.JSONDecodeError, IndexError, TypeError):
#                 # If the LLM fails to provide valid JSON, we fail the sample for safety
#                 validation_results.append(False)
#
#         return validation_results
#
#     def _prepare_validation_input(self, sample):
#         strategy = sample.get('Selected strategy')
#         strat_def = self.definitions.get(strategy, {})
#
#         return f"""
#         ### DATA TO AUDIT
#         - Group Name: {sample.get('Group name')}
#         - Strategy Applied: {strategy}
#         - Strategy Definition: {strat_def.get('definition')}
#         - Strategy Requirements: {strat_def.get('requirements')}
#
#         ### TARGET DATA
#         - Table Data: {sample.get('Relevant table data')}
#
#         ### QUESTION TO EVALUATE
#         "{sample.get('Question')}"
#         """

class QuestionValidator:
    def __init__(self, strategy_definitions):
        self.model = gptAPI(
            system_prompt=VALIDATE_PROMPT,
            model="google/gemini-2.5-flash",
            api_provider="openrouter",
            max_rpm=500,
            max_concurrent_requests=100
        )
        self.definitions = strategy_definitions['strategies']

        # New Storage attributes
        self.approved_data = []
        self.not_approved = []

    async def validate_samples(self, samples):
        """
        Processes a list of samples, populating approved_data and not_approved.
        """
        if not samples:
            return

        prompts = [self._prepare_validation_input(s) for s in samples]
        raw_responses = await self.model.async_run(prompts)

        for i, response in enumerate(raw_responses):
            sample = samples[i]
            try:
                # Clean and parse JSON
                clean_json = response.strip()
                if "```json" in clean_json:
                    clean_json = clean_json.split("```json")[1].split("```")[0].strip()

                result = json.loads(clean_json)
                is_valid = result.get("is_valid", False)
                reason = result.get("reason", "No reason provided.")

                if is_valid:
                    self.approved_data.append(sample)
                else:
                    # Store the sample along with the auditor's feedback
                    failed_entry = sample.copy()
                    failed_entry['rejection_reason'] = reason
                    self.not_approved.append(failed_entry)

            except Exception as e:
                sample['rejection_reason'] = f"Parser Error: {str(e)}"
                self.not_approved.append(sample)

        self.print_summary()

    def print_summary(self):
        """
        Prints a detailed report of the validation results.
        """
        total_approved = len(self.approved_data)
        total_rejected = len(self.not_approved)
        total = total_approved + total_rejected
        pass_rate = (total_approved / total * 100) if total > 0 else 0

        print("\n" + "=" * 50)
        print("📊 VALIDATION SUMMARY")
        print("=" * 50)
        print(f"Total Samples:   {total}")
        print(f"✅ Approved:     {len(self.approved_data)}")
        print(f"❌ Rejected:     {len(self.not_approved)}")
        print(f"📈 Pass Rate:    {pass_rate:.1f}%")
        print("-" * 50)

        # --- ADDED: REJECTION DISTRIBUTION ---
        if total_rejected > 0:
            print("📉 REJECTIONS BY STRATEGY:")
            # Extract strategy names from the not_approved list
            rejected_strats = [item.get('Selected strategy', 'Unknown') for item in self.not_approved]
            rej_counts = Counter(rejected_strats)

            # Sort by highest number of failures
            for strat, count in rej_counts.most_common():
                print(f"  - {strat:<25}: {count}")
            print("-" * 50)
        # --------------------------------------

        if self.not_approved:
            print("\n🔍 TOP REJECTION EXAMPLES:")
            # Show up to 5 examples of failures
            for i, item in enumerate(self.not_approved[:5]):
                print(f"\n{i + 1}. Strategy: {item.get('Selected strategy')}")
                print(f"   Question: {item.get('Question')}")
                print(f"   Data: {item.get('Group name')}")
                display(pd.DataFrame(item.get('Relevant table data')))
                print(f"   🚩 Reason: {item.get('rejection_reason')}")

        print("=" * 50 + "\n")

    def _prepare_validation_input(self, sample):
        strategy = sample.get('Selected strategy')
        strat_def = self.definitions.get(strategy, {})

        # --- SHUFFLE LOGIC ---
        # We create a copy so we don't permanently mess up the original sample data
        table_data = list(sample.get('Relevant table data', []))
        random.shuffle(table_data)
        # ---------------------

        return f"""
        ### DATA TO AUDIT
        - Strategy Applied: {strategy}
        - Strategy Definition: {strat_def.get('definition')}
        - Strategy Requirements: {strat_def.get('requirements')}

        ### TARGET DATA
        - Table Description: {sample.get('Group name')}
        - Table Data: {table_data}

        ### QUESTION TO EVALUATE
        "{sample.get('Question')}"
        """