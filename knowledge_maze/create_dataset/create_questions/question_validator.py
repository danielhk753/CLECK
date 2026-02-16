import json
import random
import pandas as pd
import tqdm.auto as tqdm_lib
from collections import Counter
import matplotlib.pyplot as plt
from IPython.display import display, clear_output, Markdown

# Phase 1 - Data Quality
# TODO: 4. Manual tagging mechanism.

# Phase 2 - Organize Code + Visualization
# TODO: 1. Add all the json savings into tools, add a saving mechanism to the process + simple functions to call everything (in pipeline).
# TODO: 2. Add another 2-layered pie chart of distribution by type of reasoning.
# TODO: 3. In visualizations - distribution of evidences, TF-IDF based category distribution.
# TODO: 4. Show sample (in visualizations).
# TODO: 5. After all of that i want a method flow-chart.

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


class QuestionValidator:
    def __init__(self, input_data, strategy_definitions):

        self.input_data = input_data
        self.filtered_data = []

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

        self.manual_log = {
            "✅ Approved": 0,
            "❌ Missing Question": 0,
            "❌ Missing Table Data": 0,
            "❌ Missing Data Columns": 0,
            "❌ Extra Data Columns": 0,
            "❌ Missing Data": 0,
        }

    def manual_filter(self):
        from collections import defaultdict

        # Track rejections per strategy: {Strategy: {Reason: Count}}
        strategy_rejections = defaultdict(lambda: defaultdict(int))

        for sample in tqdm_lib.tqdm(self.input_data, desc="Manual Filtering"):
            strat = sample.get('Selected strategy', 'UNKNOWN')
            reason = None

            # 1. Question String Check
            q = str(sample.get('Question', '')).strip().lower()
            if q in ['null', 'null.', 'none', '']:
                reason = '❌ Missing Question'

            # 2. Column logic
            if not reason:
                relevant_data = sample.get('Relevant table data', [])
                if not relevant_data:
                    reason = '❌ Missing Table Data'
                else:
                    entity_col = list(relevant_data[0].keys())[0]
                    target_cols = [k for k in relevant_data[0].keys() if k != entity_col]

                    if len(target_cols) > 1:
                        reason = '❌ Extra Data Columns'
                    elif len(target_cols) < 1 and strat != "COUNT_TOTAL":
                        reason = '❌ Missing Data Columns'
                    elif any(all(row.get(col) is None for row in relevant_data) for col in target_cols):
                        reason = '❌ Missing Data'

            if reason:
                self.manual_log[reason] += 1
                strategy_rejections[strat][reason] += 1
                continue

            self.manual_log['✅ Approved'] += 1
            self.filtered_data.append(sample)

        # --- PRINT SUMMARY ---
        print("\n" + "═" * 40)
        print("📊 MANUAL FILTERING SUMMARY".center(36))
        print("-" * 40)
        print(f"{'📊 Total Samples':<28} | {len(self.input_data)}")
        for key, val in self.manual_log.items():
            if val > 0: print(f"{key:<28} | {val}")

        pass_rate = 100 * self.manual_log['✅ Approved'] / len(self.input_data)
        print(f"{'📈 Pass Rate:':<28} | {pass_rate:.1f}%")
        print("-" * 40)

        # --- REJECTIONS PER STRATEGY TABLE ---
        if strategy_rejections:
            print("🚩 REJECTIONS BY STRATEGY".center(36))
            print("-" * 40)
            # Sort strategies by total number of rejections
            sorted_strats = sorted(strategy_rejections.keys(),
                                   key=lambda s: sum(strategy_rejections[s].values()),
                                   reverse=True)

            for s in sorted_strats:
                total_s_rejections = sum(strategy_rejections[s].values())
                print(f"{s:<28} | {total_s_rejections}")

        print("═" * 40 + "\n")

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

    async def model_filter(self):
        """
        Processes a list of samples, populating approved_data and not_approved.
        """

        prompts = [self._prepare_validation_input(s) for s in self.filtered_data]
        raw_responses = await self.model.async_run(prompts)

        for i, response in enumerate(raw_responses):
            sample = self.filtered_data[i]
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
                print(f"   Wikipedia URL: {item.get('Wiki link')}")
                print(f"   Data: {item.get('Group name')}")
                display(pd.DataFrame(item.get('Relevant table data')))
                print(f"   🚩 Reason for rejection: {item.get('rejection_reason')}")

        print("=" * 50 + "\n")

    async def validate(self):
        self.manual_filter()
        await self.model_filter()


class ManualTagger:
    def __init__(self, generated_data, strategy_definitions):
        self.data = generated_data
        self.definitions = strategy_definitions.get("strategies", strategy_definitions)
        self.results = []  # Stores (sample, score)

    def run(self, sample_size=10):
        # 1. Randomize and Sample
        pool = list(self.data)
        random.shuffle(pool)
        target_samples = pool[:sample_size]

        print(f"🎯 Starting manual audit for {len(target_samples)} samples...")

        for i, sample in enumerate(target_samples):
            self._display_ui(i + 1, len(target_samples), sample)

            # 2. Capture Input
            choice = input("\nIs this sample good? (y = Yes, n = No, q = Quit): ").lower().strip()

            if choice == 'q':
                break

            score = 1 if choice == 'y' else 0
            self.results.append({
                "strategy": sample['Selected strategy'],
                "score": score
            })

        clear_output()
        print("✅ Tagging Complete!")
        self.plot_results()

    def _display_ui(self, current, total, sample):
        clear_output(wait=True)
        strat = sample['Selected strategy']
        strat_def = self.definitions.get(strat, {}).get('definition', 'No definition found.')

        # Header Info
        display(Markdown(f"### 📋 Sample {current} / {total}"))
        print("=" * 80)
        print(f"GROUP NAME: {sample['Group name']}")
        print(f"WIKI LINK:  {sample.get('Wiki link', 'N/A')}")
        print(f"STRATEGY:   {strat}")
        print(f"DEFINITION: {strat_def}")
        print("-" * 80)

        # The Meat: Question & Data
        display(Markdown(f"**QUESTION:** `{sample['Question']}`"))
        print("-" * 80)
        print("RELEVANT DATA VIEW:")
        df = pd.DataFrame(sample['Relevant table data'])
        display(df)
        print("=" * 80)

    def plot_results(self):
        if not self.results:
            print("No data to plot.")
            return

        df = pd.DataFrame(self.results)

        # Calculate stats
        total_good = df['score'].sum()
        total_bad = len(df) - total_good

        # Strategy breakdown
        strat_stats = df.groupby('strategy')['score'].agg(['count', 'mean'])
        strat_stats['mean'] *= 100  # Convert to percentage

        # Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Overall Pie Chart
        ax1.pie([total_good, total_bad], labels=['Good', 'Bad'], autopct='%1.1f%%',
                colors=['#4CAF50', '#F44336'], startangle=90, explode=(0.1, 0))
        ax1.set_title(f"Overall Manual Score (n={len(df)})")

        # 2. Success Rate by Strategy
        strat_stats['mean'].sort_values().plot(kind='barh', ax=ax2, color='skyblue')
        ax2.set_title("Success Rate % by Strategy")
        ax2.set_xlabel("Success Percentage")
        ax2.set_xlim(0, 100)

        plt.tight_layout()
        plt.show()

        print(f"📊 Final Audit Result: {total_good}/{len(df)} samples passed.")
