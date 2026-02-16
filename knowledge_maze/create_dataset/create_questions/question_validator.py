import os
import json
import random
import pandas as pd
import tqdm.auto as tqdm_lib
import ipywidgets as widgets
from collections import Counter
import matplotlib.pyplot as plt
from IPython.display import display, clear_output, HTML

# Phase 1 - Tag and refine:
# TODO: 1. Tag 50-100 samples and write the problems we have.
# TODO: 2. Fix necessary problems.

# Phase 2 - Organize Code + Visualization
# TODO: 1. Add another 2-layered pie chart of distribution by type of reasoning.
# TODO: 2. In visualizations - distribution of evidences, TF-IDF based category distribution.
# TODO: 3. Show sample (in visualizations).
# TODO: 4. Add all the json savings into tools, add a saving mechanism to the process + simple functions to call everything (in pipeline).
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
    def __init__(self, generated_data, strategy_definitions, save_file="tagger_progress.json"):
        self.data = generated_data
        self.definitions = strategy_definitions.get("strategies", strategy_definitions)
        self.save_file = save_file

        # Load existing progress if it exists
        self.results = self._load_progress()

        total_in_file = len(self.results)
        print(f"📦 Progress Loaded: {total_in_file} samples already tagged in '{save_file}'.")

        self.current_idx = 0
        self.sample_pool = []

        # UI Components
        self.out = widgets.Output()

        # Slighter Progress Bar: Reduced width to 98% and height to 18px for a sleeker look
        self.progress = widgets.IntProgress(
            min=0,
            layout=widgets.Layout(
                width='98%',
                height='18px',
                border='1px solid #666',
                border_radius='1px',
                margin='0 auto'  # Center it
            )
        )
        self.pct_label = widgets.Label(value="0%", layout=widgets.Layout(margin='0 0 0 1%'))

    def _load_progress(self):
        if os.path.exists(self.save_file):
            with open(self.save_file, 'r') as f:
                return json.load(f)
        return []

    def _save_progress(self):
        with open(self.save_file, 'w') as f:
            json.dump(self.results, f, indent=4)

    def run(self, sample_size=10):
        # 1. Filter out samples already tagged (using Question as unique key)
        tagged_questions = {res['question'] for res in self.results}
        available_samples = [s for s in self.data if s['Question'] not in tagged_questions]

        if not available_samples:
            print("✨ All samples have already been tagged!")
            return

        random.shuffle(available_samples)
        self.sample_pool = available_samples[:sample_size]

        self.progress.max = sample_size
        self.current_idx = 0
        self._update_progress_ui()
        self._show_sample()

    def _update_progress_ui(self):
        self.progress.value = self.current_idx
        pct = (self.current_idx / self.progress.max) * 100
        self.pct_label.value = f"Session Progress: {pct:.0f}%"

    def _show_sample(self):
        with self.out:
            clear_output(wait=True)
            if self.current_idx >= len(self.sample_pool):
                self._finish()
                return

            sample = self.sample_pool[self.current_idx]
            strat = sample['Selected strategy']
            strat_def = self.definitions.get(strat, {}).get('definition', 'No definition found.')
            wiki_url = sample.get('Wiki link', '#')

            df = pd.DataFrame(sample['Relevant table data'])
            table_html = df.to_html(classes='data', index=False)

            table_style = """
            <style>
                .data { border-collapse: collapse; margin: 0 auto; background-color: white; border-radius: 4px; width: 100%; }
                .data th, .data td { padding: 8px 12px; border: 1px solid #ddd; text-align: left; }
                .data th { background-color: #f2f2f2; position: sticky; top: 0; z-index: 10; }
                .wiki-link { 
                    text-decoration: none; color: #27ae60; font-weight: bold; font-size: 0.85em;
                    padding: 4px 12px; border: 1px solid rgba(46, 204, 113, 0.4); border-radius: 20px;
                    background: rgba(255,255,255,0.8); transition: 0.3s;
                }
                .wiki-link:hover { background: #27ae60; color: white; }

                /* Custom Scrollbar for a cleaner look */
                .scroll-area::-webkit-scrollbar { width: 6px; }
                .scroll-area::-webkit-scrollbar-track { background: transparent; }
                .scroll-area::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.1); border-radius: 10px; }
            </style>
            """

            # FIXED HEIGHT CONFIGURATION (Adjust 450px to your screen preference)
            fixed_height = "450px"

            layout_html = f"""
            {table_style}
            <div style="display: flex; gap: 20px; font-family: 'Segoe UI', sans-serif; height: {fixed_height};">

                <div style="flex: 1; padding: 20px; border: 1px solid rgba(52, 152, 219, 0.3); 
                            border-radius: 12px; background-color: rgba(52, 152, 219, 0.05); 
                            overflow-y: auto;" class="scroll-area">
                    <h4 style="margin:0; color: #2c3e50; border-bottom: 1px solid rgba(0,0,0,0.1); padding-bottom: 8px;">
                        📋 Sample {self.current_idx + 1} / {len(self.sample_pool)}
                    </h4>
                    <p style="margin: 10px 0;"><b>Group:</b> {sample['Group name']}</p>
                    <p style="margin: 5px 0;"><b>Strategy:</b> <span style="color: #2980b9; font-weight: bold;">{strat}</span></p>
                    <p style="font-size: 0.85em; color: #666;"><i>{strat_def}</i></p>
                    <div style="margin-top: 15px; padding: 15px; background: rgba(255, 255, 255, 0.7); 
                                border-left: 5px solid #3498db; border-radius: 4px;">
                        <b style="color: #2c3e50; font-size: 0.8em; letter-spacing: 1px;">QUESTION:</b><br>
                        <span style="font-size: 1.15em; color: #111;">{sample['Question']}</span>
                    </div>
                </div>

                <div style="flex: 1.5; padding: 20px; border: 1px solid rgba(46, 204, 113, 0.3); 
                            border-radius: 12px; background-color: rgba(46, 204, 113, 0.05); 
                            display: flex; flex-direction: column;">

                    <div style="display: flex; justify-content: space-between; align-items: center; width: 100%; margin-bottom: 15px;">
                        <h4 style="margin: 0; color: #27ae60;">📊 Data Context</h4>
                        <a href="{wiki_url}" target="_blank" class="wiki-link">🌐 View Wikipedia</a>
                    </div>

                    <div style="flex-grow: 1; overflow-y: auto; display: flex; align-items: flex-start; justify-content: center;" class="scroll-area">
                        <div style="width: 100%;">
                             {table_html}
                        </div>
                    </div>
                </div>
            </div>
            """
            display(HTML(layout_html))

            # Buttons remain at exactly the same Y-coordinate now
            btn_layout = widgets.Layout(width='140px', height='35px', margin='0 5px')

            btn_good = widgets.Button(
                description="✅ VALID",
                style={'button_color': '#2ecc71', 'font_weight': 'bold'},
                layout=btn_layout
            )
            btn_bad = widgets.Button(
                description="❌ INVALID",
                style={'button_color': '#e74c3c', 'font_weight': 'bold'},
                layout=btn_layout
            )
            btn_quit = widgets.Button(
                description="💾 SAVE & QUIT",
                style={'button_color': '#7f8c8d', 'font_weight': 'bold'},
                layout=widgets.Layout(width='140px', height='35px', margin='0 5px')
            )

            btn_good.on_click(lambda x: self._handle_click(1))
            btn_bad.on_click(lambda x: self._handle_click(0))
            btn_quit.on_click(lambda x: self._finish())

            # Centering the buttons using justify_content
            button_box = widgets.HBox(
                [btn_good, btn_bad, btn_quit],
                layout=widgets.Layout(margin='20px 0', justify_content='center')
            )

            # Progress area centered as well
            progress_box = widgets.VBox(
                [self.pct_label, self.progress],
                layout=widgets.Layout(align_items='center')
            )

            display(button_box)
            display(progress_box)

    def _handle_click(self, score):
        sample = self.sample_pool[self.current_idx]
        self.results.append({
            "group": sample['Group name'],
            "strategy": sample['Selected strategy'],
            "question": sample['Question'],
            "score": score
        })
        self._save_progress()  # Save after every click
        self.current_idx += 1
        self._update_progress_ui()
        self._show_sample()

    def _finish(self):
        with self.out:
            clear_output()
            print(f"💾 Progress saved to {self.save_file}")
            if self.results:
                self.plot_results()

    def plot_results(self):
        df = pd.DataFrame(self.results)
        # Use only the results from the total history for the graph
        strat_stats = df.groupby('strategy')['score'].agg(['count', 'mean']).sort_values('mean', ascending=False)
        strat_stats['mean_pct'] = strat_stats['mean'] * 100

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        ax1.pie(
            [df['score'].sum(), len(df) - df['score'].sum()],
            labels=['Passed', 'Failed'],
            autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c'],
            startangle=140,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}  # This creates the internal outline
        )
        ax1.set_title(f"Cumulative Audit ({len(df)} Samples)", fontsize=14, fontweight='bold')

        colors = ['#2ecc71' if x >= 70 else '#f1c40f' if x >= 40 else '#e74c3c' for x in strat_stats['mean_pct']]
        bars = ax2.bar(strat_stats.index, strat_stats['mean_pct'], color=colors, edgecolor='black', alpha=0.8)
        ax2.set_title("Success % per Strategy", fontsize=14, fontweight='bold')
        ax2.set_ylabel("Success Rate (%)")
        ax2.set_ylim(0, 110)
        plt.xticks(rotation=45, ha='right')

        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 2, f'{height:.0f}%', ha='center', fontweight='bold')

        plt.tight_layout()
        plt.show()

    def display(self):
        display(self.out)
