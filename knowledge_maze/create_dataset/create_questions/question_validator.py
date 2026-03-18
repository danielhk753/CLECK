import os
import re
import json
import random
import pandas as pd
import ipywidgets as widgets
from collections import Counter
import matplotlib.pyplot as plt
from IPython.display import display, clear_output, HTML

from knowledge_maze.tools.api_tools import gptAPI


VALIDATOR_SYSTEM_PROMPT = """
### ROLE
You are a strict Data Quality Auditor. You must validate the provided {Question} + {Table Data} pair against the following 7 rules.

### INPUT DATA
- **Strategy:** {strategy}
- **Group Name:** {group_name}
- **Question:** {question}
- **Table Data:** {table_data_snippet}

### RULES FOR VALIDITY

### RULE 1: **ANSWERABILITY**
   - The question must be answerable using ONLY the provided rows and the table description.
   - Reject if outside knowledge is required.

### RULE 2: **NO_TRIVIALITY**
   - Reject if the answer is obvious from the Group Name itself (e.g., Group: "Top 5 Players" -> Question: "How many players?").
   - Reject if the question just repeats a row value without adding insight.
   - *Exception:* In COUNT_TOTAL strategy it is ok that the question is just about counting the group size, unless the size is mentioned in the question or the group name.

### RULE 3: **NO_SPOILERS**
   - The question must NOT contain the answer within itself (e.g., "Why did X win 5 awards?" when the table value is "5").

### RULE 4: NATURAL_PHRASING (THE "AI-SPEAK" KILLER)
   - **STRICT REJECTION CRITERIA**: Reject if the question contains: "according to", "in this table", "provided records", "listed here", "the data shows", "based on the records", or "in this collection".
   - **QUIZ MASTER TEST**: If the question cannot be asked in a professional Pub Quiz without sounding like a robot reading a spreadsheet, REJECT.

### RULE 5: **LOGICAL_CONSISTENCY**
   - Verbs must match the data (e.g., don't say "won" if the column says "nominated").
   - Reject if the Categorization in the question contradicts the data (e.g., calling 'Hitters' 'Pitchers', or calling 'Cities' 'Countries'). The question must accurately describe the entities found in the rows.

### RULE 6: **CLOSED_WORLD_ASSUMPTION**
   - Does this specific set of rows make sense as a complete group?
   - Reject if it looks like a random partial list (e.g., "Players of FC Barcelona" but only 4 random rows are shown).
   - *Exception:* If the Group Name specifies a filter (e.g., "Top 4 Scorers"), then a small set is valid.

### RULE 7: CONTEXT_DEFINED
   - The question MUST be self-contained. A stranger should understand the subject and timeframe without seeing the table.
   - *Note for COMPARATIVE_DIFFERENCE*: While you may focus on the two entities being compared, you must still include the necessary timeframe or category (e.g., "In the 2022 census..." or "Among European capitals...").
   - STRICT REJECTION: The question must NOT use 'pointer' words like 'these', 'those', 'the following', or 'the listed'. The question must be 100% understandable if the table were deleted. If the question refers to a subset, it must name the entities or the specific criteria (e.g., 'The three players from France' instead of 'these three players').

### OUTPUT FORMAT
Return a SINGLE JSON object. 
IMPORTANT: Provide the "reasoning" FIRST to ensure logical consistency.

**Example Output:**
{
  "answerable": ["Data columns support the query", true],
  "no_triviality": ["Question asks for count which is in Group Name", false],
  "no_spoilers": ["No answer leakage detected", true],
  "natural_phrasing": ["Contains prohibited phrase 'according to the table'", false],
  "logical_consistency": ["Verbs match column headers", true],
  "closed_world": ["Group name implies a subset", true],
  "context_defined": ["Explicitly mentions 'Premier League'", true]
}
"""


class QuestionValidator:
    def __init__(self, input_data, strategy_definitions):
        """
        :param input_data: List of dicts (samples)
        :param strategy_definitions: The 'strategies' dictionary
        """
        self.input_data = input_data
        self.definitions = strategy_definitions.get('strategies', {})
        self.model = gptAPI(
            system_prompt=VALIDATOR_SYSTEM_PROMPT,
            model="google/gemini-3.1-pro-preview",
            # model="google/gemini-3-flash-preview",
            api_provider="openrouter",
            max_rpm=1000,
            max_concurrent_requests=100
        )

        # Internal state
        self.approved_data = []  # Passed everything
        self.rejected_data = []  # All failures

        # Formalized Definitions (Logic & Language Only)
        self.fail_definitions = {
            "PARSER_ERROR": "AI response was not valid JSON or was truncated.",
            "ANSWERABLE": "Table does not contain facts to answer the question.",
            "NO_TRIVIALITY": "Answer is already in the question or group name.",
            "NO_SPOILERS": "The question contains the answer within itself.",
            "NATURAL_PHRASING": "Robotic phrasing like 'according to the table'.",
            "LOGICAL_CONSISTENCY": "Verb-data mismatch or impossible timeframe.",
            "CLOSED_WORLD": "List feels arbitrarily partial (missing context).",
            "CONTEXT_DEFINED": "Question lacks scope/context to stand alone."
        }

    def _prepare_validation_input(self, sample):
        strategy = sample.get('Selected strategy')
        strat_def = self.definitions.get(strategy, {})

        # SHUFFLE to ensure the model isn't biased by row order
        table_data = list(sample.get('Relevant table data', []))
        random.shuffle(table_data)

        return f"""
        ### DATA TO AUDIT
        - Strategy Applied: {strategy}
        - Strategy Definition: {strat_def.get('definition')}
        - Strategy Requirements: {strat_def.get('requirements')}
        - Strategy "Must Reject": {strat_def.get('unacceptable_if')}

        ### TARGET DATA
        - Table Description: {sample.get('Group name')}
        - Table Data: {table_data}

        ### QUESTION TO EVALUATE
        "{sample.get('Question')}"
        """

    async def model_filter(self):
        """Phase 1: Logic and Language checks via Gemini with a Rescue Pass."""
        if not self.input_data:
            return

        # Helper for Round 1 and Round 2
        async def process_batch(batch):
            prompts = [self._prepare_validation_input(s) for s in batch]
            responses = await self.model.async_run(prompts)
            failures = []

            for i, response in enumerate(responses):
                sample = batch[i]
                try:
                    # Robustly find JSON object using regex
                    match = re.search(r'\{.*\}', response, re.DOTALL)
                    clean_json = match.group() if match else response
                    results = json.loads(clean_json)

                    # FIXED RULE CHECK: [Verdict (bool), Explanation (str)]
                    failed_rules = [
                        f"{k.upper()}: {v[1]}"
                        for k, v in results.items()
                        if isinstance(v, list) and len(v) >= 2 and v[0] is False
                    ]

                    if not failed_rules:
                        self.approved_data.append(sample)
                    else:
                        sample.update({'rejection_reasons': failed_rules, 'raw_rules': results, 'source': 'MODEL'})
                        self.rejected_data.append(sample)
                except Exception as e:
                    failures.append(sample)
            return failures

        # --- EXECUTION ---
        # Round 1
        retry_samples = await process_batch(self.input_data)

        # Round 2: The Rescue Pass for Parser Errors
        if retry_samples:
            final_failures = await process_batch(retry_samples)

            # Anything that still fails is a permanent PARSER_ERROR
            for sample in final_failures:
                # Store using the [Verdict, Reason] format
                sample.update({
                    'rejection_reasons': ["PARSER_ERROR: Persistent malformed JSON after retry."],
                    'raw_rules': {"PARSER_ERROR": [False, "Persistent malformed JSON after retry."]},
                    'source': 'MODEL'
                })
                self.rejected_data.append(sample)

    async def validate(self):
        """Runs the validation pipeline and renders the dashboard."""
        await self.model_filter()
        self.display_dashboard()

    def display_dashboard(self):
        """Renders a robust, dark-mode inline HTML dashboard."""
        total = len(self.input_data)
        approved = len(self.approved_data)
        rejected = len(self.rejected_data)
        pass_rate = (approved / total * 100) if total > 0 else 0

        # Extract stats
        all_reasons = []
        for item in self.rejected_data:
            fails = [k.upper() for k, v in item.get('raw_rules', {}).items() if
                     isinstance(v, list) and len(v) >= 2 and v[0] is False]
            all_reasons.extend(fails)

        reason_counts = Counter(all_reasons)
        sorted_rules = [r for r, count in reason_counts.most_common()]
        sorted_rules += [r for r in self.fail_definitions.keys() if r not in sorted_rules]

        # 1. Metric Cards
        stats_html = f"""
        <div style="display: flex; gap: 12px; margin-bottom: 25px;">
            <div style="flex: 1; background: #1e293b; padding: 15px; border-radius: 12px; border: 1px solid #334155; text-align: center;">
                <div style="color: #94a3b8; font-size: 10px; font-weight: 700; text-transform: uppercase;">Total Evaluated</div>
                <div style="font-size: 24px; font-weight: 800; color: #f8fafc;">{total}</div>
            </div>
            <div style="flex: 1; background: #064e3b; padding: 15px; border-radius: 12px; border: 1px solid #059669; text-align: center;">
                <div style="color: #34d399; font-size: 10px; font-weight: 700; text-transform: uppercase;">Approved</div>
                <div style="font-size: 24px; font-weight: 800; color: #34d399;">{approved}</div>
            </div>
            <div style="flex: 1; background: #450a0a33; padding: 15px; border-radius: 12px; border: 1px solid #991b1b66; text-align: center;">
                <div style="color: #fca5a5; font-size: 10px; font-weight: 700; text-transform: uppercase;">Rejected</div>
                <div style="font-size: 24px; font-weight: 800; color: #f87171;">{rejected}</div>
            </div>
            <div style="flex: 1; background: #1e293b; padding: 15px; border-radius: 12px; border: 1px solid #334155; text-align: center;">
                <div style="color: #38bdf8; font-size: 10px; font-weight: 700; text-transform: uppercase;">Pass Rate</div>
                <div style="font-size: 24px; font-weight: 800; color: #38bdf8;">{pass_rate:.1f}%</div>
            </div>
        </div>
        """

        # 2. Audit Table Rows (Volume removed, all text-align: left)
        rejection_rows = ""
        for rule in sorted_rules:
            count = reason_counts[rule]
            desc = self.fail_definitions.get(rule, "Model Validation Rule")

            rejection_rows += f"""
            <tr style="border-bottom: 1px solid #1e293b; background: #1e293b;">
                <td style="padding: 12px 15px; color: #f8fafc; font-size: 11px; font-weight: 700; text-align: left;">{rule}</td>
                <td style="padding: 12px 15px; color: #94a3b8; font-size: 11px; text-align: left;">{desc}</td>
                <td style="padding: 12px 15px; color: #f87171; font-weight: 800; font-size: 11px; text-align: left;">{count}</td>
            </tr>
            """

        # 3. Unique Rejection Samples
        failure_cards = ""
        shown_rules = set()
        for item in self.rejected_data:
            fails = [k.upper() for k, v in item.get('raw_rules', {}).items() if
                     isinstance(v, list) and len(v) >= 2 and v[0] is False]
            new_fails = [f for f in fails if f not in shown_rules]

            if new_fails:
                primary_fail = new_fails[0]
                shown_rules.add(primary_fail)

                q = item.get('Question', 'N/A')
                strat = item.get('Selected strategy', 'N/A')
                grp = item.get('Group name', 'N/A')
                wiki = item.get('Wiki link', '#')

                # Format bullet points
                reasons_li = "".join(
                    [f"<li style='margin-bottom: 4px;'>{r}</li>" for r in item.get('rejection_reasons', [])])

                # Style pandas table for dark mode safely
                view_data = item.get('Relevant table data', [])
                if view_data:
                    df_html = pd.DataFrame(view_data).style.set_table_styles([
                        {'selector': 'th',
                         'props': [('background-color', '#0f172a'), ('color', '#38bdf8'), ('font-size', '11px'),
                                   ('text-align', 'left')]},
                        {'selector': 'td',
                         'props': [('background-color', '#1e293b'), ('color', '#cbd5e1'), ('font-size', '11px'),
                                   ('border-bottom', '1px solid #0f172a'), ('text-align', 'left')]}
                    ]).hide(axis='index').to_html()
                else:
                    df_html = "<div style='color: #64748b; font-size: 11px; padding: 10px;'>No table data available.</div>"

                failure_cards += f"""
                <div style="background: #1e293b; padding: 16px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <span style="background: #7f1d1d; color: #fca5a5; padding: 4px 12px; border-radius: 20px; font-size: 10px; font-weight: 800; border: 1px solid #f87171;">{primary_fail}</span>
                        <a href="{wiki}" target="_blank" style="color: #38bdf8; font-size: 11px; text-decoration: none; font-weight: bold;">Source ↗</a>
                    </div>
                    <div style="color: #94a3b8; font-size: 11px; margin-bottom: 8px;"><b>STRATEGY:</b> <span style="color: #cbd5e1;">{strat}</span> | <b>GROUP:</b> <span style="color: #cbd5e1;">{grp}</span></div>
                    <div style="color: #f8fafc; font-weight: 700; font-size: 14px; margin-bottom: 12px; line-height: 1.4;">"{q}"</div>
                    <ul style="color: #fca5a5; font-size: 11px; margin-bottom: 15px; background: #450a0a33; padding: 10px 10px 10px 25px; border-radius: 6px; border-left: 3px solid #f87171;">
                        {reasons_li}
                    </ul>
                    <details>
                        <summary style="cursor: pointer; color: #38bdf8; font-size: 11px; font-weight: 700; text-transform: uppercase; margin-bottom: 8px;">▶ Inspect Table Data</summary>
                        <div style="border-radius: 6px; overflow: hidden; border: 1px solid #0f172a;">{df_html}</div>
                    </details>
                </div>
                """

        # 4. Assembly
        dashboard_html = f"""
        <div style="max-width: 1200px; background-color: #0f172a; padding: 30px; border-radius: 24px; font-family: sans-serif; color: #f8fafc; border: 1px solid #1e293b; box-sizing: border-box; text-align: left;">
            <h3 style="margin: 0 0 25px 0; font-size: 18px; color: #38bdf8; text-align: center; text-transform: uppercase; font-weight: 800; letter-spacing: 1px;">🛡️ Model-Based Data Quality Analysis (Gemini 3.1 pro preview)</h3>
            {stats_html}
            <div style="display: flex; gap: 20px; align-items: flex-start;">

                <div style="flex: 1.4; background: #1e293b; border-radius: 15px; border: 1px solid #334155; overflow: hidden; display: flex; flex-direction: column;">
                    <div style="padding: 12px 15px; border-bottom: 1px solid #334155; color: #94a3b8; font-size: 10px; font-weight: 700; text-transform: uppercase;">Failure Distribution</div>
                    <div style="max-height: 600px; overflow-y: auto;">
                        <table style="width: 100%; border-collapse: collapse;">
                            <thead>
                                <tr style="background: #0f172a; color: #64748b; font-size: 9px; text-transform: uppercase;">
                                    <th style="padding: 10px 15px; text-align: left;">Rule</th>
                                    <th style="padding: 10px 15px; text-align: left;">Definition</th>
                                    <th style="padding: 10px 15px; text-align: left;">Count</th>
                                </tr>
                            </thead>
                            <tbody>{rejection_rows}</tbody>
                        </table>
                    </div>
                </div>

                <div style="flex: 1.2; display: flex; flex-direction: column;">
                    <div style="color: #94a3b8; font-size: 10px; font-weight: 700; text-transform: uppercase; margin-bottom: 12px;">🔍 Deep-Dive: Unique Rule Exhibits</div>
                    <div style="max-height: 600px; overflow-y: auto; padding-right: 8px;">
                        {failure_cards if failure_cards else "<div style='color:#64748b; font-size:12px; font-style:italic;'>No rejections to display.</div>"}
                    </div>
                </div>

            </div>
        </div>
        """
        display(HTML(dashboard_html))


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
        # FULL SOLID COLOR: This makes the bar solid emerald
        self.progress.style = {'bar_color': '#27ae60'}
        self.pct_label = widgets.Label(value="0%", layout=widgets.Layout(margin='0 0 0 1%'))

        # NEW: Strategy Selection UI
        strats = sorted(list(set(s['Selected strategy'] for s in self.data)))
        self.strat_dropdown = widgets.Dropdown(
            options=['ALL STRATEGIES'] + strats,
            value='ALL STRATEGIES',
            description='Focus On:',
            layout=widgets.Layout(width='300px')
        )
        self.start_btn = widgets.Button(description="🚀 START", layout=widgets.Layout(width='100px'))
        self.start_btn.on_click(lambda x: self.run(10, self.strat_dropdown.value))  # Default 10 samples

    def _load_progress(self):
        if os.path.exists(self.save_file):
            with open(self.save_file, 'r') as f:
                return json.load(f)
        return []

    def _save_progress(self):
        with open(self.save_file, 'w') as f:
            json.dump(self.results, f, indent=4)

    def run(self, sample_size=10, filter_strategy='ALL STRATEGIES'):

        # 1. Filter out already tagged
        tagged_questions = {res['question'] for res in self.results}
        available_samples = [s for s in self.data if s['Question'] not in tagged_questions]

        # 2. NEW: Filter by specific strategy
        if filter_strategy != 'ALL STRATEGIES':
            available_samples = [s for s in available_samples if s['Selected strategy'] == filter_strategy]

        if not available_samples:
            with self.out:
                clear_output()
                print(f"✨ No untagged samples found for: {filter_strategy}")
                display(widgets.HBox([self.strat_dropdown, self.start_btn]))
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
            # Progress area with a light grey background track
            progress_box = widgets.VBox(
                [self.pct_label, self.progress],
                layout=widgets.Layout(
                    align_items='center',
                    background_color='#f0f0f0',  # Light Grey Track
                    padding='10px',
                    border_radius='5px',
                    margin='10px 0'
                )
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

        # 1. Aggregate: Count (total checked), Sum (successful), and Mean (rate)
        strat_stats = df.groupby('strategy')['score'].agg(['count', 'sum', 'mean']).sort_values('mean', ascending=False)
        strat_stats['mean_pct'] = strat_stats['mean'] * 100

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # --- PIE CHART (with white outline fix) ---
        ax1.pie([df['score'].sum(), len(df) - df['score'].sum()],
                labels=['Passed', 'Failed'], autopct='%1.1f%%',
                colors=['#2ecc71', '#e74c3c'], startangle=140,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        ax1.set_title(f"Cumulative Audit (n={len(df)})", fontsize=14, fontweight='bold')

        # --- BAR PLOT (with n/m labels) ---
        colors = ['#2ecc71' if x >= 70 else '#f1c40f' if x >= 40 else '#e74c3c' for x in strat_stats['mean_pct']]
        bars = ax2.bar(strat_stats.index, strat_stats['mean_pct'], color=colors, edgecolor='black', alpha=0.8)

        ax2.set_title("Success % per Strategy", fontsize=14, fontweight='bold')
        ax2.set_ylabel("Success Rate (%)")
        # Increase ylim to 130 to give space for the two-line text label
        ax2.set_ylim(0, 130)
        plt.xticks(rotation=45, ha='right')

        # Add percentage + (Success/Total) labels on top
        for i, bar in enumerate(bars):
            height = bar.get_height()
            total_checked = int(strat_stats.iloc[i]['count'])
            success_count = int(strat_stats.iloc[i]['sum'])

            # Create a 2-line label: "85%\n(17/20)"
            label_text = f"{height:.0f}%\n({success_count}/{total_checked})"

            ax2.text(bar.get_x() + bar.get_width() / 2., height + 2,
                     label_text, ha='center', va='bottom',
                     fontweight='bold', fontsize=10, color='#2c3e50')

        plt.tight_layout()
        plt.show()

    def display(self):
        with self.out:
            clear_output()
            # Show the selection menu
            display(HTML("<b style='color: #2c3e50;'>Pick Strategy to Tag:</b>"))
            display(widgets.HBox([self.strat_dropdown, self.start_btn]))

            if self.results:
                print(f"📦 {len(self.results)} samples already tagged in history.")

        display(self.out)
