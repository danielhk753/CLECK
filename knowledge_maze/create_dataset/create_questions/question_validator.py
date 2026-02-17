import os
import json
import random
import pandas as pd
import ipywidgets as widgets
from collections import Counter
import matplotlib.pyplot as plt
from IPython.display import display, clear_output, HTML

# Phase 1 - Tag and refine:
# TODO: 1. Tag 50-100 samples and write the problems we have.
# TODO: 2. Add rules to the LLM validator.
# TODO: 3. Fix necessary other problems.

# Phase 2 - Organize Code + Visualization
# TODO: 1. Add another 2-layered pie chart of distribution by type of reasoning.
# TODO: 2. In visualizations - distribution of evidences, TF-IDF based category distribution.
# TODO: 3. Show sample (in visualizations).
# TODO: 4. Add all the json savings into tools, add a saving mechanism to the process + simple functions to call everything (in pipeline).
# TODO: 5. After all of that i want a method flow-chart.

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

1. **ANSWERABILITY**
   - The question must be answerable using ONLY the provided rows and the table description.
   - Reject if outside knowledge is required.

2. **NO_TRIVIALITY**
   - Reject if the answer is obvious from the Group Name itself (e.g., Group: "Top 5 Players" -> Question: "How many players?").
   - Reject if the question just repeats a row value without adding insight.
   - *Exception:* In COUNT_TOTAL strategy it is ok that the question is just about counting the group size, unless the size is mentioned in the question or the group name.

3. **NO_SPOILERS**
   - The question must NOT contain the answer within itself (e.g., "Why did X win 5 awards?" when the table value is "5").

4. **NATURAL_PHRASING** (No "AI-Speak")
   - **Strictly Reject** phrases like "according to the data", "in this table", "provided records", or "based on the list".
   - The question must sound like a natural human curiosity.

5. **LOGICAL_CONSISTENCY**
   - Verbs must match the data (e.g., don't say "won" if the column says "nominated").

6. **CLOSED_WORLD_ASSUMPTION**
   - Does this specific set of rows make sense as a complete group?
   - Reject if it looks like a random partial list (e.g., "Players of FC Barcelona" but only 4 random rows are shown).
   - *Exception:* If the Group Name specifies a filter (e.g., "Top 4 Scorers"), then a small set is valid.

7. **CONTEXT_DEFINED**
   - The question MUST mention the specific context (usually the Group Name) so it is self-contained.
   - *Exception:* This is **NOT** relevant for the strategy of COMPARATIVE_DIFFERENCE because we don't need to group name for it.

### OUTPUT FORMAT
Return a SINGLE JSON object.
Keys must be the exact rule names below.
Values must be a tuple: `[boolean, "Specific evidence or short reasoning"]`.

**Example Output:**
{
  "answerable": [true, "Data columns support the query"],
  "no_triviality": [false, "Question asks for count which is in Group Name"],
  "no_spoilers": [true, "No answer leakage detected"],
  "natural_phrasing": [false, "Contains prohibited phrase 'according to the table'"],
  "logical_consistency": [true, "Verbs match column headers"],
  "closed_world": [true, "Group name implies a subset"],
  "context_defined": [true, "Explicitly mentions 'Premier League'"]
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
            model="google/gemini-2.5-flash",
            api_provider="openrouter",
            max_rpm=500,
            max_concurrent_requests=100
        )

        # Internal state
        self.filtered_data = []  # Passed manual checks
        self.approved_data = []  # Passed everything
        self.rejected_data = []  # All failures (Manual + Model)

        # Formalized Definitions (Unified)
        self.fail_definitions = {
            "MISSING_QUESTION": "Question field was empty or null.",
            "MISSING_TABLE": "No table data was found for this sample.",
            "EXTRA_COLUMNS": "Table contains more than one attribute column (schema violation).",
            "MISSING_COLUMNS": "Table lacks an attribute column for this strategy.",
            "EMPTY_DATA": "Attribute column exists but has no data rows.",
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

    def manual_filter(self):
        """Phase 1: Structure and Schema checks."""
        for sample in self.input_data:
            strat = sample.get('Selected strategy', 'UNKNOWN')
            reason = None

            q = str(sample.get('Question', '')).strip().lower()
            table = sample.get('Relevant table data', [])

            if q in ['null', 'null.', 'none', '']:
                reason = "MISSING_QUESTION"
            elif not table:
                reason = "MISSING_TABLE"
            else:
                cols = list(table[0].keys())
                target_cols = cols[1:]  # Skip first (entity) col

                if len(target_cols) > 1:
                    reason = "EXTRA_COLUMNS"
                elif len(target_cols) < 1 and strat != "COUNT_TOTAL":
                    reason = "MISSING_COLUMNS"
                elif any(all(row.get(col) is None for row in table) for col in target_cols):
                    reason = "EMPTY_DATA"

            if reason:
                # Format to match the "tuple" style of the model results for unified stats
                sample['raw_rules'] = {reason: [False, self.fail_definitions[reason]]}
                sample['rejection_reasons'] = [f"{reason}: {self.fail_definitions[reason]}"]
                sample['source'] = 'MANUAL'
                self.rejected_data.append(sample)
            else:
                self.filtered_data.append(sample)

    async def model_filter(self):
        """Phase 2: Logic and Language checks via Gemini."""
        if not self.filtered_data:
            return

        prompts = [self._prepare_validation_input(s) for s in self.filtered_data]
        raw_responses = await self.model.async_run(prompts)

        for i, response in enumerate(raw_responses):
            sample = self.filtered_data[i]
            try:
                # Clean JSON fences
                clean_json = response.strip()
                if "```json" in clean_json:
                    clean_json = clean_json.split("```json")[1].split("```")[0].strip()

                results = json.loads(clean_json)

                # Rule check: [True/False, "Reason"]
                failed_rules = [f"{k.upper()}: {v[1]}" for k, v in results.items() if v[0] is False]

                if not failed_rules:
                    self.approved_data.append(sample)
                else:
                    sample['rejection_reasons'] = failed_rules
                    sample['raw_rules'] = results
                    sample['source'] = 'MODEL'
                    self.rejected_data.append(sample)

            except Exception as e:
                sample['rejection_reasons'] = [f"PARSER_ERROR: {str(e)}"]
                sample['raw_rules'] = {"PARSER_ERROR": [False, str(e)]}
                sample['source'] = 'MODEL'
                self.rejected_data.append(sample)

    async def validate(self):
        self.manual_filter()
        await self.model_filter()
        self.print_summary()

    def print_summary(self):
        total = len(self.input_data)
        total_rej = len(self.rejected_data) or 1
        pass_rate = (len(self.approved_data) / total * 100) if total > 0 else 0

        all_reasons = []
        rejected_strats = []
        for item in self.rejected_data:
            rejected_strats.append(item.get('Selected strategy', 'UNKNOWN'))
            fails = [k.upper() for k, v in item['raw_rules'].items() if v[0] is False]
            all_reasons.extend(fails)

        reason_counts = Counter(all_reasons)
        # Sort ALL rules: Failed rules first (descending), then 0-count rules
        sorted_rules = [r for r, count in reason_counts.most_common()]
        sorted_rules += [r for r in self.fail_definitions.keys() if r not in sorted_rules]

        # Strategy Pie Logic (Sorted by frequency)
        strat_counts = Counter(rejected_strats).most_common()
        colors = ["#3498db", "#e74c3c", "#f1c40f", "#2ecc71", "#9b59b6", "#e67e22"]
        pie_parts = []
        current_pct = 0
        for i, (strat, count) in enumerate(strat_counts):
            pct = (count / total_rej) * 100
            pie_parts.append(f"{colors[i % len(colors)]} {current_pct}% {current_pct + pct}%")
            current_pct += pct

        # --- THE DASHBOARD ---
        dashboard_html = f"""
        <div style="font-family: sans-serif; background: #f8f9fa; padding: 25px; border-radius: 12px; border: 1px solid #dee2e6;">
            <h2 style="text-align: center; color: #2c3e50; margin-bottom: 25px; font-weight: 800; letter-spacing: -1px;">🛡️ PIPELINE QUALITY CONTROL</h2>

            <div style="display: flex; gap: 20px; margin-bottom: 25px;">
                <div style="flex: 1; background: #2c3e50; padding: 15px; border-radius: 8px; text-align: center; color: white;">
                    <h4 style="margin:0; opacity: 0.8; font-size: 0.7em; letter-spacing: 1px;">TOTAL SAMPLES</h4><h1 style="margin:5px 0;">{total}</h1>
                </div>
                <div style="flex: 1; background: #27ae60; padding: 15px; border-radius: 8px; text-align: center; color: white;">
                    <h4 style="margin:0; opacity: 0.8; font-size: 0.7em; letter-spacing: 1px;">APPROVED</h4><h1 style="margin:5px 0;">{len(self.approved_data)}</h1>
                </div>
                <div style="flex: 1; background: #e74c3c; padding: 15px; border-radius: 8px; text-align: center; color: white;">
                    <h4 style="margin:0; opacity: 0.8; font-size: 0.7em; letter-spacing: 1px;">DISCARDED</h4><h1 style="margin:5px 0;">{len(self.rejected_data)}</h1>
                </div>
                <div style="flex: 1; background: #2980b9; padding: 15px; border-radius: 8px; text-align: center; color: white;">
                    <h4 style="margin:0; opacity: 0.8; font-size: 0.7em; letter-spacing: 1px;">SUCCESS RATE</h4><h1 style="margin:5px 0;">{pass_rate:.1f}%</h1>
                </div>
            </div>

            <div style="display: flex; gap: 25px;">
                <div style="flex: 1.8; background: white; padding: 20px; border-radius: 8px; border: 1px solid #e9ecef;">
                    <h4 style="margin-top:0; color: #2c3e50; border-bottom: 2px solid #f1f3f5; padding-bottom: 10px;">📉 FAILURE DISTRIBUTION (Frequency Sorted)</h4>
                    {"".join([f'''
                        <div style="margin-bottom: 12px;">
                            <div style="font-size: 0.85em; margin-bottom: 3px;">
                                <b>{rule}</b> <span style="font-weight: normal; color: #6c757d; margin-left: 8px;">— {self.fail_definitions[rule]}</span>
                            </div>
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <div style="background: #e9ecef; flex: 1; height: 10px; border-radius: 5px; overflow: hidden;">
                                    <div style="background: {'#e74c3c' if 'PARSER' in rule else '#2c3e50'}; width: {(reason_counts[rule] / total_rej * 100):.1f}%; height: 100%;"></div>
                                </div>
                                <span style="font-size: 0.75em; width: 40px; font-weight: bold; color: #2c3e50;">{reason_counts[rule]}</span>
                            </div>
                        </div>
                    ''' for rule in sorted_rules])}
                </div>

                <div style="flex: 1; background: white; padding: 20px; border-radius: 8px; border: 1px solid #e9ecef; display: flex; flex-direction: column; align-items: center;">
                    <h4 style="margin-top:0; color: #2c3e50; width: 100%; border-bottom: 2px solid #f1f3f5; padding-bottom: 10px; text-align: center;">🧩 REJECTIONS BY STRATEGY</h4>
                    <div style="width: 160px; height: 160px; border-radius: 50%; background: conic-gradient({", ".join(pie_parts) or "#dee2e6 0% 100%"}); margin: 20px 0;"></div>
                    <div style="width: 100%; font-size: 0.85em;">
                        {"".join([f'<div style="margin-bottom:6px; display: flex; justify-content: space-between;"><span><span style="color:{colors[i % len(colors)]}; font-size: 1.2em; line-height: 0;">●</span> <b>{s}</b></span><span style="color: #6c757d;">{(count / total_rej * 100):.1f}%</span></div>' for i, (s, count) in enumerate(strat_counts)])}
                    </div>
                </div>
            </div>
        </div>
        """
        display(HTML(dashboard_html))
        self._show_unique_examples()

    def _show_unique_examples(self):
        display(HTML(
            "<h3 style='margin-top: 35px; color: #e74c3c; border-left: 4px solid #e74c3c; padding-left: 10px;'>🔍 REPRESENTATIVE FAILURE EXAMPLES</h3>"))
        shown_rules = set()
        for item in self.rejected_data:
            fails = [k.upper() for k, v in item['raw_rules'].items() if v[0] is False]
            new_fails = [f for f in fails if f not in shown_rules]

            if new_fails:
                primary_fail = new_fails[0]
                shown_rules.add(primary_fail)
                reasons_li = "".join([f"<li>{r}</li>" for r in item['rejection_reasons']])
                df_html = pd.DataFrame(item.get('Relevant table data', [])).to_html(index=False)

                display(HTML(f"""
                <div style="margin-bottom: 25px; border: 1px solid #dee2e6; border-left: 6px solid #e74c3c; background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                        <span style="background: #e74c3c; color: white; padding: 3px 12px; border-radius: 20px; font-size: 0.7em; font-weight: bold; text-transform: uppercase;">{primary_fail}</span>
                        <a href="{item.get('Wiki link', '#')}" target="_blank" style="font-size: 0.75em; color: #2980b9; font-weight: bold; text-decoration: none;">🔗 WIKI SOURCE</a>
                    </div>
                    <p style="margin: 0; font-size: 0.9em; color: #6c757d;"><b>Strategy:</b> <span style="color: #2c3e50;">{item.get('Selected strategy')}</span></p>
                    <p style="margin: 10px 0; font-size: 1.15em; color: #2c3e50; line-height: 1.4;"><b>Question:</b> "{item.get('Question')}"</p>
                    <p style="margin: 10px 0; font-size: 1.15em; color: #2c3e50; line-height: 1.4;"><b>Group Name:</b> "{item.get('Group name')}"</p>
                    <ul style="font-size: 0.9em; color: #c0392b; margin-bottom: 15px; background: #fff5f5; padding: 10px 30px; border-radius: 6px;">{reasons_li}</ul>
                    <details>
                        <summary style="cursor: pointer; color: #2c3e50; font-size: 0.8em; font-weight: bold; text-transform: uppercase; opacity: 0.6;">🔍 INSPECT TABLE DATA</summary>
                        <div style="margin-top: 15px; font-size: 0.85em; border-top: 1px solid #eee; padding-top: 10px;">{df_html}</div>
                    </details>
                </div>
                """))


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

# VALIDATE_PROMPT = """
# ### ROLE
# You are a Data Quality Auditor specializing in structured data and Natural Language Processing.
# Your goal is to ensure that a generated question is logical, non-trivial, and factually supported by the data.
#
# ### CRITERIA FOR VALIDITY
# 1. **ANSWERABILITY**: The question MUST be answerable using ONLY the provided 'Table Data'. No outside knowledge.
# 2. **TRIVIALITY**: The question is INVALID if it is too simple (e.g., asking for a value already mentioned in the question)
# or if the answer is explicitly stated in the 'Group Name' (e.g., asking "How many are there?" when the group is "Top 10 players").
# 3. **IRRATIONALITY**: The question must be grammatically and logically sound. Avoid "hallucinated" verbs that don't match
# the data (e.g., don't say "won" if the data only says "nominated").
# 4. **CONTEXT**: The question MUST use the 'Group Name' to define its scope so it is self-contained for a third party.
# 5. **STRATEGY ALIGNMENT**: The question must follow the specific logic of the assigned strategy.
#
# ### OUTPUT FORMAT
# You must return a valid JSON object only. No prose.
# {
#   "is_valid": boolean,
#   "reason": "Clear explanation of why it passed or failed"
# }
# """
# class QuestionValidator:
#     def __init__(self, input_data, strategy_definitions):
#
#         self.input_data = input_data
#         self.filtered_data = []
#
#         self.model = gptAPI(
#             system_prompt=VALIDATOR_SYSTEM_PROMPT,
#             model="google/gemini-2.5-flash",
#             api_provider="openrouter",
#             max_rpm=500,
#             max_concurrent_requests=100
#         )
#         self.definitions = strategy_definitions['strategies']
#
#         # New Storage attributes
#         self.approved_data = []
#         self.not_approved = []
#
#         self.manual_log = {
#             "✅ Approved": 0,
#             "❌ Missing Question": 0,
#             "❌ Missing Table Data": 0,
#             "❌ Missing Data Columns": 0,
#             "❌ Extra Data Columns": 0,
#             "❌ Missing Data": 0,
#         }
#
#     def manual_filter(self):
#         from collections import defaultdict
#
#         # Track rejections per strategy: {Strategy: {Reason: Count}}
#         strategy_rejections = defaultdict(lambda: defaultdict(int))
#
#         for sample in tqdm_lib.tqdm(self.input_data, desc="Manual Filtering"):
#             strat = sample.get('Selected strategy', 'UNKNOWN')
#             reason = None
#
#             # 1. Question String Check
#             q = str(sample.get('Question', '')).strip().lower()
#             if q in ['null', 'null.', 'none', '']:
#                 reason = '❌ Missing Question'
#
#             # 2. Column logic
#             if not reason:
#                 relevant_data = sample.get('Relevant table data', [])
#                 if not relevant_data:
#                     reason = '❌ Missing Table Data'
#                 else:
#                     entity_col = list(relevant_data[0].keys())[0]
#                     target_cols = [k for k in relevant_data[0].keys() if k != entity_col]
#
#                     if len(target_cols) > 1:
#                         reason = '❌ Extra Data Columns'
#                     elif len(target_cols) < 1 and strat != "COUNT_TOTAL":
#                         reason = '❌ Missing Data Columns'
#                     elif any(all(row.get(col) is None for row in relevant_data) for col in target_cols):
#                         reason = '❌ Missing Data'
#
#             if reason:
#                 self.manual_log[reason] += 1
#                 strategy_rejections[strat][reason] += 1
#                 continue
#
#             self.manual_log['✅ Approved'] += 1
#             self.filtered_data.append(sample)
#
#         # --- PRINT SUMMARY ---
#         print("\n" + "═" * 40)
#         print("📊 MANUAL FILTERING SUMMARY".center(36))
#         print("-" * 40)
#         print(f"{'📊 Total Samples':<28} | {len(self.input_data)}")
#         for key, val in self.manual_log.items():
#             if val > 0: print(f"{key:<28} | {val}")
#
#         pass_rate = 100 * self.manual_log['✅ Approved'] / len(self.input_data)
#         print(f"{'📈 Pass Rate:':<28} | {pass_rate:.1f}%")
#         print("-" * 40)
#
#         # --- REJECTIONS PER STRATEGY TABLE ---
#         if strategy_rejections:
#             print("🚩 REJECTIONS BY STRATEGY".center(36))
#             print("-" * 40)
#             # Sort strategies by total number of rejections
#             sorted_strats = sorted(strategy_rejections.keys(),
#                                    key=lambda s: sum(strategy_rejections[s].values()),
#                                    reverse=True)
#
#             for s in sorted_strats:
#                 total_s_rejections = sum(strategy_rejections[s].values())
#                 print(f"{s:<28} | {total_s_rejections}")
#
#         print("═" * 40 + "\n")
#
#     def _prepare_validation_input(self, sample):
#         strategy = sample.get('Selected strategy')
#         strat_def = self.definitions.get(strategy, {})
#
#         # --- SHUFFLE LOGIC ---
#         # We create a copy so we don't permanently mess up the original sample data
#         table_data = list(sample.get('Relevant table data', []))
#         random.shuffle(table_data)
#         # ---------------------
#
#         return f"""
#         ### DATA TO AUDIT
#         - Strategy Applied: {strategy}
#         - Strategy Definition: {strat_def.get('definition')}
#         - Strategy Requirements: {strat_def.get('requirements')}
#
#         ### TARGET DATA
#         - Table Description: {sample.get('Group name')}
#         - Table Data: {table_data}
#
#         ### QUESTION TO EVALUATE
#         "{sample.get('Question')}"
#         """
#
#     # async def model_filter(self):
#     #     """
#     #     Processes a list of samples, populating approved_data and not_approved.
#     #     """
#     #
#     #     prompts = [self._prepare_validation_input(s) for s in self.filtered_data]
#     #     raw_responses = await self.model.async_run(prompts)
#     #
#     #     for i, response in enumerate(raw_responses):
#     #         sample = self.filtered_data[i]
#     #         try:
#     #             # Clean and parse JSON
#     #             clean_json = response.strip()
#     #             if "```json" in clean_json:
#     #                 clean_json = clean_json.split("```json")[1].split("```")[0].strip()
#     #
#     #             result = json.loads(clean_json)
#     #             is_valid = result.get("is_valid", False)
#     #             reason = result.get("reason", "No reason provided.")
#     #
#     #             if is_valid:
#     #                 self.approved_data.append(sample)
#     #             else:
#     #                 # Store the sample along with the auditor's feedback
#     #                 failed_entry = sample.copy()
#     #                 failed_entry['rejection_reason'] = reason
#     #                 self.not_approved.append(failed_entry)
#     #
#     #         except Exception as e:
#     #             sample['rejection_reason'] = f"Parser Error: {str(e)}"
#     #             self.not_approved.append(sample)
#     #
#     #     self.print_summary()
#     async def model_filter(self):
#         prompts = [self._prepare_validation_input(s) for s in self.filtered_data]
#         raw_responses = await self.model.async_run(prompts)
#
#         for i, response in enumerate(raw_responses):
#             sample = self.filtered_data[i]
#             try:
#                 # Clean and parse JSON
#                 clean_json = response.strip()
#                 if "```json" in clean_json:
#                     clean_json = clean_json.split("```json")[1].split("```")[0].strip()
#
#                 rules_results = json.loads(clean_json)
#
#                 # Identify failed rules
#                 failed_rules = [f"{k}: {v[1]}" for k, v in rules_results.items() if v[0] is False]
#                 is_valid = len(failed_rules) == 0
#
#                 if is_valid:
#                     self.approved_data.append(sample)
#                 else:
#                     failed_entry = sample.copy()
#                     failed_entry['rejection_reasons'] = failed_rules
#                     failed_entry['raw_rules'] = rules_results  # Used for stats
#                     self.not_approved.append(failed_entry)
#
#             except Exception as e:
#                 # Track Parser Errors as a "Rule Fail" for the distribution summary
#                 failed_entry = sample.copy()
#                 failed_entry['rejection_reasons'] = [f"PARSER_ERROR: {str(e)}"]
#                 # We mock the raw_rules structure so the stats function picks it up
#                 failed_entry['raw_rules'] = {"PARSER_ERROR": [False, str(e)]}
#                 self.not_approved.append(failed_entry)
#
#         self.print_summary()
#
#     # def print_summary(self):
#     #     """
#     #     Prints a detailed report of the validation results.
#     #     """
#     #     total_approved = len(self.approved_data)
#     #     total_rejected = len(self.not_approved)
#     #     total = total_approved + total_rejected
#     #     pass_rate = (total_approved / total * 100) if total > 0 else 0
#     #
#     #     print("\n" + "=" * 50)
#     #     print("📊 VALIDATION SUMMARY")
#     #     print("=" * 50)
#     #     print(f"Total Samples:   {total}")
#     #     print(f"✅ Approved:     {len(self.approved_data)}")
#     #     print(f"❌ Rejected:     {len(self.not_approved)}")
#     #     print(f"📈 Pass Rate:    {pass_rate:.1f}%")
#     #     print("-" * 50)
#     #
#     #     # --- ADDED: REJECTION DISTRIBUTION ---
#     #     if total_rejected > 0:
#     #         print("📉 REJECTIONS BY STRATEGY:")
#     #         # Extract strategy names from the not_approved list
#     #         rejected_strats = [item.get('Selected strategy', 'Unknown') for item in self.not_approved]
#     #         rej_counts = Counter(rejected_strats)
#     #
#     #         # Sort by highest number of failures
#     #         for strat, count in rej_counts.most_common():
#     #             print(f"  - {strat:<25}: {count}")
#     #         print("-" * 50)
#     #     # --------------------------------------
#     #
#     #     if self.not_approved:
#     #         print("\n🔍 TOP REJECTION EXAMPLES:")
#     #         # Show up to 5 examples of failures
#     #         for i, item in enumerate(self.not_approved[:5]):
#     #             print(f"\n{i + 1}. Strategy: {item.get('Selected strategy')}")
#     #             print(f"   Question: {item.get('Question')}")
#     #             print(f"   Wikipedia URL: {item.get('Wiki link')}")
#     #             print(f"   Data: {item.get('Group name')}")
#     #             display(pd.DataFrame(item.get('Relevant table data')))
#     #             print(f"   🚩 Reason for rejection: {item.get('rejection_reason')}")
#     #
#     #     print("=" * 50 + "\n")
#     def print_summary(self):
#         total_approved = len(self.approved_data)
#         total_rejected = len(self.not_approved)
#         total = total_approved + total_rejected
#         pass_rate = (total_approved / total * 100) if total > 0 else 0
#
#         # --- HTML HEADER ---
#         display(HTML(f"""
#         <div style="border: 2px solid #2c3e50; border-radius: 10px; padding: 20px; font-family: sans-serif; background-color: #f8f9fa;">
#             <h2 style="margin-top: 0; color: #2c3e50;">📊 MODEL VALIDATION: TOTAL PICTURE</h2>
#             <div style="display: flex; gap: 40px;">
#                 <div style="text-align: center;"><h1 style="color: #27ae60; margin: 0;">{total_approved}</h1><p>✅ APPROVED</p></div>
#                 <div style="text-align: center;"><h1 style="color: #e74c3c; margin: 0;">{total_rejected}</h1><p>❌ REJECTED</p></div>
#                 <div style="text-align: center; border-left: 1px solid #ccc; padding-left: 40px;">
#                     <h1 style="color: #2980b9; margin: 0;">{pass_rate:.1f}%</h1><p>SUCCESS RATE</p>
#                 </div>
#             </div>
#         </div>
#         """))
#
#         # --- 1. DISTRIBUTION OF FAILURES (Including Parser Errors) ---
#         if total_rejected > 0:
#             all_fail_keys = []
#             for item in self.not_approved:
#                 # Collect keys that failed
#                 fails = [k for k, v in item['raw_rules'].items() if v[0] is False]
#                 all_fail_keys.extend(fails)
#
#             reason_counts = Counter(all_fail_keys)
#
#             print("\n📉 FAILURE DISTRIBUTION (Checklist Breakdown):")
#             for reason, count in reason_counts.most_common():
#                 pct = (count / total_rejected) * 100
#                 bar = "█" * int(pct / 5)
#                 # Differentiate Parser Errors visually
#                 label = f" {reason}"
#                 print(f"{label:<22} | {bar} {count} samples ({pct:.1f}%)")
#
#         # --- 2. UNIQUE EXAMPLES PER ERROR TYPE ---
#         if self.not_approved:
#             display(HTML("<h3 style='margin-top: 30px; color: #c0392b;'>🔍 REPRESENTATIVE FAILURES BY RULE</h3>"))
#
#             seen_error_types = set()
#             for item in self.not_approved:
#                 # Find the primary failure in this sample that we haven't shown yet
#                 # We skip showing PARSER_ERROR samples here as requested
#                 current_fails = [k for k, v in item['raw_rules'].items() if v[0] is False and k != "PARSER_ERROR"]
#
#                 # Check if this sample contains an error type we haven't displayed
#                 new_error_found = False
#                 for f in current_fails:
#                     if f not in seen_error_types:
#                         seen_error_types.add(f)
#                         new_error_found = True
#                         break
#
#                 if new_error_found:
#                     reasons_html = "".join(
#                         [f"<li style='color: #c0392b;'><b>{r.split(':')[0]}:</b> {r.split(':')[1]}</li>" for r in
#                          item['rejection_reasons']])
#                     df_html = pd.DataFrame(item.get('Relevant table data')).to_html(index=False)
#
#                     display(HTML(f"""
#                     <div style="margin-bottom: 25px; border-left: 5px solid #e74c3c; padding: 10px; background: #fffcfc; border-bottom: 1px solid #eee;">
#                         <p style="margin: 0; font-size: 0.9em; color: #666;">Example for Rule: <b style="color: #e74c3c;">{list(seen_error_types)[-1]}</b></p>
#                         <p style="margin: 5px 0;"><b>Question:</b> "{item.get('Question')}"</p>
#                         <ul style="margin: 5px 0; font-size: 0.9em;">{reasons_html}</ul>
#                         <details><summary style="cursor: pointer; color: #2980b9; font-size: 0.85em;">View Data Context</summary>
#                         <div style="margin-top: 5px; font-size: 0.85em;">{df_html}</div></details>
#                     </div>
#                     """))
#
#     async def validate(self):
#         self.manual_filter()
#         await self.model_filter()