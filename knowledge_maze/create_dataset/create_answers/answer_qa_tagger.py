import os
import json
import random
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML


class AnswerValidityTagger:
    def __init__(self, generated_data, strategy_definitions, save_file="answer_tagger_progress.json"):
        self.data = generated_data
        # Handle both dict with 'strategies' key or direct dictionary
        self.definitions = strategy_definitions.get("strategies", strategy_definitions) if isinstance(
            strategy_definitions, dict) else strategy_definitions
        self.save_file = save_file

        # Load existing progress if it exists
        self.results = self._load_progress()

        total_in_file = len(self.results)
        print(f"📦 Progress Loaded: {total_in_file} answer samples already tagged in '{save_file}'.")

        self.current_idx = 0
        self.sample_pool = []

        # UI Components
        self.out = widgets.Output()

        # Slighter Progress Bar
        self.progress = widgets.IntProgress(
            min=0,
            layout=widgets.Layout(
                width='98%',
                height='18px',
                border='1px solid #666',
                border_radius='1px',
                margin='0 auto'
            )
        )
        self.progress.style = {'bar_color': '#2980b9'}  # Blue for Answers
        self.pct_label = widgets.Label(value="0%", layout=widgets.Layout(margin='0 0 0 1%'))

        # Strategy Selection UI
        strats = sorted(list(set(s.get('Selected strategy', 'UNKNOWN') for s in self.data)))
        self.strat_dropdown = widgets.Dropdown(
            options=['ALL STRATEGIES'] + strats,
            value='ALL STRATEGIES',
            description='Focus On:',
            layout=widgets.Layout(width='300px')
        )
        self.start_btn = widgets.Button(description="🚀 START", layout=widgets.Layout(width='100px'))
        self.start_btn.on_click(lambda x: self.run(10, self.strat_dropdown.value))

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
        available_samples = [s for s in self.data if s.get('Question') not in tagged_questions]

        # 2. Filter by specific strategy
        if filter_strategy != 'ALL STRATEGIES':
            available_samples = [s for s in available_samples if s.get('Selected strategy') == filter_strategy]

        if not available_samples:
            with self.out:
                clear_output()
                print(f"✨ No untagged samples found for: {filter_strategy}")
                display(widgets.HBox([self.strat_dropdown, self.start_btn]))
            return

        random.shuffle(available_samples)
        self.sample_pool = available_samples[:sample_size]

        self.progress.max = len(self.sample_pool)
        self.current_idx = 0
        self._update_progress_ui()
        self._show_sample()

    def _update_progress_ui(self):
        self.progress.value = self.current_idx
        if self.progress.max > 0:
            pct = (self.current_idx / self.progress.max) * 100
        else:
            pct = 0
        self.pct_label.value = f"Session Progress: {pct:.0f}%"

    def _show_sample(self):
        with self.out:
            clear_output(wait=True)
            if self.current_idx >= len(self.sample_pool):
                self._finish()
                return

            sample = self.sample_pool[self.current_idx]
            strat = sample.get('Selected strategy', 'Unknown')
            strat_def = self.definitions.get(strat, {}).get('definition', 'No definition found.')
            wiki_url = sample.get('Wiki link', '#')

            ans = sample.get('Answer', {})

            # Convert Original and Fake Data to HTML tables
            df_orig = pd.DataFrame(sample.get('Original data', []))
            table_orig_html = df_orig.to_html(classes='data', index=False) if not df_orig.empty else "<i>No Data</i>"

            df_fake = pd.DataFrame(sample.get('Fake data', []))
            table_fake_html = df_fake.to_html(classes='data',
                                              index=False) if not df_fake.empty else "<i>No Fake Data generated.</i>"

            table_style = """
            <style>
                .data { border-collapse: collapse; margin: 0 auto; background-color: white; border-radius: 4px; width: 100%; font-size: 0.9em; }
                .data th, .data td { padding: 6px 10px; border: 1px solid #ddd; text-align: left; }
                .data th { background-color: #f2f2f2; position: sticky; top: 0; z-index: 10; }
                .wiki-link { 
                    text-decoration: none; color: #2980b9; font-weight: bold; font-size: 0.85em;
                    padding: 4px 12px; border: 1px solid rgba(41, 128, 185, 0.4); border-radius: 20px;
                    background: rgba(255,255,255,0.8); transition: 0.3s;
                }
                .wiki-link:hover { background: #2980b9; color: white; }

                /* Custom Scrollbar */
                .scroll-area::-webkit-scrollbar { width: 6px; }
                .scroll-area::-webkit-scrollbar-track { background: transparent; }
                .scroll-area::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.1); border-radius: 10px; }
            </style>
            """

            fixed_height = "550px"  # Slightly taller to accommodate two tables

            layout_html = f"""
            {table_style}
            <div style="display: flex; gap: 20px; font-family: 'Segoe UI', sans-serif; height: {fixed_height};">

                <!-- LEFT PANEL: Question & Answers -->
                <div style="flex: 1; padding: 20px; border: 1px solid rgba(52, 152, 219, 0.3); 
                            border-radius: 12px; background-color: rgba(52, 152, 219, 0.05); 
                            overflow-y: auto;" class="scroll-area">
                    <h4 style="margin:0; color: #2c3e50; border-bottom: 1px solid rgba(0,0,0,0.1); padding-bottom: 8px;">
                        📋 Sample {self.current_idx + 1} / {len(self.sample_pool)}
                    </h4>
                    <p style="margin: 10px 0;"><b>Group:</b> {sample.get('Group name', 'N/A')}</p>
                    <p style="margin: 5px 0;"><b>Strategy:</b> <span style="color: #2980b9; font-weight: bold;">{strat}</span></p>
                    <p style="font-size: 0.85em; color: #666;"><i>{strat_def}</i></p>

                    <div style="margin-top: 15px; padding: 15px; background: rgba(255, 255, 255, 0.7); 
                                border-left: 5px solid #3498db; border-radius: 4px;">
                        <b style="color: #2c3e50; font-size: 0.8em; letter-spacing: 1px;">QUESTION:</b><br>
                        <span style="font-size: 1.15em; color: #111;">{sample.get('Question', 'N/A')}</span>
                    </div>

                    <!-- NEW: ANSWER DISPLAY -->
                    <div style="margin-top: 15px; padding: 15px; background: rgba(243, 156, 18, 0.1); 
                                border-left: 5px solid #f39c12; border-radius: 4px;">
                        <b style="color: #d35400; font-size: 0.8em; letter-spacing: 1px;">EVALUATE ANSWERS:</b><br>
                        <div style="margin-top: 10px; font-size: 1.1em;">
                            <div style="margin-bottom: 6px;"><span style="color: #27ae60; font-weight: bold;">🧠 Memory (True):</span> {ans.get('Memory', 'N/A')}</div>
                            <div style="margin-bottom: 6px;"><span style="color: #2980b9; font-weight: bold;">📢 Counter #1 (Claim):</span> {ans.get('Claim', 'N/A')}</div>
                            <div><span style="color: #c0392b; font-weight: bold;">📁 Counter #2 (Evidence):</span> {ans.get('Evidence', 'N/A')}</div>
                        </div>
                    </div>
                </div>

                <!-- RIGHT PANEL: Tables Side-by-Side (Stacked) -->
                <div style="flex: 1.5; padding: 20px; border: 1px solid rgba(46, 204, 113, 0.3); 
                            border-radius: 12px; background-color: rgba(46, 204, 113, 0.05); 
                            display: flex; flex-direction: column;">

                    <div style="display: flex; justify-content: space-between; align-items: center; width: 100%; margin-bottom: 15px;">
                        <h4 style="margin: 0; color: #27ae60;">📊 Data Context</h4>
                        <a href="{wiki_url}" target="_blank" class="wiki-link">🌐 View Wikipedia</a>
                    </div>

                    <div style="flex-grow: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 20px;" class="scroll-area">
                        <div>
                            <h5 style="margin: 0 0 5px 0; color: #2c3e50;">🔹 Original Data</h5>
                            {table_orig_html}
                        </div>
                        <div>
                            <h5 style="margin: 0 0 5px 0; color: #c0392b;">🔸 Fake Evidence Data (Fake)</h5>
                            {table_fake_html}
                        </div>
                    </div>
                </div>
            </div>
            """
            display(HTML(layout_html))

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

            button_box = widgets.HBox(
                [btn_good, btn_bad, btn_quit],
                layout=widgets.Layout(margin='20px 0', justify_content='center')
            )

            progress_box = widgets.VBox(
                [self.pct_label, self.progress],
                layout=widgets.Layout(
                    align_items='center',
                    background_color='#f0f0f0',
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
            "group": sample.get('Group name', 'N/A'),
            "strategy": sample.get('Selected strategy', 'N/A'),
            "question": sample.get('Question', 'N/A'),
            "score": score
        })
        self._save_progress()
        self.current_idx += 1
        self._update_progress_ui()
        self._show_sample()

    def _finish(self):
        with self.out:
            clear_output()
            print(f"💾 Progress saved to {self.save_file}")

    def plot_results(self):
        df_tagged = pd.DataFrame(self.results)
        df_all = pd.DataFrame(self.data)

        if df_tagged.empty:
            print("No tags to plot yet.")
            return

        global_counts = df_all['Selected strategy'].value_counts(normalize=True)

        strat_stats = df_tagged.groupby('strategy')['score'].agg(['count', 'sum', 'mean'])
        strat_stats['mean_pct'] = strat_stats['mean'] * 100

        weighted_pass_rate = 0
        for strat, weight in global_counts.items():
            if strat in strat_stats.index:
                success_rate = strat_stats.loc[strat, 'mean']
                weighted_pass_rate += (success_rate * weight)

        weighted_fail_rate = 1 - weighted_pass_rate

        fig = plt.figure(figsize=(20, 7), facecolor='white')
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.8])

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])

        ax1.pie([df_tagged['score'].sum(), len(df_tagged) - df_tagged['score'].sum()],
                labels=['Pass', 'Fail'], autopct='%1.1f%%',
                colors=['#2ecc71', '#e74c3c'], startangle=140,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        ax1.set_title("Answer Validity Rate\n(Tagged Only)", fontsize=11, fontweight='bold', pad=10)

        ax2.pie([weighted_pass_rate, weighted_fail_rate],
                labels=['Pass', 'Fail'], autopct='%1.1f%%',
                colors=['#27ae60', '#c0392b'], startangle=140,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        ax2.set_title("Weighted Answer Validity Rate\n(Global Distribution)", fontsize=11, fontweight='bold', pad=10)

        colors = ['#2ecc71' if x >= 70 else '#f1c40f' if x >= 40 else '#e74c3c' for x in strat_stats['mean_pct']]
        bars = ax3.bar(strat_stats.index, strat_stats['mean_pct'], color=colors, edgecolor='black', alpha=0.8)

        ax3.set_title("Validity % per Strategy", fontsize=14, fontweight='bold')
        ax3.set_ylabel("Validity Rate (%)")
        ax3.set_ylim(0, 130)
        plt.xticks(rotation=45, ha='right')

        for i, bar in enumerate(bars):
            height = bar.get_height()
            total_checked = int(strat_stats.iloc[i]['count'])
            success_count = int(strat_stats.iloc[i]['sum'])
            label_text = f"{height:.0f}%\n({success_count}/{total_checked})"
            ax3.text(bar.get_x() + bar.get_width() / 2., height + 2,
                     label_text, ha='center', va='bottom',
                     fontweight='bold', fontsize=10, color='#2c3e50')

        plt.subplots_adjust(wspace=0.1)

        n_samples = len(df_tagged)
        fig.suptitle(f"Answer Validity Audit Report: {n_samples} Samples Tagged",
                     fontsize=18, fontweight='bold', y=1.05, color='#1e293b')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def display(self):
        with self.out:
            clear_output()
            display(HTML("<b style='color: #2c3e50;'>Pick Strategy to Evaluate Answers For:</b>"))
            display(widgets.HBox([self.strat_dropdown, self.start_btn]))

            if self.results:
                print(f"📦 {len(self.results)} answer samples already evaluated in history.")

        display(self.out)