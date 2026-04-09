import os
import json
import random
import re
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML


class ParagraphAuditTagger:
    def __init__(self, dataset, paragraph_map, save_file="paragraph_audit_progress.json"):
        self.data = dataset
        self.paragraphs = paragraph_map
        self.save_file = save_file
        self.results = self._load_progress()

        self.current_idx = 0
        self.sample_pool = []
        self.out = widgets.Output()

        # Progress Bar UI
        self.progress_bar = widgets.IntProgress(value=0, min=0, max=10, description='Progress:', bar_style='info')
        self.progress_label = widgets.HTML(value="")
        self.progress_box = widgets.HBox([self.progress_bar, self.progress_label],
                                         layout=widgets.Layout(align_items='center', margin='10px 0'))

        # Strategy selection
        strats = sorted(list(set(s.get('Selected strategy', 'UNKNOWN') for s in self.data)))
        self.strat_dropdown = widgets.Dropdown(options=['ALL STRATEGIES'] + strats, value='ALL STRATEGIES',
                                               description='Strategy:')
        self.start_btn = widgets.Button(description="🚀 START AUDIT", button_style='primary')
        self.start_btn.on_click(lambda x: self.run(10, self.strat_dropdown.value))

    def _load_progress(self):
        if os.path.exists(self.save_file):
            with open(self.save_file, 'r') as f: return json.load(f)
        return []

    def _save_progress(self):
        with open(self.save_file, 'w') as f: json.dump(self.results, f, indent=4)

    def _highlight_data(self, text, fake_data_list):
        """Highlights data points by allowing flexible matching for punctuation/casing."""
        import re

        # 1. Collect unique terms from the fake data table
        terms = set()
        for row in fake_data_list:
            for val in row.values():
                s = str(val).strip()
                if len(s) > 1:
                    terms.add(s)

        if not terms: return text

        # Sort by length (longest first) to avoid partial matches of longer phrases
        sorted_terms = sorted(list(terms), key=len, reverse=True)

        highlighted_text = text
        for term in sorted_terms:
            # This regex allows for any non-word characters (punctuation, spaces)
            # to exist between the words of your search term.
            # Example: "New York" will match "New-York", "new york", or "New...York"
            parts = [re.escape(part) for part in term.split()]
            # Join parts with a 'flexible' separator
            flexible_pattern = r"[\s\W_]+".join(parts)

            # Use word boundaries \b to ensure we don't catch sub-words
            pattern = re.compile(rf"\b({flexible_pattern})\b", re.IGNORECASE)

            # Wrap in the blue highlight class
            highlighted_text = pattern.sub(r'<span class="hl hl-blue">\1</span>', highlighted_text)

        return highlighted_text

    def _extract_claims(self, p_high, p_med, p_low):
        """Finds the unique claim section. Returns full High para + just the Med/Low replacement texts."""
        prefix_len, suffix_len = 0, 0
        min_len = min(len(p_high), len(p_med), len(p_low))

        # Find common prefix
        for i in range(min_len):
            if p_high[i] == p_med[i] == p_low[i]:
                prefix_len += 1
            else:
                break

        # Find common suffix
        for i in range(1, min_len - prefix_len + 1):
            if p_high[-i] == p_med[-i] == p_low[-i]:
                suffix_len += 1
            else:
                break

        # Extract just the claim components
        c_high = p_high[prefix_len: len(p_high) - suffix_len] if suffix_len else p_high[prefix_len:]
        c_med = p_med[prefix_len: len(p_med) - suffix_len] if suffix_len else p_med[prefix_len:]
        c_low = p_low[prefix_len: len(p_low) - suffix_len] if suffix_len else p_low[prefix_len:]

        prefix = p_high[:prefix_len]
        suffix = p_high[len(p_high) - suffix_len:] if suffix_len else ""

        # Wrap the isolated claims
        hl_high = f"<span class='hl hl-green'>{c_high}</span>"
        hl_med = f"<span class='hl hl-yellow'>{c_med}</span>"
        hl_low = f"<span class='hl hl-orange'>{c_low}</span>"

        full_high = f"{prefix}{hl_high}{suffix}"
        return full_high, hl_med, hl_low

    def _show_sample(self):
        with self.out:
            clear_output(wait=True)
            if self.current_idx >= len(self.sample_pool):
                print("🏁 Audit session complete! Run `plot_results()` to view analytics.")
                return

            sample = self.sample_pool[self.current_idx]
            ans = sample.get('Answer', {})
            wiki_url = sample.get('Wiki link', '#')

            # Paragraphs & Highlighting
            evidence_raw = self.paragraphs.get(sample.get("Evidence paragraph", ""), "Evidence missing.")
            claims_uuids = sample.get("Claim paragraphs", {})
            full_high, claim_med, claim_low = self._extract_claims(
                self.paragraphs.get(claims_uuids.get("High certainty", ""), ""),
                self.paragraphs.get(claims_uuids.get("Medium certainty", ""), ""),
                self.paragraphs.get(claims_uuids.get("Low certainty", ""), "")
            )
            evidence_html = self._highlight_data(evidence_raw, sample.get("Fake data", []))

            # HTML Tables
            table_orig = pd.DataFrame(sample.get("Original data", [])).to_html(classes='data', index=False)
            table_fake = pd.DataFrame(sample.get("Fake data", [])).to_html(classes='data', index=False)

            style = """
            <style>
                .audit-container { display: flex; flex-direction: column; gap: 15px; font-family: 'Segoe UI', sans-serif; background: #f4f7f6; padding: 20px; border-radius: 12px; }
                .panel { background: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
                .data { border-collapse: collapse; width: 100%; font-size: 0.85em; margin-bottom: 0; }
                .data th, .data td { border: 1px solid #ddd; padding: 4px 8px; text-align: left; }
                .data th { background: #f8f9fa; color: #333; position: sticky; top: 0; }
                .para-box { font-size: 0.95em; line-height: 1.6; color: #2c3e50; padding: 12px; border-radius: 6px; background: #fff; border: 1px solid #efefef; }
                .label-tag { font-size: 0.75em; text-transform: uppercase; font-weight: bold; padding: 3px 8px; border-radius: 4px; display: inline-block; margin-bottom: 8px; }
                .wiki-link { text-decoration: none; color: #fff; background: #2980b9; font-weight: bold; font-size: 0.8em; padding: 4px 10px; border-radius: 4px; }
                /* Updated Highlight Styles: Subtle & No-Overlap */
                .hl { 
                    padding: 1px 3px; 
                    border-radius: 2px; 
                    font-weight: 500; /* Slightly less bold */
                    display: inline;  /* Changed from inline-block to prevent line-height issues */
                    line-height: 1;
                }
                .hl-blue { background-color: rgba(227, 242, 253, 0.8); color: #0D47A1; box-shadow: inset 0 -1px 0 #1E88E5; }
                .hl-green { background-color: rgba(232, 245, 233, 0.8); color: #1B5E20; box-shadow: inset 0 -1px 0 #43A047; }
                .hl-yellow { background-color: rgba(255, 249, 196, 0.8); color: #F57F17; box-shadow: inset 0 -1px 0 #FBC02D; }
                .hl-orange { background-color: rgba(255, 224, 178, 0.8); color: #E65100; box-shadow: inset 0 -1px 0 #FB8C00; }
            </style>
            """

            html_layout = f"""
            {style}
            <div class="audit-container">
                <div class="panel" style="border-top: 5px solid #2C3E50;">
                    <div style="font-size: 0.9em; color: #7f8c8d; text-transform: uppercase; font-weight: bold;">{sample['Group name']}</div>
                    <div style="font-size: 1.3em; font-weight: bold; color: #2C3E50; margin-top: 5px;">Q: {sample['Question']}</div>
                </div>

                <div style="display: flex; gap: 15px;">
                    <div class="panel" style="flex: 1; border-left: 4px solid #27AE60; display: flex; flex-direction: column; justify-content: center;">
                        <span class="label-tag" style="background: #e8f5e9; color: #2e7d32;">True World Answer</span>
                        <div style="font-size: 1.1em; margin: 10px 0;">🟢 <b>Memory:</b> {ans.get('Memory', 'N/A')}</div>
                        <div><a href="{wiki_url}" target="_blank" class="wiki-link">View Wikipedia Article</a></div>
                    </div>
                    <div class="panel" style="flex: 2; border-left: 4px solid #27AE60;">
                        <span class="label-tag" style="background: #e8f5e9; color: #2e7d32;">True Data Table</span>
                        <div style="max-height: 100px; overflow-y: auto; border: 1px solid #ddd;">{table_orig}</div>
                    </div>
                </div>

                <div style="display: flex; gap: 15px;">

                    <div class="panel" style="flex: 1; border-top: 4px solid #E74C3C;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span class="label-tag" style="background: #ffebee; color: #c62828;">Evidence Context</span>
                            <span style="font-size: 0.85em;">🔴 <b>Target Ans:</b> {ans.get('Evidence', 'N/A')}</span>
                        </div>
                        <div style="max-height: 400px; overflow-y: auto; margin: 10px 0; border: 1px solid #ddd;">{table_fake}</div>
                        <div class="para-box" style="background: #fdfdfd;">{evidence_html}</div>
                    </div>

                    <div class="panel" style="flex: 1; border-top: 4px solid #F39C12;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span class="label-tag" style="background: #fff3e0; color: #e65100;">Claim Context</span>
                            <span style="font-size: 0.85em;">🔵 <b>Target Ans:</b> {ans.get('Claim', 'N/A')}</span>
                        </div>
                        <div class="para-box" style="margin: 10px 0;">
                            <b>High Confidence Paragraph:</b><br/>{full_high}
                        </div>
                        <div class="para-box" style="margin-bottom: 8px;">
                            <b>Medium Certainty Replacement:</b><br/>{claim_med}
                        </div>
                        <div class="para-box">
                            <b>Low Certainty Replacement:</b><br/>{claim_low}
                        </div>
                    </div>
                </div>
            </div>
            """
            display(HTML(html_layout))

            # Action Buttons Layout
            btn_layout = widgets.Layout(flex='1 1 auto', width='auto', margin='0 4px', height='45px')

            b_good = widgets.Button(description="✅ ALL GOOD", style={'button_color': '#2ecc71'}, layout=btn_layout)
            b_bad_q = widgets.Button(description="❓ BAD QUESTION", style={'button_color': '#e74c3c'}, layout=btn_layout)
            b_bad_a = widgets.Button(description="❌ BAD ANSWERS", style={'button_color': '#e67e22'}, layout=btn_layout)
            b_bad_c = widgets.Button(description="📝 BAD CLAIM PARA", style={'button_color': '#9b59b6'},
                                     layout=btn_layout)
            b_bad_e = widgets.Button(description="📄 BAD EVID PARA",
                                     style={'button_color': '#34495e', 'text_color': 'white'}, layout=btn_layout)

            b_good.on_click(lambda x: self._handle_click("success"))
            b_bad_q.on_click(lambda x: self._handle_click("bad_q"))
            b_bad_a.on_click(lambda x: self._handle_click("bad_ans"))
            b_bad_c.on_click(lambda x: self._handle_click("bad_claim"))
            b_bad_e.on_click(lambda x: self._handle_click("bad_evid"))

            display(widgets.HBox([b_good, b_bad_q, b_bad_a, b_bad_c, b_bad_e],
                                 layout=widgets.Layout(margin='20px 0', width='100%')))

    def _handle_click(self, status):
        sample = self.sample_pool[self.current_idx]
        self.results.append({
            "uuid": sample.get("uuid"),
            "strategy": sample.get('Selected strategy', 'Unknown'),
            "status": status
        })
        self._save_progress()
        self.current_idx += 1

        # Update progress bar
        self.progress_bar.value = self.current_idx
        self.progress_label.value = f"<b>{self.current_idx} / {self.progress_bar.max}</b>"

        self._show_sample()

    def run(self, sample_size=10, filter_strategy='ALL STRATEGIES'):
        # Filter pool
        tagged_uuids = {res['uuid'] for res in self.results}
        available = [s for s in self.data if s.get('uuid') not in tagged_uuids]
        if filter_strategy != 'ALL STRATEGIES':
            available = [s for s in available if s.get('Selected strategy') == filter_strategy]

        random.shuffle(available)
        self.sample_pool = available[:sample_size]
        self.current_idx = 0

        self.progress_bar.max = len(self.sample_pool)
        self.progress_bar.value = 0
        self.progress_label.value = f"<b>0 / {len(self.sample_pool)}</b>"

        with self.out:
            clear_output()
            display(self.progress_box)

        self._show_sample()

    def plot_results(self):
        """Generates plots breaking down the audit results and error types."""
        if not self.results:
            print("No data tagged yet.")
            return

        df = pd.DataFrame(self.results)

        fig = plt.figure(figsize=(18, 5), facecolor='white')
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1.5])

        # 1. Overall Pass/Fail Pie Chart
        ax1 = fig.add_subplot(gs[0, 0])
        total = len(df)
        successes = len(df[df['status'] == 'success'])
        fails = total - successes

        ax1.pie([successes, fails], labels=['Pass', 'Fail'], autopct='%1.1f%%',
                colors=['#2ecc71', '#e74c3c'], startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        ax1.set_title(f"Overall Pass Rate\n(Total: {total} Samples)", fontweight='bold')

        # 2. Breakdown of Failures (Bar Chart)
        ax2 = fig.add_subplot(gs[0, 1])
        fail_df = df[df['status'] != 'success']
        if not fail_df.empty:
            fail_counts = fail_df['status'].value_counts()
            mapping = {"bad_q": "Bad Question", "bad_ans": "Bad Answers",
                       "bad_claim": "Bad Claim Para", "bad_evid": "Bad Evid Para"}
            fail_counts.index = fail_counts.index.map(mapping)

            bars = ax2.bar(fail_counts.index, fail_counts.values, color='#e74c3c', edgecolor='black')
            ax2.set_title("Breakdown of Failures", fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
            for bar in bars:
                ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.1,
                         f'{int(bar.get_height())}', ha='center', va='bottom', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, "No Failures Recorded", ha='center', va='center')
            ax2.axis('off')

        # 3. Success Rate by Strategy
        ax3 = fig.add_subplot(gs[0, 2])
        strat_stats = df.groupby('strategy').apply(lambda x: (x['status'] == 'success').mean() * 100)

        colors = ['#2ecc71' if x >= 80 else '#f1c40f' if x >= 50 else '#e74c3c' for x in strat_stats]
        bars3 = ax3.bar(strat_stats.index, strat_stats.values, color=colors, edgecolor='black')

        ax3.set_title("Success Rate by Strategy", fontweight='bold')
        ax3.set_ylabel("Pass Rate (%)")
        ax3.set_ylim(0, 115)
        plt.xticks(rotation=45, ha='right')

        for bar in bars3:
            ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 2,
                     f'{int(bar.get_height())}%', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.show()

    def display(self):
        display(self.out)
        with self.out:
            clear_output()
            display(widgets.HBox([self.strat_dropdown, self.start_btn]))