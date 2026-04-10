import re
import os
import json
import random
import pandas as pd
import ipywidgets as widgets
import matplotlib.pyplot as plt
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

        # Progress Bar UI - Track overall progress natively
        self.progress_bar = widgets.IntProgress(value=0, min=0, max=10, description='Progress:', bar_style='info')
        self.progress_label = widgets.HTML(value="")
        self.progress_box = widgets.HBox([self.progress_bar, self.progress_label],
                                         layout=widgets.Layout(align_items='center', margin='10px 0'))

        # Strategy selection
        strats = sorted(list(set(s.get('Selected strategy', 'UNKNOWN') for s in self.data)))
        self.strat_dropdown = widgets.Dropdown(options=['ALL STRATEGIES'] + strats, value='ALL STRATEGIES',
                                               description='Strategy:')
        self.start_btn = widgets.Button(description="🚀 START AUDIT", button_style='primary')
        self.start_btn.on_click(lambda x: self.run(self.strat_dropdown.value))

    def _load_progress(self):
        if os.path.exists(self.save_file):
            with open(self.save_file, 'r') as f: return json.load(f)
        return []

    def _save_progress(self):
        with open(self.save_file, 'w') as f: json.dump(self.results, f, indent=4)

    def _prep_table(self, data_list):
        """Adds ordinal index and standardizes table output."""
        if not data_list: return ""
        df = pd.DataFrame(data_list)
        df.insert(0, '#', range(1, len(df) + 1))
        return df.to_html(classes='data', index=False)

    def _highlight_data(self, text, fake_data_list):
        """Highlights data points by ignoring all dashes, punctuation, and casing."""
        terms = set()
        for row in fake_data_list:
            for val in row.values():
                s = str(val).strip()
                if len(s) > 1: terms.add(s)

        if not terms: return text
        sorted_terms = sorted(list(terms), key=len, reverse=True)
        highlighted_text = text

        for term in sorted_terms:
            # Split the term by ANY non-word character (spaces, dashes, hyphens, underscores)
            parts = [re.escape(part) for part in re.split(r'[\s\W_]+', term) if part]
            if not parts: continue

            # Rejoin with a flexible separator that matches any punctuation/space
            flexible_pattern = r"[\s\W_]+".join(parts)
            pattern = re.compile(rf"\b({flexible_pattern})\b", re.IGNORECASE)
            highlighted_text = pattern.sub(r'<span class="hl hl-blue">\1</span>', highlighted_text)

        return highlighted_text

    def _extract_claims(self, p_high, p_med, p_low):
        prefix_len, suffix_len = 0, 0
        min_len = min(len(p_high), len(p_med), len(p_low))

        for i in range(min_len):
            if p_high[i] == p_med[i] == p_low[i]:
                prefix_len += 1
            else:
                break

        for i in range(1, min_len - prefix_len + 1):
            if p_high[-i] == p_med[-i] == p_low[-i]:
                suffix_len += 1
            else:
                break

        c_high = p_high[prefix_len: len(p_high) - suffix_len] if suffix_len else p_high[prefix_len:]
        c_med = p_med[prefix_len: len(p_med) - suffix_len] if suffix_len else p_med[prefix_len:]
        c_low = p_low[prefix_len: len(p_low) - suffix_len] if suffix_len else p_low[prefix_len:]

        prefix = p_high[:prefix_len]
        suffix = p_high[len(p_high) - suffix_len:] if suffix_len else ""

        hl_high = f"<span class='hl hl-green'>{c_high}</span>"
        hl_med = f"<span class='hl hl-yellow'>{c_med}</span>"
        hl_low = f"<span class='hl hl-orange'>{c_low}</span>"

        return f"{prefix}{hl_high}{suffix}", hl_med, hl_low

    def _show_sample(self):
        with self.out:
            clear_output(wait=True)
            if self.current_idx >= len(self.sample_pool):
                print("🏁 Audit session complete! Run `plot_results()` to view analytics.")
                return

            sample = self.sample_pool[self.current_idx]
            ans = sample.get('Answer', {})
            wiki_url = sample.get('Wiki link', '#')

            evidence_raw = self.paragraphs.get(sample.get("Evidence paragraph", ""), "Evidence missing.")
            claims_uuids = sample.get("Claim paragraphs", {})
            full_high, claim_med, claim_low = self._extract_claims(
                self.paragraphs.get(claims_uuids.get("High certainty", ""), ""),
                self.paragraphs.get(claims_uuids.get("Medium certainty", ""), ""),
                self.paragraphs.get(claims_uuids.get("Low certainty", ""), "")
            )
            evidence_html = self._highlight_data(evidence_raw, sample.get("Fake data", []))

            table_orig = self._prep_table(sample.get("Original data", []))
            table_fake = self._prep_table(sample.get("Fake data", []))

            style = """
            <style>
                .audit-container { font-family: 'Segoe UI', sans-serif; background: #f4f7f6; padding: 20px; border-radius: 12px; }
                .grid-container { display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: auto auto; gap: 15px; }
                .panel { background: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); display: flex; flex-direction: column; }

                /* Strict 12-Row Table Containers */
                .true-table-container { height: 140px; overflow-y: auto; border: 1px solid #ddd; background: #fff; margin-top: 10px; }
                .fake-table-container { height: 250px; overflow-y: auto; border: 1px solid #ddd; background: #fff; margin-top: 10px; }
                .data { border-collapse: collapse; width: 100%; font-size: 0.85em; margin: 0; }
                .data th, .data td { border-bottom: 1px solid #ddd; border-right: 1px solid #eee; padding: 6px 8px; text-align: left !important; }
                .data th { background: #f8f9fa; color: #333; position: sticky; top: 0; z-index: 1; }
                .data td:first-child, .data th:first-child { width: 30px; font-weight: bold; background: #fafafa; color: #7f8c8d; text-align: center !important; }

                .para-box { font-size: 0.95em; line-height: 1.6; color: #2c3e50; padding: 12px; border-radius: 6px; background: #fff; border: 1px solid #efefef; margin-top: 10px; }
                .label-tag { font-size: 0.75em; text-transform: uppercase; font-weight: bold; padding: 3px 8px; border-radius: 4px; display: inline-block; margin-bottom: 5px; }
                .wiki-link { text-decoration: none; color: #2980b9;; background: #ffffff; border: 1px solid #2980b9; font-weight: bold; font-size: 0.8em; padding: 4px 10px; border-radius: 4px; display: inline-block; }

                .hl { padding: 1px 3px; border-radius: 2px; font-weight: 500; display: inline; line-height: 1; }
                .hl-blue { background-color: rgba(227, 242, 253, 0.8); color: #0D47A1; box-shadow: inset 0 -1px 0 #1E88E5; }
                .hl-green { background-color: rgba(232, 245, 233, 0.8); color: #1B5E20; box-shadow: inset 0 -1px 0 #43A047; }
                .hl-yellow { background-color: rgba(255, 249, 196, 0.8); color: #F57F17; box-shadow: inset 0 -1px 0 #FBC02D; }
                .hl-orange { background-color: rgba(255, 224, 178, 0.8); color: #E65100; box-shadow: inset 0 -1px 0 #FB8C00; }
                .para-evidence { height: 290px; overflow-y: auto; }
                .claim-container { 
                    height: 600px; /* Adjust this to match the height of your Table + Evidence block */
                    overflow-y: auto; 
                    display: flex; 
                    flex-direction: column; 
                    gap: 8px;
                    padding-right: 5px; /* Space for the scrollbar */
                }
                /* Control the Metadata Panel height here */
                .metadata-height { 
                    height: 180px; 
                    position: relative; 
                }
                
                .strategy-tag {
                    position: absolute;
                    top: 15px;
                    right: 15px;
                    background: #2c3e50;
                    color: white;
                    padding: 2px 10px;
                    border-radius: 20px;
                    font-size: 0.75em;
                    font-weight: bold;
                    letter-spacing: 0.5px;
                }
                
                .answer-grid {
                    display: grid;
                    grid-template-columns: 1fr 1fr 1fr;
                    gap: 10px;
                    margin-top: auto;
                }
                .ans-box {
                padding: 8px;
                border-radius: 6px;
                border: 1px solid #eee;
                font-size: 0.85em;
                text-align: center;
            }
            </style>
            """

            html_layout = f"""
            {style}
            <div class="audit-container">
                <div class="grid-container">
                    <div class="panel metadata-height" style="border-top: 4px solid #2C3E50;">
                        <div class="strategy-tag">
                            {sample.get('Selected strategy', 'Unknown')}
                        </div>
                    
                        <span class="label-tag" style="background: #e3f2fd; color: #1565c0; display: inline-block; width: fit-content;">Metadata</span>
                        <div style="font-size: 0.95em; color: #7f8c8d;">
                            {sample['Group name']}
                        </div>
                        
                        <div style="font-size: 1.1em; font-weight: bold; color: #2C3E50; margin: 8px 0; overflow-y: auto;">
                            Q: {sample['Question']}
                        </div>
                        
                        <div class="answer-grid">
                            <div class="ans-box" style="background: #e8f5e9; border-color: #27AE60;">
                                <div style="font-size: 0.8em; color: #2e7d32; font-weight: bold;">MEMORY (TRUE)</div>
                                <b>{ans.get('Memory', 'N/A')}</b>
                            </div>
                            <div class="ans-box" style="background: #fff3e0; border-color: #F39C12;">
                                <div style="font-size: 0.8em; color: #e65100; font-weight: bold;">TARGET CLAIM</div>
                                <b>{ans.get('Claim', 'N/A')}</b>
                            </div>
                            <div class="ans-box" style="background: #ffebee; border-color: #E74C3C;">
                                <div style="font-size: 0.8em; color: #c62828; font-weight: bold;">TARGET EVIDENCE</div>
                                <b>{ans.get('Evidence', 'N/A')}</b>
                            </div>
                        </div>
                    </div>

                    <div class="panel" style="border-top: 4px solid #27AE60;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span class="label-tag" style="background: #e8f5e9; color: #2e7d32;">True World Data</span>
                            <a href="{wiki_url}" target="_blank" class="wiki-link">🌐 View Wikipedia</a>
                        </div>
                        <div class="true-table-container">{table_orig}</div>
                    </div>

                    <div class="panel" style="border-top: 4px solid #E74C3C;">
                        <span class="label-tag" style="background: #ffebee; color: #c62828;">Fake Evidence Context</span>
                        <div class="fake-table-container">{table_fake}</div>
                        <div class="para-box para-evidence" style="background: #fdfdfd;">{evidence_html}</div>
                    </div>

                    <div class="panel" style="border-top: 4px solid #F39C12;">
                        <span class="label-tag" style="background: #fff3e0; color: #e65100;">Claim Context</span>
                        <div class="claim-container">
                            <div class="para-box">
                                <b style="color: #27AE60;">[High Confidence]</b><br/>{full_high}
                            </div>
                            <div class="para-box">
                                <b style="color: #F39C12;">[Medium Certainty Replacement]</b><br/>{claim_med}
                            </div>
                            <div class="para-box">
                                <b style="color: #E67E22;">[Low Certainty Replacement]</b><br/>{claim_low}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """
            display(HTML(html_layout))

            btn_layout = widgets.Layout(flex='1 1 auto', width='auto', margin='0 4px', height='45px')
            b_good = widgets.Button(description="✅ ALL GOOD", style={'button_color': '#2ecc71'}, layout=btn_layout)
            b_bad_q = widgets.Button(description="❓ BAD QUESTION", style={'button_color': '#e74c3c'}, layout=btn_layout)
            b_bad_a = widgets.Button(description="❌ BAD ANSWERS", style={'button_color': '#e67e22'}, layout=btn_layout)
            b_bad_c = widgets.Button(description="📝 BAD CLAIM PARAGRAPH", style={'button_color': '#9b59b6'},
                                     layout=btn_layout)
            b_bad_e = widgets.Button(description="📄 BAD EVIDENCE PARAGRAPH",
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
        self.progress_bar.value += 1
        self.progress_label.value = f"<b>{self.progress_bar.value} / {self.progress_bar.max} Tagged</b>"

        self._show_sample()

    def run(self, filter_strategy='ALL STRATEGIES'):
        tagged_uuids = {res['uuid'] for res in self.results}

        # Filter available untagged samples
        self.sample_pool = [s for s in self.data if s.get('uuid') not in tagged_uuids]
        if filter_strategy != 'ALL STRATEGIES':
            self.sample_pool = [s for s in self.sample_pool if s.get('Selected strategy') == filter_strategy]

        random.shuffle(self.sample_pool)
        self.current_idx = 0

        # Set accurate progress based on total corpus vs tagged
        total_in_scope = len(self.data) if filter_strategy == 'ALL STRATEGIES' else len(
            [s for s in self.data if s.get('Selected strategy') == filter_strategy])
        already_tagged = total_in_scope - len(self.sample_pool)

        self.progress_bar.max = total_in_scope
        self.progress_bar.value = already_tagged
        self.progress_label.value = f"<b>{already_tagged} / {total_in_scope} Tagged</b>"

        with self.out:
            clear_output()
            display(self.progress_box)

        self._show_sample()

    def plot_results(self):
        """Generates a Pie chart (normalized to dataset strategy dist) and a Bar chart."""
        if not self.results:
            print("No data tagged yet.")
            return

        df = pd.DataFrame(self.results)

        # 1. Normalize Pie Chart based on True Dataset Strategy Distribution
        strat_counts = pd.Series([s.get('Selected strategy', 'Unknown') for s in self.data]).value_counts(
            normalize=True)
        status_distributions = df.groupby('strategy')['status'].value_counts(normalize=True).unstack(fill_value=0)
        status_distributions = status_distributions.reindex(strat_counts.index, fill_value=0)

        # Multiply distributions by the dataset weights to get true global error rates
        weighted_statuses = (status_distributions.T * strat_counts).sum(axis=1)

        labels_map = {
            "success": "Pass", "bad_q": "Bad Question",
            "bad_ans": "Bad Answers", "bad_claim": "Bad Claim Para", "bad_evid": "Bad Evid Para"
        }
        colors_map = {
            "success": "#2ecc71", "bad_q": "#e74c3c",
            "bad_ans": "#e67e22", "bad_claim": "#9b59b6", "bad_evid": "#34495e"
        }

        pie_labels = [labels_map[idx] for idx in weighted_statuses.index]
        pie_colors = [colors_map[idx] for idx in weighted_statuses.index]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor='white')

        # Pie Plot
        ax1.pie(weighted_statuses.values, labels=pie_labels, autopct='%1.1f%%', colors=pie_colors,
                startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        ax1.set_title(f"Normalized Audit Results\n(Weighted by Dataset Distribution)", fontweight='bold')

        # Bar Plot (Success Rate + N counts)
        strat_stats = df.groupby('strategy').apply(lambda x: (x['status'] == 'success').mean() * 100)
        counts_tagged = df.groupby('strategy').size()
        counts_total = pd.Series([s.get('Selected strategy', 'Unknown') for s in self.data]).value_counts()

        bar_colors = ['#2ecc71' if x >= 80 else '#f1c40f' if x >= 50 else '#e74c3c' for x in strat_stats]
        bars = ax2.bar(strat_stats.index, strat_stats.values, color=bar_colors, edgecolor='black')

        ax2.set_title("Success Rate by Strategy", fontweight='bold')
        ax2.set_ylabel("Pass Rate (%)")
        ax2.set_ylim(0, 120)
        plt.xticks(rotation=45, ha='right')

        for i, bar in enumerate(bars):
            strat = strat_stats.index[i]
            tagged_n = counts_tagged.get(strat, 0)
            total_n = counts_total.get(strat, 0)
            ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 2,
                     f'{int(bar.get_height())}%\n({tagged_n}/{total_n})',
                     ha='center', va='bottom', fontweight='bold', fontsize=9)

        plt.tight_layout()
        plt.show()

    def display(self):
        display(self.out)
        with self.out:
            clear_output()
            display(widgets.HBox([self.strat_dropdown, self.start_btn]))