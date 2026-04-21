"""
ParagraphAuditTagger: HITL Validation Suite for Evidence & Claim Paragraphs.

This module provides an interactive IPython/Jupyter widget interface for
human-in-the-loop (HITL) auditing of generated research datasets. It is
specifically designed to validate how well large language models (LLMs)
translate tabular data into narrative prose while maintaining factual
consistency and handling "Knowledge Conflict" scenarios.

Core Functionalities:
- **Interactive Tagger**: A multi-panel Jupyter UI that displays side-by-side
  comparisons of original Wikipedia data, "fake" source data, and generated
  paragraphs.
- **Visual Highlighting**: Implements fuzzy-matching regex engines to
  dynamically highlight data points in text, ensuring that specific numeric
  metrics and entity groups are present and accurate.
- **Constraint Monitoring**: Automatically identifies and highlights specific
  LLM formatting constraints, such as the [prefix A / B / C] certainty
  branches, to ensure structural compliance.
- **Progress Persistence**: Saves audit states to a JSON ledger, allowing
  researchers to pause and resume validation sessions across multiple days.
- **Weighted Analytics**: Includes a robust visualization suite using
  Matplotlib to calculate success rates per strategy, adjusting for the
  global distribution of the dataset to provide a "Weighted Pass Rate."

Statistical Formulas:
The module calculates the Adjusted Success Rate using:
$$Weighted\ Pass\ Rate = \sum_{i=1}^{n} (SuccessRate_{i} \times GlobalWeight_{i})$$
where $i$ represents each unique sampling strategy.

Required Environment:
- IPython / Jupyter Notebook
- ipywidgets, pandas, matplotlib
"""

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

        # 3. Slighter Progress Bar
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
        self.progress_label = widgets.HTML(value="")
        self.progress_box = widgets.VBox([self.progress, self.progress_label], layout=widgets.Layout(width='100%', margin='10px 0'))

        # 2. Pick number of samples
        self.sample_size_input = widgets.BoundedIntText(
            value=50, min=1, max=len(self.data), step=1,
            description='Samples to Audit:', style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.start_btn = widgets.Button(description="🚀 START AUDIT", button_style='primary')
        self.start_btn.on_click(lambda x: self.run(self.sample_size_input.value))

        # 4. Save & Exit Button
        self.save_exit_btn = widgets.Button(description="⏹️ SAVE & EXIT", style={'button_color': '#bdc3c7'}, layout=widgets.Layout(flex='1 1 auto', margin='0 4px', height='45px'))
        self.save_exit_btn.on_click(lambda x: self._save_and_exit())

    def display(self):
        display(self.out)
        with self.out:
            clear_output()
            # Show the input box and start button side by side
            display(widgets.HBox([self.sample_size_input, self.start_btn], layout=widgets.Layout(margin='15px 0')))

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

    def _highlight_data(self, paragraph, fake_data_list):
        """Highlights data points by ignoring all dashes, punctuation, and casing."""
        text = paragraph['Text']
        title = paragraph['Title']

        terms = set()
        for row in fake_data_list:
            for val in row.values():
                s = str(val).strip()
                if len(s) > 1 or re.fullmatch(r'\d', s):
                    terms.add(s)

        if not terms:
            highlighted_text = text
        else:
            sorted_terms = sorted(list(terms), key=len, reverse=True)
            highlighted_text = text

            for term in sorted_terms:
                parts = [re.escape(part) for part in re.split(r'[\s\W_]+', term) if part]
                if not parts: continue

                flexible_pattern = r"[\s\W_]+".join(parts)

                # --- NUMERIC SAFE MATCHING ---
                if re.fullmatch(r'\d+(\.\d+)?', term):
                    # Ensure we don't match numbers inside HTML tags (e.g., in hex codes)
                    pattern = re.compile(rf"(?<!\d)({re.escape(term)})(?!\d)(?![^<]*>)")

                # --- FUZZY TEXT MATCHING ---
                else:
                    pattern = re.compile(rf"\b({flexible_pattern})\b(?![^<]*>)", re.IGNORECASE)

                highlighted_text = pattern.sub(
                    r'<span class="hl hl-red" style="font-weight: bold; color: #721c24;">\1</span>',
                    highlighted_text
                )

        # --- RETURN FORMATTED HTML BLOCK ---
        return (
            f"<div style='margin-bottom: 24px; font-family: sans-serif;'>"
            f"  <strong style='"
            f"      font-size: 1.1em; "
            f"      color: #721c24; "  # Deep red/maroon color
            f"      text-decoration: underline; "
            f"      text-underline-offset: 5px; "
            f"      text-decoration-thickness: 2px;"
            f"      '>Title: {title}</strong>"
            f"  <p style='margin-top: 12px; line-height: 1.6; color: #333;'>{highlighted_text}</p>"
            f"</div>"
        )

    def highlight_answer_sentence(self, paragraph_data: dict, answer: str) -> str:
        # 1. Extract data from the new dictionary structure
        title = paragraph_data.get('Title', 'Untitled')
        text = paragraph_data.get('Text', '')
        confidence = paragraph_data.get('Confidence', 'Unknown')

        # 2. Normalize the answer into a set of lowercase words for scoring
        # This handles both numbers (e.g., "7") and multi-word answers (e.g., "Jon Priestley")
        answer_words = set(re.findall(r'\w+', str(answer).lower()))

        # 3. Find all sentence boundaries in the text
        sentence_spans = []
        start = 0
        for m in re.finditer(r"(?<=[.!?])\s+", text):
            end = m.end()
            sentence_spans.append((start, end))
            start = end
        if start < len(text):
            sentence_spans.append((start, len(text)))

        # 4. Score each sentence based on word overlap with the answer
        best_score = -1
        best_span = None

        for start, end in sentence_spans:
            sentence = text[start:end]
            sentence_words = set(re.findall(r'\w+', sentence.lower()))
            overlap = len(answer_words.intersection(sentence_words))

            if overlap > best_score:
                best_score = overlap
                best_span = (start, end)

        # 5. Apply the highlight wrap if an overlap was found
        if best_span and best_score > 0:
            start, end = best_span
            prefix = text[:start]
            raw_sentence = text[start:end]

            # UI Polish: Strip trailing space so the underline doesn't bleed between sentences
            stripped_sentence = raw_sentence.rstrip()
            trailing_spaces = raw_sentence[len(stripped_sentence):]
            suffix = text[end:]

            highlighted_text = (
                f"{prefix}"
                f"<span style='"
                f"background-color: #fff3cd; "
                f"color: #856404; "
                f"font-weight: 600; "
                f"text-decoration: underline; "
                f"text-underline-offset: 4px; "
                f"text-decoration-thickness: 1px;"
                f"'>{stripped_sentence}</span>"
                f"{trailing_spaces}{suffix}"
            )
        else:
            # Fallback in case no words match at all
            highlighted_text = text

        # 6. Format and return the final HTML block
        return (
            f"<div style='margin-bottom: 20px; font-family: sans-serif;'>"
            f"  <strong style='"
            f"      font-size: 1.1em; "
            f"      color: #856404; "  # Same deep gold as highlight text
            f"      text-decoration: underline; "
            f"      text-underline-offset: 5px; "  # Slightly more offset for the title
            f"      text-decoration-thickness: 2px;"  # Slightly thicker line for the header
            f"      '>Title: {title} ({confidence} confidence)</strong>"
            f"  <p style='margin-top: 12px; line-height: 1.6;'>{highlighted_text}</p>"
            f"</div>"
        )

    def _apply_group_highlight(self, html_text, group_name):
        if not group_name or not html_text:
            return html_text

        # 1. Define the 'Kill List' of words that cannot be at the start or end
        kill_list = {
            "the", "of", "for", "to", "and", "a", "an", "in", "on",
            "at", "with", "by", "from", "as", "is", "was"
        }

        # Normalize group words into a set for membership testing
        group_words_all = set(w.lower() for w in re.findall(r'\w+', group_name))
        if not group_words_all:
            return html_text

        words_in_text = list(re.finditer(r'\w+', html_text))
        best_match_indices = None
        max_overlap_count = 0
        min_window_length = float('inf')

        # 2. Sliding Window (checking clusters of words)
        window_limit = len(group_words_all) + 8

        for i in range(len(words_in_text)):
            for j in range(i + 1, min(i + window_limit + 1, len(words_in_text))):
                current_window = words_in_text[i:j]
                window_word_set = set(m.group(0).lower() for m in current_window)

                # Calculate how many words from our group name are present
                overlap = group_words_all.intersection(window_word_set)
                overlap_count = len(overlap)

                # 3. TRIMMING LOGIC using the Kill List
                if overlap_count >= 3:
                    # Shrink start: move forward if word is in kill_list OR not in group_words
                    start_idx = 0
                    while start_idx < len(current_window):
                        word = current_window[start_idx].group(0).lower()
                        if word in kill_list or word not in group_words_all:
                            start_idx += 1
                        else:
                            break

                    # Shrink end: move backward if word is in kill_list OR not in group_words
                    end_idx = len(current_window) - 1
                    while end_idx > start_idx:
                        word = current_window[end_idx].group(0).lower()
                        if word in kill_list or word not in group_words_all:
                            end_idx -= 1
                        else:
                            break

                    trimmed_window = current_window[start_idx:end_idx + 1]
                    if not trimmed_window:
                        continue

                    # Character span of the final trimmed cluster
                    current_span_length = trimmed_window[-1].end() - trimmed_window[0].start()

                    # Update if we found a better/tighter cluster
                    if (overlap_count > max_overlap_count) or \
                            (overlap_count == max_overlap_count and current_span_length < min_window_length):
                        max_overlap_count = overlap_count
                        min_window_length = current_span_length
                        best_match_indices = (trimmed_window[0].start(), trimmed_window[-1].end())

        # 4. Final Substitution
        if best_match_indices:
            start, end = best_match_indices
            match_text = html_text[start:end]
            return (
                    html_text[:start] +
                    f'<span class="hl hl-blue" style="font-weight: bold;">{match_text}</span>' +
                    html_text[end:]
            )

        return html_text

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
            claim_raw = self.paragraphs.get(sample.get("Claim paragraph", ""), "Evidence missing.")
            evidence_html = self._highlight_data(evidence_raw, sample.get("Fake data", []))
            claim_html = self.highlight_answer_sentence(claim_raw, ans['Claim'])

            group_name = sample.get('Group name', '')
            evidence_html = self._apply_group_highlight(evidence_html, group_name)
            claim_html = self._apply_group_highlight(claim_html, group_name)

            table_orig = self._prep_table(sample.get("Original data", []))
            table_fake = self._prep_table(sample.get("Fake data", []))

            # Get distractor IDs from the sample and look up their titles
            distractor_ids = sample.get('Distractor_IDs', [])
            distractor_titles = []
            for d_id in distractor_ids:
                # Look up title in paragraph_dataset; fallback to ID if Title missing
                para_info = self.paragraphs.get(d_id, {})  # Adjust 'paragraphs_metadata' to your actual dict name
                distractor_titles.append(para_info.get('Title', d_id))

            # Pad with empty strings to ensure we always have 8 slots for the 2x4 grid
            while len(distractor_titles) < 8:
                distractor_titles.append("---")

            distractor_slots_html = "".join(
                [f'<div class="distractor-slot" title="{t}">{t}</div>' for t in distractor_titles[:8]])

            style = """
            <style>
                .audit-container { font-family: 'Segoe UI', sans-serif; background: #f4f7f6; padding: 20px; border-radius: 12px; }
                .grid-container { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
                .panel { background: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); display: flex; flex-direction: column; position: relative; }

                /* Table Containers */
                .true-table-container { height: 140px; overflow-y: auto; border: 1px solid #ddd; background: #fff; margin-top: 10px; }
                /* Increased height for fake table since it has its own panel now */
                .fake-table-container { height: 300px; overflow-y: auto; border: 1px solid #ddd; background: #fff; margin-top: 10px; flex-grow: 1; }

                .data { border-collapse: collapse; width: 100%; font-size: 0.85em; margin: 0; }
                .data th, .data td { border-bottom: 1px solid #ddd; border-right: 1px solid #eee; padding: 6px 8px; text-align: left !important; }
                .data th { background: #f8f9fa; color: #333; position: sticky; top: 0; z-index: 1; }
                .data td:first-child, .data th:first-child { width: 30px; font-weight: bold; background: #fafafa; color: #7f8c8d; text-align: center !important; }

                /* Shared fixed height for both paragraphs */
                .para-box { font-size: 0.95em; line-height: 1.6; color: #2c3e50; padding: 12px; border-radius: 6px; background: #fdfdfd; border: 1px solid #efefef; margin-top: 10px; }
                .para-fixed-height { height: 320px; overflow-y: auto; }

                .label-tag { font-size: 0.75em; text-transform: uppercase; font-weight: bold; padding: 3px 8px; border-radius: 4px; display: inline-block; margin-bottom: 5px; }
                .wiki-link { text-decoration: none; color: #2980b9; background: #ffffff; border: 1px solid #2980b9; font-weight: bold; font-size: 0.8em; padding: 4px 10px; border-radius: 4px; display: inline-block; }

                .hl { padding: 1px 3px; border-radius: 2px; font-weight: 500; display: inline; line-height: 1; }
                .hl-blue { background-color: rgba(227, 242, 253, 0.8); color: #0D47A1; box-shadow: inset 0 -1px 0 #1E88E5; }
                .hl-green { background-color: rgba(232, 245, 233, 0.8); color: #1B5E20; box-shadow: inset 0 -1px 0 #43A047; }
                .hl-yellow { background-color: rgba(255, 249, 196, 0.8); color: #F57F17; box-shadow: inset 0 -1px 0 #FBC02D; }
                .hl-orange { background-color: rgba(255, 224, 178, 0.8); color: #E65100; box-shadow: inset 0 -1px 0 #FB8C00; }
                .hl-red { background-color: rgba(255, 205, 210, 0.8); color: #B71C1C; box-shadow: inset 0 -1px 0 #E53935; }


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
                    margin-top: 10px;
                }
                .ans-box {
                    padding: 8px;
                    border-radius: 6px;
                    border: 1px solid #eee;
                    font-size: 0.85em;
                    text-align: center;
                }
                
                /* Distractor Container */
                /* Match the established panel style */
                .distractor-panel { 
                    grid-column: span 2; 
                    background: white; 
                    border: 1px solid #e0e0e0; 
                    border-top: 4px solid #7f8c8d; /* Grey bold line */
                    border-radius: 8px; 
                    padding: 15px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    margin-top: 10px;
                    position: relative;
                }
    
                /* Full-width stretch for the label background */
                .distractor-label {
                    display: block; /* Stretches background to edges of padding */
                    background: #f8f9fa; 
                    color: #7f8c8d; 
                    font-size: 0.75em; 
                    text-transform: uppercase; 
                    font-weight: bold; 
                    padding: 4px 10px; 
                    margin: -15px -15px 10px -15px; /* Pulls label to the very edges of the panel */
                    border-bottom: 1px solid #eee;
                    border-radius: 4px 4px 0 0;
                }
    
                .distractor-grid { 
                    display: grid; 
                    grid-template-columns: repeat(4, 1fr); 
                    grid-gap: 8px; 
                }
    
                .distractor-slot { 
                    background: #ffffff; 
                    border: 1px solid #dfe6e9; 
                    padding: 8px; 
                    border-radius: 4px; 
                    font-size: 0.95em; 
                    color: #2c3e50; 
                    text-align: center;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }
            </style>
            """

            html_layout = f"""
            {style}
            <div class="audit-container">
                <div class="grid-container">

                    <div class="panel" style="border-top: 4px solid #2C3E50;">
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

                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px;">
                            <span class="label-tag" style="background: #e8f5e9; color: #2e7d32;">True World Data</span>

                            <a href="{wiki_url}" target="_blank" class="wiki-link">🌐 View Wikipedia</a>
                        </div>
                        <div class="true-table-container">{table_orig}</div>
                    </div>

                    <div class="panel" style="border-top: 4px solid #E74C3C;">
                        <span class="label-tag" style="background: #ffebee; color: #c62828;">Fake Evidence Context (Table)</span>
                        <div class="fake-table-container">{table_fake}</div>
                    </div>

                    <div class="panel" style="border-top: 4px solid #F39C12;">
                        <span class="label-tag" style="background: #fff3e0; color: #e65100;">Claim Context</span>
                        <div class="para-box para-fixed-height">{claim_html}</div>
                    </div>

                    <div class="panel" style="border-top: 4px solid #E74C3C;">
                        <span class="label-tag" style="background: #ffebee; color: #c62828;">Fake Evidence Context (Paragraph)</span>
                        <div class="para-box para-fixed-height">{evidence_html}</div>
                    </div>
                    <div class="distractor-panel">
                        <span class="distractor-label">Distractor Paragraphs</span>
                        <div class="distractor-grid">
                            {distractor_slots_html}
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

            display(self.progress_box)

            display(widgets.HBox(
                [b_good, b_bad_q, b_bad_a, b_bad_c, b_bad_e, self.save_exit_btn],
                layout=widgets.Layout(margin='20px 0', width='100%')
            ))

    def run(self, requested_size):
        # 1. Freshly identify what hasn't been tagged yet
        tagged_uuids = {res['ID'] for res in self.results}
        available = [s for s in self.data if s.get('ID') not in tagged_uuids]

        # 2. Guard: If nothing is left, don't even try to show a sample
        if not available:
            with self.out:
                clear_output()
                print("🎉 All samples in the dataset have been tagged!")
            return

        # 3. Create the session pool
        random.shuffle(available)
        actual_size = min(requested_size, len(available))
        self.sample_pool = available[:actual_size]

        # 4. RESET the session index
        self.current_idx = 0

        # 5. Reset the Progress Bar for this specific session
        self.progress.max = len(self.sample_pool)
        self.progress.value = 0
        self.progress_label.value = f"<div style='text-align: center;'>Session Progress: 0 / {len(self.sample_pool)}</div>"

        with self.out:
            clear_output()
            self._show_sample()

    def _handle_click(self, status):
        sample = self.sample_pool[self.current_idx]
        self.results.append({
            "ID": sample.get("uuid"),
            "strategy": sample.get('Selected strategy', 'Unknown'),
            "status": status
        })
        self._save_progress()

        self.current_idx += 1
        self.progress.value += 1
        self.progress_label.value = f"<div style='text-align: center; font-size: 0.9em; margin-top: 5px;'>Session Progress: <b>{self.progress.value} / {self.progress.max}</b></div>"

        self._show_sample()

    def _save_and_exit(self):
        """Allows stopping in the middle of a session."""
        self._save_progress()
        with self.out:
            clear_output()
            print(f"💾 Progress saved! You tagged {self.current_idx} items in this session.")
            print(f"Total dataset completion: {len(self.results)} / {len(self.data)}")
            # Provide option to restart UI
            display(widgets.HBox([self.sample_size_input, self.start_btn], layout=widgets.Layout(margin='15px 0')))

    def plot_results(self):
        if not self.results:
            print("⚠️ No data tagged yet.")
            return

        df_tagged = pd.DataFrame(self.results)
        df_all = pd.DataFrame(self.data)

        if 'score' not in df_tagged.columns:
            if 'status' in df_tagged.columns:
                # Convert "success" string to 1, everything else to 0
                df_tagged['score'] = (df_tagged['status'] == 'success').astype(int)
            else:
                raise KeyError("The results must contain either a 'score' or 'status' column.")

        global_counts = df_all['Selected strategy'].value_counts(normalize=True)

        # This line will no longer crash!
        strat_stats = df_tagged.groupby('strategy')['score'].agg(['count', 'sum', 'mean'])
        strat_stats['mean_pct'] = strat_stats['mean'] * 100

        # 3. Calculate Weighted Pass Rate
        # Formula: $\sum (\text{Strategy Success Rate} \times \text{Global Strategy Weight})$
        weighted_pass_rate = 0
        for strat, weight in global_counts.items():
            if strat in strat_stats.index:
                success_rate = strat_stats.loc[strat, 'mean']
                weighted_pass_rate += (success_rate * weight)
            else:
                # If a strategy hasn't been audited yet, we treat it as 0% for caution
                # or skip it to normalize among audited types.
                pass

        weighted_fail_rate = 1 - weighted_pass_rate

        # 4. Plotting - Layout Config
        fig = plt.figure(figsize=(20, 8), facecolor='white')
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.8])

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])

        # --- PIE 1: UNWEIGHTED (Raw Audit Data) ---
        passes = df_tagged['score'].sum()
        fails = len(df_tagged) - passes

        ax1.pie([passes, fails],
                labels=['Pass', 'Fail'], autopct='%1.1f%%',
                colors=['#2ecc71', '#e74c3c'], startangle=140,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2},
                textprops={'fontweight': 'bold'})
        ax1.set_title("Unweighted Success Rate\n(Tagged Samples Only)", fontsize=12, fontweight='bold', pad=15)

        # --- PIE 2: GLOBAL WEIGHTED (The "Real" Result) ---
        ax2.pie([weighted_pass_rate, weighted_fail_rate],
                labels=['Pass', 'Fail'], autopct='%1.1f%%',
                colors=['#27ae60', '#c0392b'], startangle=140,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2},
                textprops={'fontweight': 'bold'})
        ax2.set_title("Weighted Success Rate\n(Adjusted for Population)", fontsize=12, fontweight='bold', pad=15)

        # --- BAR PLOT: PER-STRATEGY PERFORMANCE ---
        # Dynamic coloring: Green (>=70%), Yellow (40-70%), Red (<40%)
        colors = ['#2ecc71' if x >= 70 else '#f1c40f' if x >= 40 else '#e74c3c' for x in strat_stats['mean_pct']]
        bars = ax3.bar(strat_stats.index, strat_stats['mean_pct'], color=colors, edgecolor='black', alpha=0.85)

        ax3.set_title("Success % per Strategy", fontsize=14, fontweight='bold', pad=20)
        ax3.set_ylabel("Success Rate (%)", fontweight='bold')
        ax3.set_ylim(0, 135)  # Extra space for the 2-line text labels
        ax3.grid(axis='y', linestyle='--', alpha=0.3)

        plt.setp(ax3.get_xticklabels(), rotation=40, ha='right', fontsize=10)

        # Add descriptive labels: Percentage + (Success/Total)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            total_checked = int(strat_stats.iloc[i]['count'])
            success_count = int(strat_stats.iloc[i]['sum'])

            label_text = f"{height:.0f}%\n({success_count}/{total_checked})"

            ax3.text(bar.get_x() + bar.get_width() / 2., height + 3,
                     label_text, ha='center', va='bottom',
                     fontweight='bold', fontsize=10, color='#2c3e50')

        # Final Polish
        n_samples = len(df_tagged)
        fig.suptitle(f"Human Validation Audit Report: {n_samples} Samples",
                     fontsize=20, fontweight='bold', y=1.02, color='#1e293b')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()