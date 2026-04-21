"""
Knowledge Conflict Dataset Analytics & Visualization
====================================================

This module provides the `KnowledgeConflictDatasetStats` and `SampleVisualizer`
classes designed to quantify, audit, and visualize diagnostic datasets where
Large Language Models (LLMs) must navigate conflicting information.

Core Capabilities:
-----------------
1.  **Strategy Distribution**: Visualizes question generation strategies via
    hierarchical Sunburst charts.
2.  **Lexical Bias Analysis**: Measures n-gram recall overlap to detect whether
    questions are lexically biased toward the Claim or Evidence paragraphs.
3.  **Semantic Metadata Stats**: Plots the distribution of context lengths,
    subject categories (via Treemaps), and claim certainty levels.
4.  **High-Fidelity Rendering**: Provides a dual-panel HTML UI for Jupyter
    Notebooks that highlights contradictory data points, subject entities, and
    answer-bearing sentences using regex-based fuzzy matching.

Dependencies:
------------
- Visualization: matplotlib, seaborn, squarify
- Data Analysis: pandas, collections, nltk (ngrams)
- UI: IPython.display, urllib.parse

Usage:
------
    stats = KnowledgeConflictDatasetStats(my_dataset, my_paragraphs)
    await stats.show_statistics()
"""

import re
import random
import squarify
import pandas as pd
import seaborn as sns
from nltk.util import ngrams
from collections import Counter
import matplotlib.pyplot as plt
from urllib.parse import unquote
from IPython.display import display, HTML

from knowledge_maze.create_dataset.create_questions.wikipedia_mining import ENTITY_GROUPS
from knowledge_maze.create_dataset.create_context.claim_generator import ClaimStrengthCheck


class KnowledgeConflictDatasetStats:
    """
    A unified class to calculate and visualize statistics for the
    Knowledge Conflict diagnostic dataset.
    """

    def __init__(self, dataset, paragraphs):
        """
        :param dataset: List of dictionaries representing the generated dataset.
                        Expected keys per sample: 'Selected strategy', 'Fake data',
                        'Wiki link'/'Wiki url', 'Question', 'Claim paragraph',
                        'Evidence paragraph'.
        :param entity_groups: Dictionary mapping broad subjects to Wikipedia keywords.
        """
        self.dataset = dataset
        self.paragraph_dataset = paragraphs
        self.entity_groups = ENTITY_GROUPS

        # Define visual palettes for consistency
        self.colors = {
            'claim': '#e74c3c',  # Red for authority/conflict
            'evidence': '#3498db',  # Blue for grounded truth
            'bg': '#f8fafc',
            'text': '#1e293b'
        }

    async def show_statistics(self):
        """
        Orchestrates the generation of all dataset statistics and visualizations.
        Ideal for calling at the end of a Jupyter Notebook cell.
        """
        print(f"📊 Generating statistics for N={len(self.dataset)} samples...\n")

        triplets = [(s['Question'], self.paragraph_dataset[s['Claim paragraph']]['Text'], self.paragraph_dataset[s['Evidence paragraph']]['Text']) for s in self.dataset]
        claim_para = [self.paragraph_dataset[s['Claim paragraph']]['Text'] for s in self.dataset]
        evidence_para = [self.paragraph_dataset[s['Evidence paragraph']]['Text'] for s in self.dataset]


        self.plot_strategy_sunburst()
        self.plot_evidence_distribution()
        self.plot_subject_treemap()
        self.plot_claim_certainty_pie()

        certainty_auditor = ClaimStrengthCheck(
            dataset=self.dataset,
            paragraph_dataset=self.paragraph_dataset
        )

        await certainty_auditor.run()

        self.plot_retrieval_bias_distribution(triplets, n=2)
        self.plot_token_length_distribution(claim_para, evidence_para)
        self.display_strategy_examples()

    def plot_strategy_sunburst(self):
        """Generates a two-layered pie chart of question generation strategies."""
        mapping = {
            "Numerical": [
                "AGGREGATION", "EXTREME_NUMERICAL", "TOP_K_RANKING",
                "COMPARATIVE_DIFFERENCE", "FILTER_COUNT_NUMERICAL"
            ],
            "Categorical": [
                "UNIQUE_DIVERSITY", "FREQUENCY_MODE",
                "CATEGORICAL_FILTER_LOOKUP", "COUNT_TOTAL"
            ],
            "Temporal": [
                "EXTREME_TEMPORAL", "FILTER_COUNT_TEMPORAL"
            ]
        }

        strat_counts = Counter([sample.get('Selected strategy', 'UNKNOWN') for sample in self.dataset])
        known_strats = [s for sublist in mapping.values() for s in sublist]
        other_strats = [s for s in strat_counts if s not in known_strats]
        if other_strats:
            mapping["Other"] = other_strats

        inner_labels, inner_sizes, inner_colors = [], [], []
        outer_labels, outer_sizes, outer_colors = [], [], []

        colors_map = {
            "Numerical": plt.cm.Blues,
            "Categorical": plt.cm.Reds,
            "Temporal": plt.cm.Greens,
            "Other": plt.cm.Greys
        }

        for group, strats in mapping.items():
            group_total = sum(strat_counts[s] for s in strats)
            if group_total > 0:
                inner_labels.append(group)
                inner_sizes.append(group_total)
                inner_colors.append(colors_map.get(group, plt.cm.Purples)(0.6))

                for i, s in enumerate(strats):
                    count = strat_counts[s]
                    if count > 0:
                        outer_labels.append(f"{s} ({count})")
                        outer_sizes.append(count)
                        base_color = colors_map.get(group, plt.cm.Purples)
                        outer_colors.append(base_color(0.4 - (i * 0.05)))

        fig, ax = plt.subplots(figsize=(10, 10), facecolor=self.colors['bg'])

        ax.pie(outer_sizes, labels=outer_labels, radius=1, colors=outer_colors,
               wedgeprops=dict(width=0.3, edgecolor='w'), pctdistance=0.85,
               autopct='%1.1f%%', textprops={'fontsize': 9})

        ax.pie(inner_sizes, labels=inner_labels, radius=0.7, colors=inner_colors,
               labeldistance=0.4, wedgeprops=dict(width=0.3, edgecolor='w'),
               textprops={'fontsize': 11, 'fontweight': 'bold', 'color': 'black'})

        ax.set(aspect="equal")
        plt.title('Question Generation: Strategy Distribution', pad=20,
                  fontsize=16, fontweight='bold', color=self.colors['text'])
        plt.tight_layout()
        plt.show()

    def plot_evidence_distribution(self):
        """Plots a histogram of the number of evidence rows per question."""
        counts = [len(s.get('Fake data', [])) for s in self.dataset]
        if not counts:
            print("⚠️ No evidence data found for distribution plot.")
            return

        plt.figure(figsize=(10, 6), facecolor=self.colors['bg'])
        sns.set_theme(style="whitegrid")

        ax = sns.histplot(counts, bins=min(len(set(counts)), 20),
                          kde=True, color='#0ea5e9', edgecolor='#0284c7', alpha=0.7)

        plt.title('Distribution of Evidence (Table Rows per Question)',
                  fontsize=15, fontweight='bold', pad=20)
        plt.xlabel('Number of Evidence Rows', fontsize=12)
        plt.ylabel('Frequency (Number of Questions)', fontsize=12)

        median_val = sum(counts) / len(counts) if counts else 0
        plt.axvline(median_val, color='#ef4444', linestyle='--', label=f'Mean: {median_val:.1f}')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_subject_treemap(self):
        """Visualizes subjects as a Treemap to save vertical space."""
        subjects_found = []
        for s in self.dataset:
            url = s.get('Wiki link', '') or s.get('Wiki url', '')
            if not url:
                continue

            title = unquote(url.split('/')[-1].split('#')[0]).replace('_', ' ').lower()
            clean_title = re.sub(r'^list of ', '', title)

            best_subject, max_matches = "Other/General", 0
            for subject, keywords in self.entity_groups.items():
                matches = sum(1 for kw in keywords if kw.lower() in clean_title)
                if matches > max_matches:
                    max_matches = matches
                    best_subject = subject
            subjects_found.append(best_subject)
        counts = Counter(subjects_found)

        # Sort by size
        sorted_counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))

        # 1. Create the full list of labels and sizes
        labels = [f"{k}\n({v})" for k, v in sorted_counts.items()]
        sizes = list(sorted_counts.values())

        # 2. FIX: Create a padded label list so it matches the length of 'sizes'
        # This keeps the first 12 and makes the rest invisible.
        display_labels = labels[:12] + [""] * (len(sizes) - 12)

        # 3. Ensure colors also match the full length
        colors = plt.cm.Spectral([i / len(sizes) for i in range(len(sizes))])

        plt.figure(figsize=(12, 8), facecolor=self.colors['bg'])

        # Pass 'display_labels' instead of the sliced list
        squarify.plot(sizes=sizes,
                      label=display_labels,
                      color=colors,
                      alpha=0.8,
                      edgecolor="white",
                      linewidth=2)

        plt.title('Dataset Composition: Subject Treemap', fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def calculate_ngram_recall(self, query: str, paragraph: str, n: int = 2) -> float:
        """
        Calculates the n-gram recall between a query and a paragraph.
        Returns the percentage of query n-grams that are present in the paragraph.
        """

        # Simple cleaner: lowercase and remove punctuation
        def clean_text(text):
            return re.sub(r'[^\w\s]', '', str(text).lower()).split()

        q_tokens = clean_text(query)
        p_tokens = clean_text(paragraph)

        if len(q_tokens) < n:
            return 0.0  # Query too short for this n-gram size

        # Generate sets of n-grams
        q_ngrams = set(ngrams(q_tokens, n))
        p_ngrams = set(ngrams(p_tokens, n))

        # Calculate recall: Intersection / Total Query N-grams
        overlap = q_ngrams.intersection(p_ngrams)

        return len(overlap) / len(q_ngrams)

    def _apply_standard_style(self, title, xlabel, ylabel):
        """Internal helper to keep aesthetics consistent across all plots."""
        plt.title(title, fontsize=16, fontweight='bold', pad=15, color='#1e293b')
        plt.xlabel(xlabel, fontsize=12, fontweight='bold', color='#1e293b')
        plt.ylabel(ylabel, fontsize=12, fontweight='bold', color='#1e293b')

        # Remove plt.legend() from here; Seaborn handles it via 'legend=True' in histplot
        sns.despine(left=True, bottom=True)
        plt.tight_layout()

    def plot_retrieval_bias_distribution(self, triplets, n: int = 2):
        data = []
        for q, c, e in triplets:
            data.append({'Type': 'Claim', 'Score': self.calculate_ngram_recall(q, c, n)})
            data.append({'Type': 'Evidence', 'Score': self.calculate_ngram_recall(q, e, n)})

        df = pd.DataFrame(data)

        plt.figure(figsize=(10, 6), facecolor='#f8fafc')
        sns.set_theme(style="whitegrid")

        # legend=True ensures Seaborn maps the 'Type' to the legend automatically
        sns.histplot(
            data=df, x='Score', hue='Type',
            palette={'Claim': '#e74c3c', 'Evidence': '#3498db'},
            kde=True, stat='density', element='step', alpha=0.4, bins=25, linewidth=0,
            legend=True
        )

        self._apply_standard_style(
            title=f'Retrieval Bias: {n}-Gram Query Coverage',
            xlabel=f'Lexical Overlap Score (% of {n}-grams matched)',
            ylabel='Density'
        )
        plt.show()

    def plot_token_length_distribution(self, claim_paragraphs, evidence_paragraphs):
        data = []
        for p in claim_paragraphs:
            if isinstance(p, str): data.append({'Type': 'Claim', 'Length': len(p.split())})
        for p in evidence_paragraphs:
            if isinstance(p, str): data.append({'Type': 'Evidence', 'Length': len(p.split())})

        df = pd.DataFrame(data)

        plt.figure(figsize=(10, 6), facecolor='#f8fafc')
        sns.set_theme(style="whitegrid")

        sns.histplot(
            data=df, x='Length', hue='Type',
            palette={'Claim': '#e74c3c', 'Evidence': '#3498db'},
            kde=True, stat='density', element='step', alpha=0.4, bins=30, linewidth=0,
            legend=True
        )

        # For Mean Lines, we manually add them to the legend if needed,
        # but usually, for a clean look, let the hue legend speak for itself.
        means = df.groupby('Type')['Length'].mean()
        for label, m_val in means.items():
            plt.axvline(m_val, color={'Claim': '#e74c3c', 'Evidence': '#3498db'}[label],
                        linestyle='--', alpha=0.8)

        self._apply_standard_style(
            title='Context Length Distribution: Claim vs. Evidence',
            xlabel='Paragraph Word Count',
            ylabel='Density'
        )
        plt.show()

    def plot_claim_certainty_pie(self):
        """
        Extracts confidence levels from claim paragraphs and plots a professional pie chart.
        """
        # 1. Extract confidence levels
        confidences = []
        for sample in self.dataset:
            claim_id = sample.get('Claim paragraph')
            # Look up the paragraph in the metadata
            para_info = self.paragraph_dataset.get(claim_id, {})
            # Default to 'Unknown' if the key is missing
            level = para_info.get('Confidence', 'Unknown')
            confidences.append(level)

        if not confidences:
            print("⚠️ No confidence data found in claim paragraphs.")
            return

        # 2. Count and sort by logical order (High -> Medium -> Low -> Unknown)
        counts = Counter(confidences)
        order = ['High', 'Medium', 'Low', 'Unknown']

        # Filter to only include categories that actually exist in your data
        labels = [lvl for lvl in order if lvl in counts]
        sizes = [counts[lvl] for lvl in labels]

        # 3. Design Aesthetics
        # Using a clean, professional palette
        colors = {
            'High': '#2ecc71',  # Emerald Green
            'Medium': '#f1c40f',  # Sunflower Yellow
            'Low': '#e74c3c',  # Alizarin Red
            'Unknown': '#95a5a6'  # Concrete Grey
        }
        slice_colors = [colors.get(l, '#bdc3c7') for l in labels]

        # 4. Create the Plot
        plt.figure(figsize=(10, 7), facecolor='#f8fafc')

        # Create the pie chart
        wedges, texts, autotexts = plt.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=140,
            colors=slice_colors,
            pctdistance=0.75,
            textprops={'fontsize': 12, 'fontweight': 'bold', 'color': '#2c3e50'}
        )

        # Style the percentage text inside the slices
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')

        # Add a hole in the center to make it a "Donut Chart" (often looks cleaner for theses)
        centre_circle = plt.Circle((0, 0), 0.550, fc='#f8fafc')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)

        # 5. Formatting
        plt.title('Distribution of Claim Certainty Levels',
                  fontsize=18, fontweight='bold', pad=20, color='#1e293b')

        # Add a legend to the side
        plt.legend(wedges, [f"{l}: {counts[l]} samples" for l in labels],
                   title="Certainty Levels",
                   loc="center left",
                   bbox_to_anchor=(1, 0, 0.5, 1),
                   frameon=True)

        plt.tight_layout()
        plt.show()

    def display_strategy_examples(self):
        """Displays one random high-fidelity sample for every unique strategy in the dataset."""
        visualizer = SampleVisualizer(self.paragraph_dataset)

        # Group samples by strategy
        strat_map = {}
        for sample in self.dataset:
            strat = sample.get('Selected strategy', 'Unknown')
            if strat not in strat_map:
                strat_map[strat] = []
            strat_map[strat].append(sample)

        print(f"🔍 Displaying examples for {len(strat_map)} strategies...\n")

        # Iterate through strategies (sorted for consistency)
        for strat in sorted(strat_map.keys()):
            sample = random.choice(strat_map[strat])
            visualizer.render_sample(sample)



class SampleVisualizer:
    """Handles the high-fidelity HTML rendering of dataset samples."""

    def __init__(self, paragraph_dataset):
        self.paragraphs = paragraph_dataset

    def _highlight_text(self, text, terms, color="#721c24", bg_color="rgba(255, 205, 210, 0.8)", is_bold=True):
        """Returns text snippet with red term highlights including underline for parity."""
        if not terms or not text: return text
        if isinstance(text, dict): text = text.get('Text', 'Text Missing')

        sorted_terms = sorted(list(set(str(t) for t in terms if len(str(t)) > 0)), key=len, reverse=True)
        highlighted = text
        sep = r"[\s\W_]+"

        for term in sorted_terms:
            if not term.strip(): continue
            parts = [re.escape(p) for p in re.split(sep, term) if p]
            if not parts: continue
            flexible_pattern = sep.join(parts)

            if re.fullmatch(r'\d+(\.\d+)?', term):
                pattern = re.compile(rf"(?<!\d)({re.escape(term)})(?!\d)(?![^<]*>)")
            else:
                pattern = re.compile(rf"\b({flexible_pattern})\b(?![^<]*>)", re.IGNORECASE)

            weight = "bold" if is_bold else "normal"
            # Added text-decoration and offset here:
            highlighted = pattern.sub(
                f'<span style="background:{bg_color}; color:{color}; font-weight:{weight}; '
                f'text-decoration: underline; text-underline-offset: 3px; '
                f'text-decoration-thickness: 1px; border-radius:2px; padding:0 2px;">\\1</span>',
                highlighted
            )
        return highlighted

    def highlight_answer_sentence(self, paragraph_data: dict, answer: str):
        """Finds the most relevant sentence and applies a soft gold highlight."""
        title = paragraph_data.get('Title', 'Untitled')
        text = paragraph_data.get('Text', '')

        # 1. Sentence Scoring
        answer_words = set(re.findall(r'\w+', str(answer).lower()))
        sentence_spans = []
        start = 0
        for m in re.finditer(r"(?<=[.!?])\s+", text):
            sentence_spans.append((start, m.end()))
            start = m.end()
        sentence_spans.append((start, len(text)))

        best_score, best_span = -1, None
        for s, e in sentence_spans:
            sentence = text[s:e]
            overlap = len(answer_words.intersection(set(re.findall(r'\w+', sentence.lower()))))
            if overlap > best_score:
                best_score, best_span = overlap, (s, e)

        # 2. Apply Sentence Highlight
        if best_span and best_score > 0:
            s, e = best_span
            raw_sentence = text[s:e].rstrip()
            trailing = text[s:e][len(raw_sentence):]

            # Wrap the specific sentence in a gold span
            highlighted_sentence = (
                f"<span style='background-color:#fff3cd; color:#856404; font-weight:bold; "  # Change 600 to bold
                f"text-decoration:underline; text-underline-offset:4px;'>{raw_sentence}</span>{trailing}"
            )
            text = text[:s] + highlighted_sentence + text[e:]

        return text

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

    def render_sample(self, sample):
        """Generates the dual-panel HTML view for a single sample with fixed heights."""
        q = sample.get('Question', 'N/A')
        strat = sample.get('Selected strategy', 'Unknown')
        ans = sample.get('Answer', {})

        evid_id = sample.get('Evidence paragraph', '')
        claim_id = sample.get('Claim paragraph', '')

        evid_raw = self.paragraphs.get(evid_id, {})
        claim_raw = self.paragraphs.get(claim_id, {})

        fake_terms = [val for row in sample.get('Fake data', []) for val in row.values()]

        # Process HTML content
        evid_html = self._apply_group_highlight(self._highlight_text(
            evid_raw,
            fake_terms,
            color="#B71C1C",
            bg_color="rgba(255, 205, 210, 0.8)"
        ), sample.get('Group name', ''))

        claim_html = self._apply_group_highlight(
            self.highlight_answer_sentence(claim_raw, ans.get('Claim', '')),
            sample.get('Group name', '')
        )

        # UI Configuration
        panel_height = "320px"  # Fixed height for parity
        line_spacing = "1.6"  # Standard spacing for readability

        html = f"""
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; border: 1px solid #e0e0e0; border-radius: 12px; margin-bottom: 20px; overflow: hidden; background: #f4f7f6; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
            <div style="background: #2c3e50; color: white; padding: 12px 18px; display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: bold; letter-spacing: 0.5px;">STRATEGY: {strat}</span>
                <span style="font-size: 0.85em; background: rgba(255,255,255,0.2); padding: 2px 8px; border-radius: 4px;">{sample.get('Group name', 'General')}</span>
            </div>

            <div style="padding: 18px; background: white; border-bottom: 1px solid #eee;">
                <strong style="color: #2c3e50; font-size: 1.15em;">Q: {q}</strong>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 18px; background: white;">
                <div style="display: flex; flex-direction: column;">
                    <div style="font-size: 0.75em; text-transform: uppercase; color: #e65100; font-weight: bold; margin-bottom: 8px; border-bottom: 2px solid #F39C12; width: fit-content;">Claim Context (Counterfactual)</div>
                    <div style="
                        height: {panel_height}; 
                        overflow-y: auto; 
                        line-height: {line_spacing}; 
                        font-size: 0.95em; 
                        color: #333; 
                        padding-right: 10px;
                        scrollbar-width: thin;">
                        {claim_html}
                    </div>
                    <div style="margin-top: 12px; padding-top: 8px; border-top: 1px dashed #eee; font-size: 0.9em;">
                        <span style="color: #7f8c8d;">Target:</span> <b>{ans.get('Claim', 'N/A')}</b>
                    </div>
                </div>

                <div style="display: flex; flex-direction: column;">
                    <div style="font-size: 0.75em; text-transform: uppercase; color: #c62828; font-weight: bold; margin-bottom: 8px; border-bottom: 2px solid #E74C3C; width: fit-content;">Evidence Context (Fake Table)</div>
                    <div style="
                        height: {panel_height}; 
                        overflow-y: auto; 
                        line-height: {line_spacing}; 
                        font-size: 0.95em; 
                        color: #333; 
                        padding-right: 10px;
                        scrollbar-width: thin;">
                        {evid_html}
                    </div>
                    <div style="margin-top: 12px; padding-top: 8px; border-top: 1px dashed #eee; font-size: 0.9em;">
                        <span style="color: #7f8c8d;">Target:</span> <b>{ans.get('Evidence', 'N/A')}</b>
                    </div>
                </div>
            </div>
        </div>
        """
        display(HTML(html))