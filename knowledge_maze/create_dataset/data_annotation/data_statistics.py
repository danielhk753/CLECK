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
from rank_bm25 import BM25Okapi
from collections import Counter
import matplotlib.pyplot as plt
from urllib.parse import unquote
from rouge_score import rouge_scorer
from IPython.display import display, HTML
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer

from knowledge_maze.create_dataset.create_questions.wikipedia_mining import ENTITY_GROUPS


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
        self.plot_human_validation_pie()

        auditor = RelevanceAuditor()
        sample_triplets = [
            {
                'question': s['Question'],
                'claim': self.paragraph_dataset[s['Claim paragraph']]['Text'],
                'evidence': self.paragraph_dataset[s['Evidence paragraph']]['Text']
            }
            for s in self.dataset
        ]
        df_results = auditor.calculate_scores(sample_triplets)
        auditor.plot_distributions(df_results)

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
        order = ['General', 'Hedged']

        # Filter to only include categories that actually exist in your data
        labels = [lvl for lvl in order if lvl in counts]
        sizes = [counts[lvl] for lvl in labels]

        # 3. Design Aesthetics
        # Using a clean, professional palette
        colors = {
            'General': '#2ecc71',  # Emerald Green
            'Hedged': '#2980b9',  # Sunflower Yellow
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
        plt.title('Portion of Hedged Control Subset',
                  fontsize=18, fontweight='bold', pad=20, color='#1e293b')

        # Add a legend to the side
        plt.legend(wedges, [f"{l}: {counts[l]} samples" for l in labels],
                   title="Certainty Levels",
                   loc="center left",
                   bbox_to_anchor=(1, 0, 0.5, 1),
                   frameon=True)

        plt.tight_layout()
        plt.show()

    def plot_human_validation_pie(self):
        """
        Plots a donut chart showing the proportion of human-verified vs. pending samples.
        """
        # 1. Extract validation status from the question dataset
        # We look at the question dataset because 'Human Validated' is a question-level gate
        validations = []
        for sample in self.dataset:
            status = sample.get('Human Validated', False)
            # Convert boolean to readable string
            label = 'Verified' if status is True else 'Not Verified'
            validations.append(label)

        if not validations:
            print("⚠️ No samples found to plot.")
            return

        # 2. Count occurrences
        counts = Counter(validations)
        order = ['Verified', 'Not Verified']

        labels = [lvl for lvl in order if lvl in counts]
        sizes = [counts[lvl] for lvl in labels]

        # 3. Design Aesthetics (Verification Palette)
        colors = {
            'Verified': '#059669',  # Professional Forest Green
            'Not Verified': '#f59e0b',  # Alert Amber
        }
        slice_colors = [colors.get(l, '#94a3b8') for l in labels]

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
            textprops={'fontsize': 12, 'fontweight': 'bold', 'color': '#1e293b'}
        )

        # Style the percentage text
        for autotext in autotexts:
            autotext.set_color('white')

        # Create the Donut hole
        centre_circle = plt.Circle((0, 0), 0.55, fc='#f8fafc')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)

        # 5. Formatting
        plt.title('Dataset Audit Progress',
                  fontsize=18, fontweight='bold', pad=20, color='#1e293b')

        # Legend with raw counts
        plt.legend(wedges, [f"{l}: {counts[l]} samples" for l in labels],
                   title="Audit Status",
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


class RelevanceAuditor:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(embedding_model_name)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def _tokenize(self, text):
        """Simple whitespace tokenizer for BM25."""
        return text.lower().replace('.', '').replace(',', '').replace('?', '').split()

    def calculate_scores(self, triplets):
        results = []
        for item in triplets:
            q, c, e = item['question'], item['claim'], item['evidence']

            # 1. Semantic (Cosine)
            vec_q, vec_c, vec_e = self.encoder.encode([q, c, e])
            cos_c, cos_e = cos_sim(vec_q, vec_c).item(), cos_sim(vec_q, vec_e).item()

            # 2. Lexical (ROUGE-L)
            rouge_c = self.rouge_scorer.score(q, c)['rougeL'].fmeasure
            rouge_e = self.rouge_scorer.score(q, e)['rougeL'].fmeasure

            # 3. Keyword (BM25)
            tokenized_corpus = [self._tokenize(c), self._tokenize(e)]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = self._tokenize(q)
            bm25_scores = bm25.get_scores(tokenized_query)
            bm25_c, bm25_e = bm25_scores[0], bm25_scores[1]

            results.append({
                'Cosine_Claim': cos_c, 'Cosine_Evidence': cos_e, 'Delta_Cosine': cos_c - cos_e,
                'ROUGE_Claim': rouge_c, 'ROUGE_Evidence': rouge_e, 'Delta_ROUGE': rouge_c - rouge_e,
                'BM25_Claim': bm25_c, 'BM25_Evidence': bm25_e, 'Delta_BM25': bm25_c - bm25_e
            })
        return pd.DataFrame(results)

    def _apply_style(self, ax, data_col, title, threshold_negligible=0.05, threshold_significant=0.10):
        """Helper to style the parity plots with centered axes and background zones."""
        # 1. Symmetric limits so 0 is centered
        limit = max(abs(data_col.min()), abs(data_col.max()), 0.2)  # min 0.2 spread
        ax.set_xlim(-limit, limit)

        # 2. Background Zones
        # Negligible Zone (Green)
        ax.axvspan(-threshold_negligible, threshold_negligible, color='green', alpha=0.1, label='Negligible Diff')

        # Bias Zones (Red) - Where one side is significantly more relevant
        ax.axvspan(threshold_significant, limit, color='red', alpha=0.05, label='High Claim Bias')
        ax.axvspan(-limit, -threshold_significant, color='red', alpha=0.05, label='High Evidence Bias')

        # 3. Center and Mean lines
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Perfect Parity')
        ax.axvline(data_col.mean(), color='black', linestyle=':', linewidth=2,
                   label=f'Mean Shift ({data_col.mean():.3f})')

        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xlabel(r"$\leftarrow$ Evidence More Rel. | Claim More Rel. $\rightarrow$", fontsize=9)
        ax.legend(fontsize='x-small', loc='upper right')

    def plot_distributions(self, df):
        sns.set_theme(style="whitegrid")
        # Shifted to a 2x3 layout with a more compact figure size
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))

        # --- ROW 1: ABSOLUTE DISTRIBUTIONS ---
        # Cosine
        sns.kdeplot(df['Cosine_Claim'], ax=axes[0, 0], fill=True, label='Claim', color='purple')
        sns.kdeplot(df['Cosine_Evidence'], ax=axes[0, 0], fill=True, label='Evidence', color='orange')
        axes[0, 0].set_title("Absolute Semantic (Cosine)", fontweight='bold', fontsize=11)
        axes[0, 0].legend(fontsize='small')

        # ROUGE-L
        sns.kdeplot(df['ROUGE_Claim'], ax=axes[0, 1], fill=True, label='Claim', color='teal')
        sns.kdeplot(df['ROUGE_Evidence'], ax=axes[0, 1], fill=True, label='Evidence', color='green')
        axes[0, 1].set_title("Absolute Structural (ROUGE-L)", fontweight='bold', fontsize=11)
        axes[0, 1].legend(fontsize='small')

        # BM25
        sns.kdeplot(df['BM25_Claim'], ax=axes[0, 2], fill=True, label='Claim', color='coral')
        sns.kdeplot(df['BM25_Evidence'], ax=axes[0, 2], fill=True, label='Evidence', color='steelblue')
        axes[0, 2].set_title("Absolute Keyword (BM25)", fontweight='bold', fontsize=11)
        axes[0, 2].legend(fontsize='small')

        # --- ROW 2: DELTAS (Detailed with Zones) ---
        # Cosine Delta
        sns.histplot(df['Delta_Cosine'], ax=axes[1, 0], kde=True, color='purple', bins=30)
        self._apply_style(axes[1, 0], df['Delta_Cosine'], "Semantic Parity ($\Delta$ Cosine)")

        # ROUGE Delta
        sns.histplot(df['Delta_ROUGE'], ax=axes[1, 1], kde=True, color='teal', bins=30)
        self._apply_style(axes[1, 1], df['Delta_ROUGE'], "Structural Parity ($\Delta$ ROUGE-L)")

        # BM25 Delta (Using wider thresholds due to unbounded scale)
        sns.histplot(df['Delta_BM25'], ax=axes[1, 2], kde=True, color='coral', bins=30)
        self._apply_style(axes[1, 2], df['Delta_BM25'], "Keyword Parity ($\Delta$ BM25)",
                          threshold_negligible=0.10, threshold_significant=0.20)

        plt.suptitle("Question-Paragraph Semantic & Lexical Relevance", fontweight='bold', fontsize=16, y=1.05)
        plt.tight_layout()
        plt.show()