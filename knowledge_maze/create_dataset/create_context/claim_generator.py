"""
Claim Paragraph Generation Module
---------------------------------
This module automates the construction of "Contextual Haystacks"—synthetic,
knowledge-rich paragraphs designed to house specific factual claims within
abstract filler text.

Key Features:
- **Dynamic Filler Engine**: Utilizes a curated pool of abstract thematic fillers
  (Academic, Cultural, Philosophical) to pad paragraphs without introducing
  competing factual noise.
- **Strategic Claim Placement**: Implements "Needle-in-a-Haystack" positioning
  rules (Start, Middle, End) to vary the difficulty of information retrieval.
- **Epistemic Variation**: Injects multi-level certainty markers (High, Medium,
  Low) into a single claim sentence using a bracketed choice syntax.
  This facilitates testing an LLM's sensitivity to source attribution and
  probabilistic language.
- **Strict Abstraction**: Enforces a "Zero External Fact" rule for filler text,
  ensuring that the only specific information in the paragraph is the target
  answer, thereby preventing data leakage or accidental distractor creation.

Classes:
    ClaimParagraph: Manages the generation, formatting, and storage of claim-bearing
    contextual paragraphs.
"""

import re
import json
import random
import pandas as pd
import seaborn as sns
from uuid import uuid4
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from IPython.display import display, HTML


from knowledge_maze.tools.api_tools import OpenRouterAPI

class ClaimParagraph:
    def __init__(self):
        self.question_dataset = []
        self.paragraph_dataset = {}

        # --- Claim Placement Directives ---
        self.PLACEMENT_RULES = {
            "start": "Place the claim in the very FIRST sentence of the paragraph. The rest of the paragraph must organically flow out from this opening statement.",
            "middle": "Place the claim somewhere in the MIDDLE of the paragraph, burying it deeply within the filler text.",
            "end": "Place the claim in the very LAST sentence of the paragraph, building up to it as a concluding thought."
        }

        # --- Epistemic Marker Syntax Dictionary ---
        self.CERTAINTY_COMPONENTS = {
            "High": [
                # Level 1: Axiomatic & Declarative (The "Ground Truth" approach)
                "The verified record establishes that",
                "The confirmed data indicates that",
                "It is an objective and established fact that",
                "The final tally confirms that",
                "Official documentation shows that",
                "The primary source confirms that",
                "It is a matter of documented record that",
                "The specific classification is that",
                "The authenticated figure is",
                "The evidence unequivocally shows that",
                "As a matter of proven fact,",
                "The validated result is that",
                "The actual state of affairs is that",
                "The recorded status is that",
                "The conclusive finding is that",
                "As established by the data,",
                "The confirmed reality is that",
                "The definitive result shows that",
                "By all official accounts,",
                "The verified outcome is that"
            ],
            "Medium": [
                # Level 2: Logical & Probable (The "Reasoned" approach)
                "It is logically consistent to assume that",
                "Given the context, it stands to reason that",
                "Based on typical patterns, it is likely that",
                "Following the established trend, it appears that",
                "A standard analysis suggests that",
                "It is a reasonable inference that",
                "Based on the available information, it follows that",
                "The most plausible conclusion is that",
                "Everything indicates that it is suggested that",
                "Under normal circumstances, it is understood that",
                "Applying deductive logic, it is expected that",
                "The prevailing understanding is that",
                "It is highly probable that",
                "All signs point to the fact that",
                "Based on a synthesis of the data, it seems that",
                "It is internally consistent to conclude that",
                "There is a strong logical case that",
                "Following the common trajectory, it is likely",
                "By general interpretation, it is assumed that",
                "The most likely scenario is that"
            ],
            "Low": [
                "Despite a lack of corroboration, it is rumored that",
                "A speculative and unverified account suggests that",
                "While the evidence is notably thin, some claim that",
                "It is a frequent but unsubstantiated rumor that",
                "Without any official documentation, one might guess that",
                "Though likely a common misconception, it is often said that",
                "Purely on a conjectural basis, it could be argued that",
                "Based on anecdotal and unreliable hearsay, it appears that",
                "Pending a more rigorous investigation, it is hypothesized that",
                "An unconfirmed and potentially dubious report indicates that",
                "While most experts remain skeptical, a fringe theory holds that",
                "A tentative and largely unsupported observation hints that",
                "It is an open and unsettled question whether it is true that",
                "Although the primary sources are silent, some suspect that",
                "There is a marginal and unverified possibility that",
                "In the absence of any concrete proof, it is posited that",
                "It is a matter of significant doubt, yet some suggest that",
                "Initial and likely incomplete notes tentatively state that",
                "While the consensus leans elsewhere, a few observers claim that",
                "Following a cursory and inconclusive review, it seems that"
            ]
        }

    def _build_user_prompt(self, question: str, claim: str, group: str) -> tuple[str, Any]:
        """
        Constructs the prompt for the claim generation.
        """
        # Randomize placement and fillers
        placement_key = random.choice(list(self.PLACEMENT_RULES.keys()))
        placement_rule = self.PLACEMENT_RULES[placement_key]
        prefixes = [random.choice(self.CERTAINTY_COMPONENTS['High']),
                    random.choice(self.CERTAINTY_COMPONENTS['Medium']),
                    random.choice(self.CERTAINTY_COMPONENTS['Low'])]
        levels = ['High', 'Medium', 'Low']
        options = list(zip(prefixes, levels))
        chosen_prefix, confidence_level = random.choice(options)

        normalized_group = re.sub(r'^\s*Top\s+(?:[3-9]|1[0-2])\s+', '', group, flags=re.IGNORECASE).lower()

        return f"""### SYSTEM ROLE
        You are a Senior Academic Editor for a global encyclopedia. You write self-contained, authoritative entries that are comprehensible to a reader who has no prior context. 

        ### TASK
        Generate a research-grade "filler" paragraph that contextualizes a specific claim. You must output the result in a STRICT JSON format.

        ### INPUT DATA
        - SUBJECT GROUP: "{normalized_group}"
        - QUESTION: "{question}"
        - TARGET ANSWER (THE CLAIM): "{claim}"
        - PLACEMENT RULE: {placement_rule}
        - CERTAINTY PREFIX: {chosen_prefix}

        ### INTERNAL REASONING PROCESS (CoT)
        Before generating the JSON, you must mentally perform these steps:
        1. **Identify the Anchor:** Find the broader "Parent Entity" for "{normalized_group}". (e.g., If the group is "Historical buildings in Paris," the anchor is "Urban Development of Paris").
        2. **Determine the Lens:** Select a professional theme appropriate for the paragraph.
        3. **Verify Constraints:** Ensure NO dates, NO unauthorized names, and NO stats are used in the filler.

        ### WRITING CONSTRAINTS
        1. **The Anchor Rule:** The paragraph must introduce the "{normalized_group}" and its significance.
           - Examples of proper "Anchoring": 
             - "Swindon Town F.C. managers" -> Anchor on the history/culture of **Swindon Town F.C.**
             - "Ambassadors of Palestine to Chile" -> Anchor on the **Palestine and Chile relations**.
             - "Top 10 highest-grossing Telugu films by opening day" -> Anchor on **Telugu films**.
        2. **The No Evidence Rule:** Use ZERO dates, specific names, or numbers. Focus on *how* or *why*. 
           - **EXCEPTION:** The **TARGET ANSWER (THE CLAIM)** must remain specific.
        3. **Initials Expansion:** Expand all initials in the claim (e.g., "the United States of America (USA)").
        4. **Claim Syntax & Anti-Clipping Rule:** - Start the claim sentence with the provided CERTAINTY PREFIX, make sure it is a grammatically correct sentence.
           - **CRITICAL:** Do not "clip" the claim. Because the TARGET ANSWER is often just a raw number or short name (e.g., "8" or "Jon"), you MUST build a full, grammatically complete noun phrase around it using context from the QUESTION so the answer will be self contained. 
           - *INCORRECT:* "{chosen_prefix} 8."
           - *CORRECT:* "{chosen_prefix} 8 Tamil films were released during the year 2022."
        5. **Group name insertion:** The group name must appear in the text, you can rephrase it to match the grammar or delete parts of name if they are redundant but one should be able to identify the group.

        ### OUTPUT FORMAT
        Output ONLY a valid JSON object. Do not include any markdown preamble or post-text.
        {{
          "paragraph_subject": "A 3-5 word title identifying the Anchor and Lens identified in CoT step 1 & 2.",
          "paragraph_text": "A 150-200 word standalone paragraph following all constraints above."
        }}
        """, confidence_level

    async def generate_claim_paragraphs(self, samples: List[Dict[str, Any]]) -> None:
        """
        Takes a list of sample dictionaries, queries the LLM for the base paragraph and 3 claims,
        assembles them, and stores them.
        """
        global_system_prompt = (
            "You are an automated data-generation pipeline. You strictly output valid, raw JSON "
            "matching the requested schema. You never output markdown formatting, conversational text, "
            "or explanations."
        )

        api = OpenRouterAPI(
            system_prompt=global_system_prompt,
            model="google/gemini-3-flash-preview",
            output_format="string",  # We request string but instruct JSON to parse it manually
            max_concurrent_requests=15,
            temperature=0.4  # Slightly higher temp to allow for good prose in the filler
        )

        queries = []
        confidences = []
        for sample in samples:
            query, confidence = self._build_user_prompt(
                question=sample["Question"],
                claim=sample["Answer"]["Claim"],
                group=sample["Group name"]
            )
            queries.append(query)
            confidences.append(confidence)

        # Execute async calls
        results = await api.async_run(queries, desc="Generating Claim Paragraphs")

        for sample, result, confidence in zip(samples, results, confidences):

            para_uuid = str(uuid4())
            try:
                para_content = json.loads(result)
            except (json.JSONDecodeError, TypeError, AttributeError):
                print(f"FAILED TO PARSE JSON FOR UUID: {para_uuid}, skipping sample...")
                continue

            # Store paragraphs
            self.paragraph_dataset[para_uuid] = {
                'Title': para_content['paragraph_subject'],
                'Text':  para_content['paragraph_text'],
                'Confidence': confidence
            }
            sample["Claim paragraph"] = para_uuid

            self.question_dataset.append(sample)


class ClaimStrengthCheck:
    def __init__(self, dataset: List[Dict], paragraph_dataset: Dict, text_key: str = 'Claim paragraph',
                 api_model: str = "google/gemini-2.0-flash-001"):
        self.dataset = dataset
        self.paragraph_dataset = paragraph_dataset
        self.text_key = text_key

        self.system_prompt = """
You are a highly pedantic Epistemic Investigator. 
SCENARIO: You have been asked a critical question, and the provided text is the ONLY piece of evidence you have found in the world to answer it.

Your task is to output a "Certainty Coefficient" (0.0 to 1.0) based solely on the linguistic strength and evidential framing of that sentence.

RULES:
1. FORGET THE REAL WORLD: Ignore your internal training data. If the text says "The moon is made of blue cheese," but frames it as a "verified absolute fact," your score must be 1.0.
2. CONTINUOUS SPECTRUM: Use the full granularity of the scale (e.g., 0.87, 0.42). 
3. FRAMEWORK:
   - 1.0: Axiomatic, documented, or mathematically certain.
   - 0.7 - 0.9: Strong expert consensus or high-quality documentation.
   - 0.5: Purely neutral; a 50/50 logical toss-up.
   - 0.2 - 0.4: Hedged, speculative, or based on "some scattered reports."
   - 0.0: Definitive falsehood or an admitted error.

Output ONLY a JSON object: {"Output": [float]}
"""
        self.api = OpenRouterAPI(
            system_prompt=self.system_prompt,
            model=api_model,
            output_format='string'
        )

    async def run(self) -> pd.DataFrame:
        """Orchestrates auditing based on pre-classified paragraphs."""
        test_batch = []

        for item in self.dataset:
            lookup_key = item.get(self.text_key)
            para_data = self.paragraph_dataset.get(lookup_key)

            if para_data and isinstance(para_data, dict):
                content = para_data.get('Text', '')
                confidence_level = para_data.get('Confidence', 'Unknown')

                if content:
                    test_batch.append({
                        "level": confidence_level,
                        "prompt": content
                    })

        if not test_batch:
            print("🛑 No valid paragraphs found.")
            return pd.DataFrame()

        print(f"🚀 Evaluating {len(test_batch)} samples via {self.api.model}...")

        queries = [t['prompt'] for t in test_batch]
        raw_outputs = await self.api.async_run(queries, desc="Epistemic Audit")

        results = []
        for i, output in enumerate(raw_outputs):
            try:
                # 1. Clean the string
                clean_output = str(output).strip().strip("`").replace("json", "", 1).strip()

                # 2. Parse JSON
                data = json.loads(clean_output)

                # 3. ROBUST EXTRACTION
                score = None

                # Case A: It's a dictionary (The requested format)
                if isinstance(data, dict):
                    val = data.get("Output")
                    score = float(val[0]) if isinstance(val, list) else float(val)

                # Case B: It's a list (e.g., [0.85])
                elif isinstance(data, list):
                    score = float(data[0])

                # Case C: It's a raw float/int (The cause of your error)
                elif isinstance(data, (float, int)):
                    score = float(data)

                if score is not None:
                    results.append({
                        "level": test_batch[i]['level'],
                        "score": score
                    })

            except (ValueError, TypeError, json.JSONDecodeError, IndexError, KeyError):
                # Fallback: Regex search for the first decimal number in the string
                try:
                    import re
                    match = re.search(r"(\d?\.\d+)", str(output))
                    if match:
                        results.append({
                            "level": test_batch[i]['level'],
                            "score": float(match.group(1))
                        })
                except:
                    continue

        df = pd.DataFrame(results)

        if not df.empty:
            self.summarize(df)

        return df

    @staticmethod
    def summarize(df):
        html_styles = """
            <style>
                .audit-table { 
                    width: 75%; 
                    border-collapse: collapse; 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin-bottom: 20px;
                    table-layout: auto;
                }

                /* Strict Left Alignment Override */
                .audit-table th, .audit-table td { 
                    text-align: left !important; 
                    padding: 12px 15px; 
                    border-bottom: 1px solid #dee2e6; 
                    vertical-align: middle;
                }

                .audit-table th { 
                    background-color: #f8f9fa; 
                    color: #333; 
                    font-weight: 600;
                    border-bottom: 2px solid #dee2e6;
                }

                /* Badges should stay inline-block to respect left alignment */
                .badge { 
                    padding: 4px 10px; 
                    border-radius: 4px; 
                    font-weight: bold; 
                    font-size: 0.8em; 
                    text-transform: uppercase; 
                    display: inline-block;
                    margin: 0;
                }

                .bg-high { background-color: #d4edda; color: #155724; }
                .bg-med { background-color: #fff3cd; color: #856404; }
                .bg-low { background-color: #f8d7da; color: #721c24; }

                /* Bar aligned right of the text */
                .progress-bg { 
                    background: #e9ecef; 
                    border-radius: 10px; 
                    width: 80px; 
                    height: 8px; 
                    display: inline-block; 
                    margin-left: 10px; /* Space between number and bar */
                    vertical-align: middle;
                }

                .progress-fill { height: 100%; border-radius: 10px; }
                .score-text { font-weight: bold; font-family: monospace; display: inline-block; vertical-align: middle; }
            </style>
            """

        rows_html = ""
        colors = {"High": "#28a745", "Medium": "#ffc107", "Low": "#dc3545"}
        badges = {"High": "bg-high", "Medium": "bg-med", "Low": "bg-low"}

        for lvl in ["High", "Medium", "Low"]:
            sub = df[df['level'] == lvl]['score']
            if sub.empty: continue

            mean_val = sub.mean()
            pct = mean_val * 100

            rows_html += f"""
                <tr>
                    <td><span class="badge {badges[lvl]}">{lvl} Certainty</span></td>
                    <td>
                        <span class="score-text">{mean_val:.3f}</span>
                        <div class="progress-bg">
                            <div class="progress-fill" style="width: {pct}%; background-color: {colors[lvl]};"></div>
                        </div>
                    </td>
                    <td>{sub.median():.3f}</td>
                    <td>{sub.std():.3f}</td>
                    <td style="color: #6c757d;">{len(sub)} samples</td>
                </tr>
                """

        table_html = f"""
            {html_styles}
            <div style="background: white; padding: 25px; border-radius: 10px; border: 1px solid #ddd; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <h3 style="margin-top: 0; color: #2c3e50; text-align: left !important; font-family: sans-serif;">📊 Claim Conclusiveness Test</h3>
                <table class="audit-table">
                    <thead>
                        <tr>
                            <th>Confidence Level</th>
                            <th>Mean Score & Strength</th>
                            <th>Median</th>
                            <th>Std Dev</th>
                            <th>Sample Size</th>
                        </tr>
                    </thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
            """
        display(HTML(table_html))

        # 2. Plotting Distribution
        plt.figure(figsize=(14, 5))
        sns.set_theme(style="white")
        palette = {"High": "#28a745", "Medium": "#ffc107", "Low": "#dc3545"}

        sns.histplot(
            data=df, x="score", hue="level", hue_order=["High", "Medium", "Low"],
            palette=palette, binwidth=0.05, binrange=(0, 1),
            kde=True, multiple="dodge", shrink=0.8, alpha=0.6, common_norm=False
        )

        plt.xticks([i / 10 for i in range(11)])
        plt.title("Visual Density of Auditor Confidence", loc='left', fontsize=14, color='#2c3e50')
        plt.xlabel("Investigator Confidence Score")
        plt.ylabel("Frequency")
        plt.xlim(-0.02, 1.02)
        sns.despine()
        plt.show()
