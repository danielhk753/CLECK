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
        self.PREFIX_HEDGES = [
            "It is generally accepted that",
            "It is generally recognized that",
            "It is commonly understood that",
            "It is broadly acknowledged that",
            "It is typically considered that",
            "It appears that",
            "It seems that",
            "It is likely that",
            "It is reasonable to conclude that",
            "Available information suggests that",
            "There is a strong likelihood that",
            "The most plausible explanation is that",
            "Evidence tends to support the claim that",
            "It can be argued that",
            "Current interpretations indicate that",
            "It is widely held that",
            "It is often noted that",
            "It is commonly regarded that",
            "It is broadly considered that",
            "It is typically understood that",
            "It is largely accepted that",
            "It is frequently observed that",
            "It is reasonable to infer that",
            "It is generally believed that",
            "It is standard to assume that"
        ]

    def _build_user_prompt(self, question: str, claim: str, group: str) -> tuple[str, Any]:
        """
        Constructs the prompt for the claim generation.
        """
        # Randomize placement and fillers
        placement_key = random.choice(list(self.PLACEMENT_RULES.keys()))
        placement_rule = self.PLACEMENT_RULES[placement_key]

        # Randomly choose samples for the hedged control subset:
        levels = ['General', 'Hedged']
        weights = [85, 15]
        is_hedged = random.choices(levels, weights=weights, k=1)[0]
        chosen_prefix = random.choice(self.PREFIX_HEDGES)

        # Logic for the instruction injection
        hedge_instruction = ""
        if is_hedged == 'Hedged':
            hedge_instruction = f"""
            6. **THE BRACKETED PREFIX RULE:**
               - **The Container:** You must use double brackets `[[ ]]` at the very beginning of the claim.
               - **The Content:** Place the hedging prefix (based on "{chosen_prefix}") **INSIDE** these brackets.
               - **No Floating Text:** No words should appear before the `[[`. The sentence must start with the opening brackets.
               - **The Standalone Requirement:** The text following the `]]` must be a complete, independent sentence.
               - **The Capitalization Rule:** The first letter after the `]]` MUST be capitalized.
               - **The Dual-Grammar Test:** 1. If I remove only the `[[` and `]]` characters, the whole line must be a smooth, grammatical sentence.
                 2. If I remove the entire `[[ ... ]]` block, the remaining text must be a perfect standalone sentence.

               - **Example Output:** [[ {chosen_prefix} ]] The experimental results remain inconclusive.
            """

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
        2. The Semantic Independence Rule (Zero-Context Answerability):
           - The claim sentence must be written so that a reader who has never seen the QUESTION can answer that question with 100% certainty after reading the paragraph.
           - Self-Contained Logic: You must explicitly link the specific entity from the question to the specific value/fact of the claim.
           - The "No-Ambiguity" Test: If the question asks for a "difference in meters," the sentence must not just state two numbers; it must describe the relationship so clearly that the "difference" is the only logical conclusion.
           - Unit Fidelity: You must use the exact units of measurement provided in the question (e.g., if the data is in "square kilometres", you must use "square kilometres").
        3. **The No Evidence Rule:** Use ZERO dates, specific names, or numbers. Focus on *how* or *why*. 
           - **EXCEPTION:** The **TARGET ANSWER (THE CLAIM)** must remain specific.
           - THE PROHIBITED LIST: You are strictly forbidden from using any of the following "softening" words in the paragraph: seems, appears, likely, probably, potentially, suggests, indicates, might, could, arguably, allegedly.
           - TONE LOCKDOWN: Write the claim sentence as if it is a mathematical constant or an undisputed law of physics. Use absolute, declarative verbs (e.g., "is", "constitutes", "represents").
        4. **The Proper Noun Sandwich Rule (Structural Fragmentation):**
           - Start the claim sentence with the provided CERTAINTY PREFIX.
           - **ENTITY LOCKDOWN:** You MUST retain the exact Proper Nouns and specific entities from the QUESTION. Do not use pronouns like "it", "they", or generic titles like "the film" or "the city" to replace them.
           - **LEXICAL DISSOLUTION:** You are strictly forbidden from using the common nouns or verbs that form the "logic chain" of the question (e.g., if the question uses "difference", "how many", "population", or "earned", these words are BANNED).
           - **DIVERSE SYNTACTIC EXAMPLES:**
             - *Geography/Stats:* 'What is the population of Tokyo?' -> 'A census count of 37 million residents is currently attributed to Tokyo.' (Banned: "population", "of")
             - *Sports/Count:* 'How many goals did Messi score in 2012?' -> 'Lionel Messi attained a career-best 91 successful strikes throughout the duration of 2012.' (Banned: "goals", "score", "how many")
             - *History/Difference:* 'What is the gap in meters between the Huanglong and Yumen stations?' -> 'The Huanglong station sits exactly 150 meters lower than the Yumen station when comparing their respective altitudes.' (Banned: "gap", "meters", "difference")
             - *Media/Award:* 'Which film earned Ajay Devgn his first Zee Cine Award nomination?' -> 'The Zee Cine Award jury first recognized Ajay Devgn for his performance in [Film].' (Banned: "earned", "nomination")
           - **THE 5-WORD BUFFER:** You must ensure at least 5 non-question words (prepositions, descriptors, or verbs) separate any two Proper Nouns in the claim sentence.
        5. **Group name insertion:** The group name must appear in the text, you can rephrase it to match the grammar or delete parts of name if they are redundant but one should be able to identify the group.
        {hedge_instruction}

        ### OUTPUT FORMAT
        Output ONLY a valid JSON object. Do not include any markdown preamble or post-text.
        {{
          "paragraph_subject": "A 3-5 word title identifying the Anchor and Lens identified in CoT step 1 & 2.",
          "paragraph_text": "A 150-200 word standalone paragraph following all constraints above."
        }}
        """, is_hedged

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
            output_format="string",
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
                # print(f"FAILED TO PARSE JSON FOR UUID: {para_uuid}, skipping sample...")
                continue

            if isinstance(para_content, dict):
                # Store paragraphs
                self.paragraph_dataset[para_uuid] = {
                    'Title': para_content['paragraph_subject'],
                    'Text':  para_content['paragraph_text'],
                    'Confidence': confidence
                }
                sample["Claim paragraph"] = para_uuid

                self.question_dataset.append(sample)
