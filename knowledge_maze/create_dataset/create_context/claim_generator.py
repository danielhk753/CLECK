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

import random
from uuid import uuid4
from typing import List, Dict, Any

from knowledge_maze.tools.api_tools import OpenRouterAPI


class ClaimParagraph:
    def __init__(self):
        self.question_dataset = []
        self.paragraph_dataset = {}

        # --- Paragraph Filler Pool (Knowledge-Rich) ---
        self.FILLER_POOL = [
            # --- Category: Academic & Methodological ---
            "The structural, academic, or organizational frameworks typically used to make sense of this field.",
            "The theoretical criteria or foundational definitions that generally govern this domain.",
            "The methodological challenges experts face when categorizing and studying this subject.",
            "The typical debates, subjective interpretations, or differing opinions that arise during classification.",
            "The complexities involved in curating, filtering, and verifying truths within this specific landscape.",

            # --- Category: Cultural & Societal ---
            "The broader cultural, historical, or popular fascination with topics of this nature.",
            "The way this general topic reflects wider societal, economic, or artistic trends over time.",
            "The evolution of public interest and shifting perceptions regarding this broader category.",
            "The universal themes or global commonalities found when discussing this overarching subject.",
            "The niche communities, fan-bases, or professional circles dedicated to preserving this knowledge.",

            # --- Category: Practical & Industrial ---
            "The practical or academic value of tracking and analyzing detailed data within this specific area.",
            "How modern standards, technology, or contemporary thought have changed our understanding of this subject.",
            "The general importance of maintaining precise records and thorough documentation within this field.",
            "Current industry standards and the protocols for managing such high-granularity information.",
            "The logistical overhead required to maintain long-term databases in a rapidly shifting professional environment.",

            # --- Category: Abstract & Philosophical ---
            "The abstract concepts of legacy, permanence, and memory as they relate to this type of information.",
            "The informational value of preserving exhaustive historical details for future generations.",
            "The inherent tension between subjective narrative and objective data points in archival science.",
            "The philosophical debate over whether certain categories of information should remain immutable or evolve.",
            "The way specific knowledge fragments contribute to the collective consciousness of a professional domain.",

            # --- Category: "Subject-Specific" Flavor (Neutral) ---
            "The recurring patterns that emerge when one examines the sub-components of this field in isolation.",
            "The subtle nuances that distinguish elite-level contributions from standard entries in this domain.",
            "The environmental or contextual factors that often influence how these specific events are recorded.",
            "The interplay between established precedents and the introduction of novel, disruptive information.",
            "The necessity of cross-referencing multiple data streams to ensure a holistic view of the subject matter."
        ]

        # --- Claim Placement Directives ---
        self.PLACEMENT_RULES = {
            "start": "Place the claim in the very FIRST sentence of the paragraph. The rest of the paragraph must organically flow out from this opening statement.",
            "middle": "Place the claim somewhere in the MIDDLE of the paragraph, burying it deeply within the filler text.",
            "end": "Place the claim in the very LAST sentence of the paragraph, building up to it as a concluding thought."
        }

        # --- Epistemic Marker Syntax Dictionary ---
        self.CERTAINTY_COMPONENTS = {
            "High": [
                # Sources/Attributions
                "According to definitive archival records,",
                "Verified historical data confirms that",
                "Primary source documentation establishes that",
                "Official consensus indicates that",
                "Rigorous empirical analysis proves that",
                "Peer-reviewed findings demonstrate that",
                "As established by comprehensive research,",
                "Governmental registries clearly state that",
                # Boosters
                "It is an undeniable, verified fact that",
                "It is clearly the case that",
                "It has been proven beyond doubt that",
                "There is no question that",
                "It is an absolute certainty that",
                "Records unequivocally show that",
                "The evidence leaves no room for debate:",
                "It is a matter of public record that",
                "The reality is fundamentally that",
                "Crucially, it is established that",
                "Essentially, the truth is that",
                "Without exception, the data shows"
            ],
            "Medium": [
                # Sources/Attributions
                "Standard industry reports suggest that",
                "Most academic models propose that",
                "Commonly cited evidence indicates that",
                "General observations tend to show that",
                "According to prevailing theories,",
                "Mainstream literature often notes that",
                "Based on typical interpretations,",
                "Current working papers imply that",
                # Hedges/Probabilistic
                "It is generally accepted that",
                "It appears probable that",
                "Most experts believe that",
                "It is widely considered that",
                "The current understanding is that",
                "It is likely the case that",
                "Evidence usually points to the fact that",
                "There is a strong case for",
                "For the most part, it seems that",
                "In many instances, it is noted that",
                "Regularly, it is assumed that",
                "It is reasonable to suggest that"
            ],
            "Low": [
                # Sources/Attributions
                "Some fringe theories speculate that",
                "Unconfirmed rumors suggest that",
                "Preliminary, isolated studies hint that",
                "According to anecdotal accounts,",
                "Early, non-peer-reviewed notes mention that",
                "Speculative reports indicate that",
                "According to disputed testimony,",
                "As noted in obscure sources,",
                # Hedges
                "It might possibly be true that",
                "It could conceivably be the case that",
                "Perhaps it may tentatively be that",
                "There are whispers that",
                "It is far from certain, but",
                "Some have hypothesized that",
                "One possible explanation is that",
                "It is a matter of some debate whether",
                "There is a slight chance that",
                "A controversial viewpoint is that",
                "Under some interpretations, maybe",
                "It has been suggested, though not proven, that"
            ]
        }

    def _build_user_prompt(self, question: str, claim: str, group: str) -> str:
        """
        Constructs the prompt for the claim generation.
        """
        # Randomize placement and fillers
        placement_key = random.choice(list(self.PLACEMENT_RULES.keys()))
        placement_rule = self.PLACEMENT_RULES[placement_key]
        fillers = random.sample(self.FILLER_POOL, 4)
        prefixes = [random.choice(self.CERTAINTY_COMPONENTS['High']),
                    random.choice(self.CERTAINTY_COMPONENTS['Medium']),
                    random.choice(self.CERTAINTY_COMPONENTS['Low'])]

        return f"""You are an expert encyclopedic writer. Your task is to write a paragraph that answers a specific question.

        QUESTION TO ANSWER: "{question}"
        TARGET ANSWER (THE CLAIM): "{claim}"

        INSTRUCTIONS:
        1. BASE PARAGRAPH: Write a 150-200 word paragraph discussing the broader subject of the question: {group}. 
           - You MUST make sure there is a sentence containing a claim to the answer.
           - PLACEMENT RULE: {placement_rule}
           - ABSTRACTION RULE (CRITICAL): The filler text must remain entirely abstract, generalized, and conceptual. 
             Do NOT introduce specific names, dates, external facts, or examples to pad the word count, other than entities explicitly mentioned in the question itself. 
             Talk strictly *about* the domain of the subject and the group in a way unrelated to the question itself.
           - Weave in these themes to reach the word count naturally: 
             1) {fillers[0]}
             2) {fillers[1]}
             3) {fillers[2]}
             4) {fillers[3]}

        2. CERTAINTY VARIATIONS: Take the following three prefixes that indicates different levels of certainty: {prefixes}.
           - Insert these in the main claim sentence in the paragraph in the following way:
                - "[prefix A / prefix B / prefix C] with the rest of the claim sentence leading to the exact answer provided."
                - The prefixes must appear with ' / ' between them and inside these brackets '[]'.
           - You can change the grammar a bit to make sure every one of them is acceptable as a natural sounding sentence.
           - Make sure to adjust upper/lower-case if the prefixes are in the start or middle of the sentence.

        OUTPUT FORMAT: 
        Output strictly a string paragraph, with the relevant claim sentence inside of it and the prefixes in the requested format above.
        """

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
        for sample in samples:
            query = self._build_user_prompt(
                question=sample["Question"],
                claim=sample["Answer"]["Claim"],
                group=sample["Group name"]
            )
            queries.append(query)

        # Execute async calls
        results = await api.async_run(queries, desc="Generating Claim Paragraphs")

        for sample, result in zip(samples, results):

            para_uuid = str(uuid4())
            para_content = result

            # Store paragraphs
            self.paragraph_dataset[para_uuid] = para_content
            sample["Claim paragraph"] = para_uuid

            self.question_dataset.append(sample)