import json
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
            "The general importance of maintaining precise records and thorough documentation within this specific field.",
            "The broader cultural, historical, or popular fascination with topics of this nature.",
            "The methodological challenges experts or enthusiasts face when categorizing and studying this subject.",
            "The way this general topic reflects wider societal, economic, or artistic trends over time.",
            "The theoretical criteria or foundational definitions that generally govern this domain.",
            "The evolution of public interest and shifting perceptions regarding this broader category.",
            "The practical or academic value of tracking and analyzing detailed data within this specific area.",
            "The universal themes or global commonalities that are often found when discussing this overarching subject.",
            "The typical debates, subjective interpretations, or differing opinions that arise when classifying this topic.",
            "The educational or informational value of preserving exhaustive historical details about this area.",
            "How modern standards, technology, or contemporary thought have changed our understanding of this subject.",
            "The structural, academic, or organizational frameworks typically used to make sense of this field.",
            "The niche communities, fan-bases, or professional circles dedicated to preserving knowledge in this domain.",
            "The abstract concepts of legacy, permanence, and memory as they relate to this type of information.",
            "The complexities involved in curating, filtering, and verifying truths within this specific landscape."
        ]

        # --- Claim Placement Directives ---
        self.PLACEMENT_RULES = {
            "start": "Place the [CLAIM_PLACEHOLDER] in the very FIRST sentence of the paragraph. The rest of the paragraph must organically flow out from this opening statement.",
            "middle": "Place the [CLAIM_PLACEHOLDER] somewhere in the MIDDLE of the paragraph, burying it deeply within the filler text.",
            "end": "Place the [CLAIM_PLACEHOLDER] in the very LAST sentence of the paragraph, building up to it as a concluding thought."
        }

    def _build_user_prompt(self, question: str, claim: str) -> str:
        """
        Constructs the prompt for the claim generation.
        """
        # Randomize placement and fillers
        placement_key = random.choice(list(self.PLACEMENT_RULES.keys()))
        placement_rule = self.PLACEMENT_RULES[placement_key]
        fillers = random.sample(self.FILLER_POOL, 4)

        return f"""You are an expert encyclopedic writer. Your task is to write a paragraph that answers a specific question, but you must format your output strictly as a JSON object.

        QUESTION TO ANSWER: "{question}"
        TARGET ANSWER (THE CLAIM): "{claim}"

        INSTRUCTIONS:
        1. BASE PARAGRAPH: Write a 150-200 word paragraph discussing the broader subject of the question. 
           - You MUST insert the exact string "[CLAIM_PLACEHOLDER]" exactly once where the answer to the question belongs.
           - PLACEMENT RULE: {placement_rule}
           - ABSTRACTION RULE (CRITICAL): The filler text must remain entirely abstract, generalized, and conceptual. Do NOT introduce specific names, dates, external facts, or examples to pad the word count. Talk strictly *about* the domain of the subject.
           - DO NOT reveal the actual answer in the base paragraph. Only use the placeholder.
           - Weave in these themes to reach the word count naturally: 
             1) {fillers[0]}
             2) {fillers[1]}
             3) {fillers[2]}
             4) {fillers[3]}

        2. CERTAINTY VARIATIONS: Write 3 variations of the sentence that will replace [CLAIM_PLACEHOLDER]. Each variation must explicitly state the TARGET ANSWER ("{claim}"), but with different levels of epistemological certainty:
           - HIGH: Absolute, undeniable, verified fact (e.g., "It is an undeniable, proven fact that...", "Definitive records confirm...").
           - MEDIUM: Plausible, widely believed, probable (e.g., "Most experts generally agree that...", "Current evidence suggests...").
           - LOW: Speculative, rumored, highly debated (e.g., "There are unconfirmed rumors that...", "Some fringe theories speculate...").

        OUTPUT FORMAT:
        Output strictly valid JSON. Do not include markdown code blocks (like ```json). 
        Use this exact schema:
        {{
            "base_paragraph": "<The 150-200 word paragraph containing [CLAIM_PLACEHOLDER]>",
            "claim_high": "<The high certainty sentence>",
            "claim_medium": "<The medium certainty sentence>",
            "claim_low": "<The low certainty sentence>"
        }}
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
                claim=sample["Answer"]["Claim"]
            )
            queries.append(query)

        # Execute async calls
        results = await api.async_run(queries, desc="Generating Claim Paragraphs")

        for sample, result in zip(samples, results):
            uuid_high = str(uuid4())
            uuid_medium = str(uuid4())
            uuid_low = str(uuid4())

            try:
                # Clean the result in case the model added markdown despite instructions
                clean_result = result.strip()
                if clean_result.startswith("```json"):
                    clean_result = clean_result[7:]
                if clean_result.endswith("```"):
                    clean_result = clean_result[:-3]

                parsed_json = json.loads(clean_result)

                base_para = parsed_json.get("base_paragraph", "")

                # Assemble the 3 final paragraphs
                para_high = base_para.replace("[CLAIM_PLACEHOLDER]", parsed_json.get("claim_high", ""))
                para_medium = base_para.replace("[CLAIM_PLACEHOLDER]", parsed_json.get("claim_medium", ""))
                para_low = base_para.replace("[CLAIM_PLACEHOLDER]", parsed_json.get("claim_low", ""))

            except (json.JSONDecodeError, AttributeError):
                # Fallback if the LLM fails to output valid JSON
                para_high, para_medium, para_low = "", "", ""

            # Store paragraphs
            self.paragraph_dataset[uuid_high] = para_high
            self.paragraph_dataset[uuid_medium] = para_medium
            self.paragraph_dataset[uuid_low] = para_low

            # Link back to the sample
            sample["Claim paragraphs"] = {
                "High certainty": uuid_high,
                "Medium certainty": uuid_medium,
                "Low certainty": uuid_low
            }

            self.question_dataset.append(sample)