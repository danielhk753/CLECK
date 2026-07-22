"""
Data Verification and LLM Validation Module
------------------------------------------
This module provides a robust validation pipeline for Question-Answering datasets.
It utilizes a multiphase verification process to ensure that generated questions
are answerable by an LLM using provided context paragraphs.

Key Features:
- **Cost-Optimized Pipeline**: Validates 'Evidence' paragraphs first. If a sample fails
  this phase, it is discarded before the 'Claim' validation, significantly reducing
  token consumption and API costs.
- **Strict Verification Logic**: Employs distinct specialized system prompts for
  verbatim extraction and factual reasoning.
- **Semantic Judgment**: Uses an LLM-based judge to compare model outputs against
  ground truth, accounting for synonyms while enforcing strict numerical accuracy.
- **Data Pruning**: Automatically filters and stores valid question-answer pairs
  and maintains a mapping of only the paragraphs required for successful samples.
- **Visual Analytics**: Includes a high-fidelity, dark-mode dashboard for real-time
  monitoring of pass rates and failure distributions.

Classes:
    LLMValidator: The primary engine for reviewing, validating, and auditing datasets.
"""
import re
import unicodedata
from typing import List, Dict
from difflib import SequenceMatcher
from IPython.display import display, HTML
from sentence_transformers import SentenceTransformer, util

from knowledge_maze.tools.api_tools import OpenRouterAPI

EVIDENCE_VERIFIER_PROMPT = """
You are a highly restricted reasoning and extraction function.

You will receive:
1. QUESTION
2. CONTEXT

Your task:
Determine the answer ONLY by reasoning over the explicit facts in the CONTEXT.

Allowed operations:
- counting
- sorting
- comparison
- arithmetic
- filtering
- temporal ordering
- selecting max/min/frequency

CRITICAL RULES:

1. STRICT DATA GROUNDING
Use only the explicit information in the CONTEXT.

2. VALID REASONING ONLY
You may derive the answer only if it follows unambiguously from the provided data.

3. NO ASSUMPTIONS
Do NOT:
- use external knowledge
- infer unstated equivalences
- resolve ambiguity heuristically
- guess missing values

4. ABSTAIN IF ANY DOUBT
If:
- multiple answers are possible
- required information is missing
- the question is ill-defined
- the answer cannot be derived with certainty

5. COHERENCE CHECK
Before answering, verify that the asserted or derived answer is a semantically valid answer to the question.

Return None if:
- the asserted answer is incompatible with the expected answer type
- the answer fills the wrong semantic role
- the paragraph contains internally nonsensical or schema-incoherent mappings

Examples:
- Question asks for operational status, but answer is a geographic unit
- Question asks for number, but answer is an entity
- Question asks for person, but answer is an organization

Output exactly:
None

OUTPUT FORMAT:
Return STRICT JSON only:

{
  "explanation": "One sentence explaining why the answer is certain or why it is not answerable.",
  "answer": "<raw answer or None>"
}

Rules:
- explanation must be exactly one sentence
- answer must contain only the raw answer or None
- no markdown
- no code blocks
- no extra text
"""

CLAIM_VERIFIER_PROMPT = """
You are a highly restricted claim extraction function.

You will receive:
1. QUESTION
2. CONTEXT

Your task:
Determine whether the CONTEXT itself strongly commits to a single answer.

Answer ONLY based on what is explicitly stated or strongly asserted in the CONTEXT.

CRITICAL RULES:

1. TEXT ONLY
Use only what is written in the CONTEXT.

2. IGNORE BRACKET MARKERS ONLY
Text inside double square brackets is removable editorial noise.

Remove the bracketed text as if it were deleted from the sentence, but continue interpreting the surrounding sentence normally.

Example:
Input:
[[ It is likely that ]] John Doe was the first winner.

Interpret as:
John Doe was the first winner.

Do NOT discard or distrust the rest of the sentence merely because bracketed text appears within it.


3. NO MULTI-STEP REASONING
Do NOT:
- count
- compare multiple entries
- sort values
- compute statistics
- infer from tables or latent structure
- derive answers from implicit data relationships

4. BELIEF CRITERION (The Strict Anchor Test)
Answer ONLY if the provided CONTEXT would make a careful reader believe the answer with no reasonable doubt. You must apply the following anchors:

- METRIC & UNIT FIDELITY: If the question specifies a unit (e.g., "meters", "kg", "USD"), the context MUST explicitly mention that exact unit. A numerical match without a unit match constitutes reasonable doubt.
- TEMPORAL ANCHORING: If the question includes a specific year, timespan, or era (e.g., "1995", "2005-2010"), the context MUST explicitly contain that temporal marker. If the timeframe is missing in the context, the answer is uncertain.
- IDENTITY ALIGNMENT: The context must explicitly link the Subject Entity to the Value. We never assume a fact is "evergreen" or applies to the question's subject by proximity alone.

5. ALTERNATIVE INTERPRETATION CHECK
Before answering, actively check whether another reasonable interpretation of the QUESTION or CONTEXT could lead to a different answer.

If ANY reasonable alternative interpretation exists, the answer is uncertain.

Examples of uncertainty:
- multiple plausible entities
- ambiguous scope
- unclear reference target
- wording that supports more than one interpretation

In these cases, answer must be:
None

6. ABSTAIN IF ANY DOUBT
If the answer is:
- not explicitly or strongly asserted
- ambiguous
- contradicted elsewhere in the text
- merely implied through data rather than stated
- vulnerable to another reasonable interpretation

answer must be:
None

7. COHERENCE CHECK
Before answering, verify that the asserted or derived answer is a semantically valid answer to the question.

Return None if:
- the asserted answer is incompatible with the expected answer type
- the answer fills the wrong semantic role
- the paragraph contains internally nonsensical or schema-incoherent mappings

Examples:
- Question asks for operational status, but answer is a geographic unit
- Question asks for number, but answer is an entity
- Question asks for person, but answer is an organization

A confidently written but semantically incoherent answer must be treated as unanswerable.


OUTPUT FORMAT:
Return STRICT JSON only:

{
  "explanation": "One sentence explaining why the answer is certain or why it is uncertain.",
  "answer": "<raw answer or None>"
}

Rules:
- explanation must be exactly one sentence
- answer must contain only the raw answer or None
- no markdown
- no code blocks
- no extra text
"""


class LLMValidator:
    """
    Uses OpenRouterAPI to validate questions against context and ground truth.
    """

    def __init__(self, model: str = "google/gemini-3-flash-preview", verbose: bool = True):
        # We set the output_format to 'bool' so the API returns True/False directly

        self.model = model
        self.claim_api = OpenRouterAPI(
            system_prompt=CLAIM_VERIFIER_PROMPT,
            model=self.model,
            temperature=0.0,
            output_format='cot'
        )
        self.evidence_api = OpenRouterAPI(
            system_prompt=EVIDENCE_VERIFIER_PROMPT,
            model=self.model,
            temperature=0.0,
            output_format='cot'
        )

        self.verbose = verbose

        self.total_samples = 0
        self.wrong_answers = []
        self.abstained_answers = []

        self.typo_threshold = 0.9
        self.iou_threshold = 1.0
        self.embedding_threshold = 0.8
        self.validate_model = SentenceTransformer('all-MiniLM-L6-v2')

        self.failed_claim = []
        self.failed_evidence = []
        self.valid_data = []  # Stores fully validated samples
        self.approved_paragraphs = {}  # Stores only the paragraphs used by valid_data

    @staticmethod
    def _prepare_prompt(sample: Dict[str, str]) -> str:
        """Formats the data sample into a clear evaluation prompt."""
        return (
            f"CONTEXT: {sample.get('context')}\n"
            f"QUESTION: {sample.get('question')}\n"
            f"ANSWER: "
        )

    async def validate_samples(self, samples: List[Dict[str, str]], ver_type: str) -> List[bool]:
        """
        Processes a list of samples and returns a list of booleans.
        """
        prompts = [self._prepare_prompt(s) for s in samples]
        true_answers = [s.get('answer') for s in samples]

        # async_run handles the batching, progress bar, and concurrency
        if ver_type == 'claim':
            results = await self.claim_api.async_run(prompts, desc="Validating Samples")
        elif ver_type == 'evidence':
            results = await self.evidence_api.async_run(prompts, desc="Validating Samples")
        else:
            raise ValueError

        results_pairs = []
        model_explanations = []

        for true_ans, model_ans in zip(true_answers, results):
            try:
                ans = model_ans.answer
                exp = model_ans.explanation
            except (AttributeError, TypeError):
                # Fallback if the API returns None or an object missing these attributes
                ans = "API_ERROR"
                exp = "Failed to parse structured output."

            results_pairs.append((true_ans, ans))
            model_explanations.append(exp)

        final_summary = [self.are_equal(*pair) for pair in results_pairs]

        true_count = 0
        false_count = 0
        none_count = 0

        for a, b, s, ex in zip(results_pairs, final_summary, samples, model_explanations):
            if b:
                true_count += 1
            elif a[1] == 'None':
                none_count += 1
                self.abstained_answers.append(
                    {
                        'Question': s['question'],
                        'Intended Answer': s['answer'],
                        'Model Answer': a[1],
                        'Explanation': ex,
                        'Context': s['context']['Text']
                    }
                )
            else:
                false_count += 1
                self.wrong_answers.append(
                    {
                        'Question': s['question'],
                        'Intended Answer': s['answer'],
                        'Model Answer': a[1],
                        'Explanation': ex,
                        'Context': s['context']['Text']
                    }
                )

        print('----------------')
        print(f"Summary:")
        print(f"Model answered correct: {true_count}")
        print(f"Model answered wrong: {false_count}")
        print(f"Model abstained: {none_count}")
        print('----------------')

        return final_summary

    async def validate_evidence_only(self, question_data: List[Dict], paragraphs: Dict) -> List[Dict]:
        """
        GATE 1: Tests if the LLM can derive the answer from the Evidence paragraph alone.
        """
        self.total_samples = len(question_data)
        self.failed_evidence = []
        passed_evidence_samples = []

        if not question_data:
            return passed_evidence_samples

        evidence_input = [{'question': s.get('Question'),
                           'answer': s.get('Answer').get('Evidence'),
                           'context': paragraphs[s.get('Evidence paragraph')]} for s in question_data]

        evidence_results = await self.validate_samples(evidence_input, ver_type='evidence')

        # Filter out failed evidence samples
        for s, evidence_ans in zip(question_data, evidence_results):
            if evidence_ans:
                passed_evidence_samples.append(s)
            else:
                self.failed_evidence.append(s)

        return passed_evidence_samples

    async def validate_claim_only(self, question_data: List[Dict], paragraphs: Dict) -> List[Dict]:
        """
        GATE 2: Tests if the LLM can extract the answer from the newly generated Claim paragraph.
        Runs final cleanup and generates the dashboard.
        """
        self.failed_claim = []
        self.valid_data = []

        if not question_data:
            self.display_summary()
            return self.valid_data

        claim_input = [{'question': s.get('Question'),
                        'answer': s.get('Answer').get('Claim'),
                        'context': paragraphs[s.get('Claim paragraph')]} for s in question_data]

        claim_results = await self.validate_samples(claim_input, ver_type='claim')

        # Filter final valid samples
        for s, claim_ans in zip(question_data, claim_results):
            if claim_ans:
                self.valid_data.append(s)
            else:
                self.failed_claim.append(s)

        # ---------------------------------------------------------
        # FINAL PHASE: SAVE APPROVED PARAGRAPHS
        # ---------------------------------------------------------
        if self.valid_data:
            # Get all unique paragraph indices used by the approved questions
            valid_indices = set()
            for s in self.valid_data:
                valid_indices.add(s.get('Claim paragraph'))
                valid_indices.add(s.get('Evidence paragraph'))

            # Save only the relevant paragraphs
            self.approved_paragraphs = {idx: paragraphs[idx] for idx in valid_indices if idx in paragraphs}

        if self.verbose:
            self.display_summary()
        return self.valid_data

    def clean(self, text: str):
        """Minimalist cleaning: lowercase, no accents, no list markers."""
        text = str(text).lower()
        text = "".join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
        # Remove common list symbols and markers
        text = re.sub(r'(?:\d+[\.\)]|\(\d+\)|[†\-\*])\s*', ' ', text)
        # Replace punctuation with spaces to avoid merging words
        text = re.sub(r'[,;."]', ' ', text)
        # Remove common stop words (and, the, a, is, of, in, etc.)
        # Note the \b (word boundary) to ensure we don't accidentally remove "and" from "island"
        stop_words = r'\b(and|the|a|an|is|of|in|to|with|for|it|on|as|at|by|this)\b'
        text = re.sub(stop_words, ' ', text)
        return " ".join(text.split())

    def get_set(self, text: str):
        """Splits into a set of unique words/terms."""
        cleaned = self.clean(text)
        return set(cleaned.split())

    def _is_numeric(self, s: str):
        """
        Returns a float only if the string is primarily a number.
        Prevents "1. Luanda" from being treated as the number 1.0.
        """
        # CRITICAL: Force cast to string to handle int/float inputs
        s = str(s)

        # 1. Clean the string of commas and extra whitespace
        cleaned = s.replace(',', '').strip()

        # 2. Match the number
        match = re.search(r"[-+]?\d*\.\d+|\d+", cleaned)
        if not match:
            return None

        # 3. Validation: Is there too much 'text' left?
        # We remove the matched number and common units/signs
        remaining_text = cleaned.replace(match.group(), '', 1)
        # Remove units like 'MW', '%', 'm', or 'kg' and punctuation
        remaining_text = re.sub(r'[a-zA-Z%$\. ]', '', remaining_text)

        # If there's still significant text left, it's a list or a name, not a number
        if len(remaining_text) > 0:
            return None

        return float(match.group())

    def calculate_iou(self, str_a: str, str_b: str):
        # --- NUMERICAL OVERRIDE ---
        # If both contain numbers, they MUST be identical.
        num_a, num_b = self._is_numeric(str_a), self._is_numeric(str_b)
        if num_a is not None or num_b is not None:
            return 1.0 if num_a == num_b else 0.0

        set_a = self.get_set(str_a)
        set_b = self.get_set(str_b)

        if not set_a and not set_b: return 1.0
        if not set_a or not set_b: return 0.0

        # --- FUZZY INTERSECTION ---
        intersection_count = 0
        temp_set_b = list(set_b)

        for item_a in set_a:
            best_score = 0
            best_match_idx = -1

            for i, item_b in enumerate(temp_set_b):
                # Quick exact check
                if item_a == item_b:
                    score = 1.0
                else:
                    score = SequenceMatcher(None, item_a, item_b).ratio()

                if score > best_score:
                    best_score = score
                    best_match_idx = i

            if best_score >= self.typo_threshold:
                intersection_count += 1
                if best_match_idx != -1:
                    temp_set_b.pop(best_match_idx)  # Don't double-count matches

        union_count = len(set_a) + len(set_b) - intersection_count
        return intersection_count / union_count

    def get_semantic_similarity(self, a: str, b: str):
        """Tier 3: Check if the meaning is the same even if words differ."""
        try:
            embedding_a = self.validate_model.encode(a, convert_to_tensor=True)
            embedding_b = self.validate_model.encode(b, convert_to_tensor=True)
            return util.pytorch_cos_sim(embedding_a, embedding_b).item()
        except:
            return 0.0

    def are_equal(self, a: str, b: str) -> bool:
        # TIER 1: Numerical Check
        num_a, num_b = self._is_numeric(a), self._is_numeric(b)
        if num_a is not None or num_b is not None:
            return num_a == num_b

        # TIER 2: IoU Check
        iou_score = self.calculate_iou(a, b)
        if iou_score >= self.iou_threshold:
            return True

        # TIER 3: Semantic Fallback
        # We only run this if IoU fails, saving compute.
        semantic_score = self.get_semantic_similarity(a, b)
        return semantic_score >= self.embedding_threshold

    def display_summary(self):
        """Displays a stylized dark-mode dashboard of the validation results."""
        passed = len(self.valid_data)
        fail_ev = len(self.failed_evidence)
        fail_cl = len(self.failed_claim)
        rate = (passed / self.total_samples * 100) if self.total_samples > 0 else 0

        # Color logic for the progress bar and percentage
        accent_color = "#4caf50" if rate >= 80 else "#ff9800" if rate >= 50 else "#f44336"

        dashboard_html = f"""
        <div style="background-color: #1e1e1e; color: #e0e0e0; padding: 20px; border-radius: 12px; 
                    border: 1px solid #333; font-family: 'Segoe UI', sans-serif; width: 400px; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.5); margin: 10px 0;">

            <h3 style="color: #bb86fc; margin-top: 0; margin-bottom: 5px;">
                📊 Approved Data Summary:
            </h3>
            
            <div style="color: #9aa0a6; font-size: 14px; margin-bottom: 15px;">
                Model Could Answer Both paragraphs
            </div>


            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="font-size: 0.9em; color: #aaa;">Overall Pass Rate</span>
                <span style="font-weight: bold; color: {accent_color};">{rate:.1f}%</span>
            </div>

            <div style="background-color: #333; border-radius: 10px; height: 8px; width: 100%; margin-bottom: 20px;">
                <div style="background-color: {accent_color}; height: 8px; width: {rate}%; border-radius: 10px; transition: width 0.5s;"></div>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px;">
                <div style="background: #252525; padding: 10px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 0.8em; color: #aaa;">Total Samples</div>
                    <div style="font-size: 1.2em; font-weight: bold; color: #03dac6;">{self.total_samples}</div>
                </div>
                <div style="background: #252525; padding: 10px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 0.8em; color: #aaa;">Passed</div>
                    <div style="font-size: 1.2em; font-weight: bold; color: #4caf50;">{passed}</div>
                </div>
            </div>

            <div style="font-size: 0.9em; color: #aaa; margin-bottom: 10px;">Failures Breakdown</div>
            <div style="display: flex; flex-direction: column; gap: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center; background: #2d2d2d; padding: 6px 12px; border-radius: 6px;">
                    <span style="font-size: 0.85em;">❌ Failed Evidence Paragraphs</span>
                    <span style="color: #f44336; font-weight: bold;">{fail_ev}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; background: #2d2d2d; padding: 6px 12px; border-radius: 6px;">
                    <span style="font-size: 0.85em;">❌ Failed Claim Paragraphs</span>
                    <span style="color: #ff9800; font-weight: bold;">{fail_cl}</span>
                </div>
            </div>
        </div>
        """
        display(HTML(dashboard_html))
