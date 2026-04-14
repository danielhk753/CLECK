import asyncio
import random
import re
from typing import List, Dict, Any, Optional

# Assuming api_tools.py and judge_utils.py are in the same directory or accessible
from knowledge_maze.tools.api_tools import gptAPI, ListResponse, StringResponse
from knowledge_maze.evaluation_old.llm_judge import batch_judge_answers  # This needs to be imported if not already.


# --- LLM API Client Initializations ---

# Model for generating the first two alternative answers (plausible but incorrect)
alt_answer_generator_model = gptAPI(
    system_prompt="""You are an expert at generating plausible, but factually incorrect, alternative answers for factual sub-questions.
                     Your goal is to provide exactly TWO DIFFERENT, single, concise entities or phrases as alternative answers.
                     These answers should be *plausible* in a counter-factual scenario but fundamentally *incorrect* compared to the true answer.
                     Do NOT use your memory or real-world knowledge to verify correctness; focus on plausibility within a counter-factual frame.
                     Do NOT directly copy from the original context. Do NOT use the original answer.
                     Provide ONLY the two distinct alternative answers as a Python list of strings.
                     """,
    output_format='list',  # Expecting a list of strings (the two alternatives)
    temperature=0.8,  # Moderate to high temperature for diverse, plausible alternatives
    max_concurrent_requests=50,
    max_rpm=4900
)

# Model for generating subsequent sub-answers in a counter-factual chain
sub_answer_chain_model = gptAPI(
    system_prompt="""You are a logical reasoner, tasked with building a chain of answers in a counter-factual scenario.
                     Given a previous sub-question and its (potentially counter-factual) answer, and a new sub-question,
                     your task is to provide a plausible (though possibly factually incorrect in the real world) answer to the new sub-question.
                     This new answer must be *logically consistent* with the preceding information and the question asked, even if the preceding info is incorrect.
                     Your answer must be a single, concise entity or phrase.
                     Do NOT use external or real-world knowledge. Only use the provided 'previous' context to logically infer a plausible answer for the 'current' sub-question.
                     If the previous context makes the current question unanswerable plausibly, respond clearly that the answer cannot be determined.
                     """,
    output_format='string',  # Expecting a single string answer
    temperature=0.5,  # Balance between consistency and some creativity for new plausible answers
    max_concurrent_requests=50,
    max_rpm=4900
)

# Model for generating new paragraphs to support specific (sub-question, sub-answer) pairs
paragraph_generator_model = gptAPI(
    system_prompt="""You are a skilled writer tasked with crafting new factual paragraphs.
                     Your goal is to create a CONCISE paragraph that explicitly provides information
                     to answer a given sub-question, leading to a specific desired answer.
                     Ensure the paragraph is self-contained and directly answers the sub-question with the provided answer.
                     The paragraph should ideally be a few sentences long.
                     Do NOT use your memory or external knowledge beyond the instruction.
                     """,
    output_format='string',  # Expecting a single string paragraph
    temperature=0.3,  # Low temperature for factual generation
    max_concurrent_requests=50,
    max_rpm=4900
)

# NEW: LLM Client for rephrasing sub-questions
question_rephraser_model = gptAPI(
    system_prompt="""You are an expert at rephrasing short, fact-based statements into natural language questions.
                     If the statement refers to '#N' (where N is a number), you should replace '#N' with the provided 'Entity'.
                     Your output should be ONLY the rephrased natural language question, ending with a question mark.
                     Examples:
                     'Let the Good Times Roll >> director' -> 'Who directed Let the Good Times Roll?'
                     '#1 >> place of birth', Entity: 'Robert Abel' -> 'Where was Robert Abel born?'
                     '#2 >> located in the administrative territorial entity', Entity: 'Cleveland' -> 'In which administrative territorial entity is Cleveland located?'
                     '#3 >> shares border with', Entity: 'Rowan County' -> 'What does Rowan County share a border with?'
                     """,
    output_format='string',
    temperature=0.1,  # Keep it consistent and non-creative
    max_concurrent_requests=50,
    max_rpm=4900
)

# LLM for answering questions from context during evaluation_old (high temperature for diversity)
qa_context_eval_model = gptAPI(
    system_prompt="""You are a logical reasoner. Your task is to answer the question using ONLY the information provided in the context below.

    Rules:
    - Do NOT use any external or real-world knowledge.
    - Only rely on statements explicitly made in the context, even if they are factually incorrect or incomplete.
    - If the context does not contain enough information to answer, respond clearly that the answer cannot be determined.
    - Your answer must be a single entity or phrase, taken directly from or unambiguously supported by the context.
    """,
    output_format='list',  # We want a list of answers (even if it's just one, for consistency)
    temperature=1.5,  # High temperature for diverse responses for self-consistency analysis
    max_concurrent_requests=100,
    max_rpm=4900
)


# --- Dataset Structuring and Generation Functions ---

async def create_alternative_first_sub_answers(
        original_decomposition_step: Dict[str, Any],
        original_paragraph_text: str
) -> List[str]:
    """
    Generates two distinct, plausible, but factually incorrect alternative answers for the first sub-question.
    """
    question = original_decomposition_step['question']
    original_answer = original_decomposition_step['answer']

    prompt = f"""Question: {question}
    Original Answer: {original_answer}
    Original Context (for background only, DO NOT copy from it):
    {original_paragraph_text}

    Provide exactly two distinct, plausible, but factually INCORRECT alternative answers for the question.
    Each answer must be a single, concise entity or phrase and should NOT be '{original_answer}'.
    Provide your answers as a Python list of two strings.
    """

    generated_alts = await alt_answer_generator_model.run_in_thread(prompt)

    # Ensure we get exactly two distinct strings, handle potential LLM errors
    if not isinstance(generated_alts, list) or len(generated_alts) != 2:
        print(f"Warning: Alt answer generator returned unexpected format: {generated_alts}. Returning placeholders.")
        # Fallback if LLM doesn't produce list of two, e.g., run again or return placeholders
        return ["Alternative Answer A", "Alternative Answer B"]  # Provide distinct placeholders

    return [alt.strip() for alt in generated_alts]


async def generate_chained_sub_answers(
        original_decomposition_structure: List[Dict[str, Any]],
        first_sub_question_new_answer: str,
        generation_label: str  # e.g., 'alt_1' or 'alt_2' for logging
) -> List[Dict[str, str]]:
    """
    Generates a full chain of counter-factual sub-answers based on an alternative first sub-answer.
    The questions in the returned chain are still in their raw format (e.g., '#1 >> ...').
    """
    new_decomposition_chain_raw = []

    # First step uses the provided alternative answer
    first_step = original_decomposition_structure[0]
    new_decomposition_chain_raw.append({
        'question': first_step['question'],  # Raw question format
        'answer': first_sub_question_new_answer
    })

    # Generate subsequent sub-answers in a chained fashion
    for i in range(1, len(original_decomposition_structure)):
        prev_step_in_chain = new_decomposition_chain_raw[-1]
        current_original_step = original_decomposition_structure[i]

        prompt = f"""Previous Sub-Question (raw format): {prev_step_in_chain['question']}
        Previous Sub-Answer: {prev_step_in_chain['answer']}

        Current Sub-Question (raw format): {current_original_step['question']}

        Given the previous sub-question and its (potentially counter-factual) answer, what is a plausible
        (though possibly factually incorrect) answer to the Current Sub-Question?
        Your answer must be a single, concise entity or phrase.
        Do NOT use external or real-world knowledge. Only use the provided 'previous' context to logically infer a plausible answer for the 'current' sub-question.
        If the previous information makes the current question unanswerable plausibly, state 'Cannot Answer'.
        """

        new_sub_answer = await sub_answer_chain_model.run_in_thread(prompt)

        new_decomposition_chain_raw.append({
            'question': current_original_step['question'],  # Raw question format
            'answer': new_sub_answer.strip()
        })

    return new_decomposition_chain_raw


async def rephrase_question_text(raw_question_template: str, entity_for_placeholder: Optional[str] = None) -> str:
    """
    Rephrases a raw sub-question text (potentially containing '#X' placeholder) into a natural language question.
    """
    # Check for presence of '#N' pattern in the raw question template
    if re.search(r'#\d+', raw_question_template):
        if entity_for_placeholder is None:
            # Fallback if an entity is expected but not provided
            return f"Rephrase: {raw_question_template.replace(re.search(r'#\d+', raw_question_template).group(0), '...')}?"
        prompt = f"""Raw Statement: {raw_question_template}
        Entity to substitute for '#N' in the Raw Statement: {entity_for_placeholder}
        Rephrased Question:"""
    else:
        prompt = f"""Raw Statement: {raw_question_template}
        Rephrased Question:"""

    rephrased = await question_rephraser_model.run_in_thread(prompt)
    return rephrased.strip()


async def format_decomposition_chain_async(raw_decomposition_chain: List[Dict[str, str]]) -> List[str]:
    """
    Formats a raw decomposition chain (list of dicts with raw questions) into a list of strings:
    'Natural language question? --> answer'
    Intelligently substitutes #X placeholders using the rephrasing LLM.
    """
    formatted_chain = []

    for i, step in enumerate(raw_decomposition_chain):
        raw_question = step['question']
        current_answer = step['answer']

        entity_for_substitution = None
        # If this step refers to a previous answer (e.g., #1, #2, #3)
        if re.search(r'#\d+', raw_question) and i > 0:
            entity_for_substitution = raw_decomposition_chain[i - 1]['answer']

        rephrased_q = await rephrase_question_text(raw_question, entity_for_substitution)
        formatted_chain.append(f"{rephrased_q} --> {current_answer}")

    return formatted_chain


async def create_supporting_paragraph(
        sub_question: str,  # This should be the raw sub_question text for clarity to paragraph_generator
        desired_answer: str,
        exclude_text: str = ""  # To avoid copying from original, if any
) -> str:
    """
    Generates a new paragraph that explicitly answers a sub-question with a desired answer.
    """
    prompt = f"""Create a concise, factual paragraph that directly answers the following sub-question
    with the given desired answer.
    The paragraph should contain all information needed for the answer, and only that information.
    The paragraph should be a few sentences long.

    Sub-Question: {sub_question}
    Desired Answer: {desired_answer}
    """
    if exclude_text:  # Not strictly necessary if prompts are strong enough, but good for safety
        prompt += f"\n\nDO NOT use any content from this text: {exclude_text}"

    generated_paragraph = await paragraph_generator_model.run_in_thread(prompt)
    return generated_paragraph.strip()


async def generate_paragraphs_for_decomposition_chain(
        decomposition_chain_raw: List[Dict[str, str]],  # Expects raw dict format for question/answer
        original_paragraphs_map: Dict[int, str]  # Map idx to text, for exclusion if needed
) -> List[str]:
    """
    Generates a list of paragraphs, one for each (sub-question, sub-answer) pair in a decomposition chain.
    """
    paragraphs = []
    for step in decomposition_chain_raw:
        para_text = await create_supporting_paragraph(
            step['question'],  # Pass the raw question format to the paragraph generator
            step['answer'],
            # exclude_text=original_paragraphs_map.get(step.get('paragraph_support_idx', -1), "") # If original was passed
        )
        paragraphs.append(para_text)
    return paragraphs


async def process_dataset_entry(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Processes a single dataset entry to generate alternative contexts and answers.
    """
    processed_entry = {
        'id': entry['id'],
        'question': entry['question'],
        'answer': {'original': entry['answer'], 'alt_1': '', 'alt_2': ''},
        'original decomposition': [],
        'alt 1 decomposition': [],
        'alt 2 decomposition': [],
        'distractor paragraphs': [],
        'original paragraphs': [],
        'alt 1 paragraphs': [],
        'alt 2 paragraphs short': [],
        'alt 2 paragraphs blocked': [],
        'model answer statistics': {}
    }

    # 1. Separate original and distractor paragraphs
    original_paragraph_texts_map = {}
    original_supporting_paragraphs_list = []

    # Collect indices of paragraphs used in the original decomposition
    original_decomposition_raw = entry['question_decomposition']
    if not original_decomposition_raw:
        print(f"Skipping entry {entry['id']}: No question decomposition found.")
        return None

    original_decomp_p_idxs = {step['paragraph_support_idx'] for step in original_decomposition_raw if
                              'paragraph_support_idx' in step}

    for p in entry['paragraphs']:
        if p['idx'] in original_decomp_p_idxs:
            original_supporting_paragraphs_list.append(p['paragraph_text'])
            original_paragraph_texts_map[p['idx']] = p['paragraph_text']
        else:
            processed_entry['distractor paragraphs'].append(p['paragraph_text'])
    processed_entry['original paragraphs'] = original_supporting_paragraphs_list

    # Format the original decomposition immediately for storage
    # We need to transform the original entry's decomposition to match the question/answer dict format first
    original_decomp_for_formatting = [{'question': step['question'], 'answer': step['answer']} for step in
                                      original_decomposition_raw]
    processed_entry['original decomposition'] = await format_decomposition_chain_async(original_decomp_for_formatting)

    # 2. Generate 2 alternative plausible (but incorrect) answers for the first sub-question
    first_sub_question_step = original_decomposition_raw[0]
    first_original_para_text = original_paragraph_texts_map.get(
        first_sub_question_step.get('paragraph_support_idx', -1), "")

    alt_sub_answers_list = await create_alternative_first_sub_answers(
        first_sub_question_step,
        first_original_para_text
    )
    if len(alt_sub_answers_list) < 2:
        print(f"Warning: Could not generate enough distinct alternative sub-answers for {entry['id']}. Skipping entry.")
        return None

    alt_1_first_sub_answer = alt_sub_answers_list[0]
    alt_2_first_sub_answer = alt_sub_answers_list[1]

    # 3. Generate full alternative decomposition chains (raw dict format first)
    alt_1_decomposition_chain_raw = await generate_chained_sub_answers(
        original_decomposition_raw,
        alt_1_first_sub_answer,
        'alt_1'
    )
    processed_entry['alt 1 decomposition'] = await format_decomposition_chain_async(alt_1_decomposition_chain_raw)

    if alt_1_decomposition_chain_raw:
        processed_entry['answer']['alt_1'] = alt_1_decomposition_chain_raw[-1]['answer']
    else:
        processed_entry['answer']['alt_1'] = 'Cannot Determine'

    alt_2_decomposition_chain_raw = await generate_chained_sub_answers(
        original_decomposition_raw,
        alt_2_first_sub_answer,
        'alt_2'
    )
    processed_entry['alt 2 decomposition'] = await format_decomposition_chain_async(alt_2_decomposition_chain_raw)

    if alt_2_decomposition_chain_raw:
        processed_entry['answer']['alt_2'] = alt_2_decomposition_chain_raw[-1]['answer']
    else:
        processed_entry['answer']['alt_2'] = 'Cannot Determine'

    # 4. Generate paragraphs for these alternative decomposition chains

    # --- Alt 1 Paragraphs ---
    processed_entry['alt 1 paragraphs'] = await generate_paragraphs_for_decomposition_chain(
        alt_1_decomposition_chain_raw,
        original_paragraph_texts_map
    )

    # --- Alt 2 Paragraphs (Short Version) ---
    alt_2_full_paragraphs = await generate_paragraphs_for_decomposition_chain(
        alt_2_decomposition_chain_raw,
        original_paragraph_texts_map
    )

    alt_2_short_paragraphs = []
    if len(alt_2_decomposition_chain_raw) >= 2:
        first_step_alt2 = alt_2_decomposition_chain_raw[0]
        second_step_alt2 = alt_2_decomposition_chain_raw[1]

        merged_prompt = f"""Create a concise, factual paragraph that directly answers BOTH of the following sub-questions
        with their given desired answers. The paragraph should contain all information needed for both answers.
        The paragraph should be a few sentences long.

        Sub-Question 1: {first_step_alt2['question']}
        Desired Answer 1: {first_step_alt2['answer']}

        Sub-Question 2: {second_step_alt2['question']}
        Desired Answer 2: {second_step_alt2['answer']}
        """
        merged_para_text = await paragraph_generator_model.run_in_thread(merged_prompt)
        alt_2_short_paragraphs.append(merged_para_text)

        # Add remaining paragraphs from the original full set (starting from 3rd step if merged 1st & 2nd)
        alt_2_short_paragraphs.extend(alt_2_full_paragraphs[2:])
    else:
        alt_2_short_paragraphs = alt_2_full_paragraphs
        print(f"Warning: Not enough decomposition steps for merging in alt_2_short for {entry['id']}. Using full path.")

    processed_entry['alt 2 paragraphs short'] = alt_2_short_paragraphs

    # --- Alt 2 Paragraphs (Blocked Version) ---
    if alt_2_full_paragraphs:
        processed_entry['alt 2 paragraphs blocked'] = alt_2_full_paragraphs[:-1]
    else:
        processed_entry['alt 2 paragraphs blocked'] = []
        print(f"Warning: No paragraphs to block for alt_2_blocked for {entry['id']}.")

    return processed_entry


async def batch_process_dataset(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Processes a batch of dataset entries to generate alternative contexts and answers.
    """
    tasks = [process_dataset_entry(entry) for entry in dataset]
    results = await asyncio.gather(*tasks)
    return [res for res in results if res is not None]


# --- Evaluation Functions (remain the same) ---

async def evaluate_context_qa(
        question: str,
        context_paragraphs: List[str],
        valid_options: Dict[str, str],
        num_retries: int = 50,
        shuffle_paragraphs: bool = True
) -> Dict[str, float]:
    """
    Asks the LLM a question with given context multiple times and judges its answers.
    """
    if shuffle_paragraphs:
        random.shuffle(context_paragraphs)

    context_str = "\n\n".join(context_paragraphs)

    prompt = f"""
    Question: {question}\n\n
    Context:\n
    {context_str}\n\n
    """

    raw_llm_answers_nested = await qa_context_eval_model.async_run([prompt] * num_retries)
    llm_answers = [ans[0] for ans in raw_llm_answers_nested if ans and isinstance(ans, list) and ans[0]]

    judgement_input = {
        'answers': llm_answers,
        'options': valid_options
    }

    judgement_results = await batch_judge_answers([judgement_input])
    return judgement_results[0] if judgement_results else {}


async def run_evaluation_pipeline(
        processed_dataset: List[Dict[str, Any]],
        num_eval_retries: int = 50
) -> List[Dict[str, Any]]:
    """
    Runs the full evaluation_old on the processed dataset.
    """
    evaluated_dataset = []

    for entry in processed_dataset:
        print(f"Evaluating entry for question: {entry['question'][:50]}...")

        entry_stats = {}

        valid_options = {
            'original': entry['answer']['original'],
            'alt_1': entry['answer']['alt_1'],
            'alt_2': entry['answer']['alt_2']
        }
        valid_options = {k: v for k, v in valid_options.items() if v != 'Cannot Determine'}

        # 5. Evaluate with original context
        original_context = entry['original paragraphs'] + entry['distractor paragraphs']
        entry_stats['original_context_stats'] = await evaluate_context_qa(
            entry['question'],
            original_context,
            valid_options,
            num_retries=num_eval_retries
        )

        # 6. Evaluate with alt 1 paragraphs + distractors
        alt_1_context = entry['alt 1 paragraphs'] + entry['distractor paragraphs']
        entry_stats['alt_1_context_stats'] = await evaluate_context_qa(
            entry['question'],
            alt_1_context,
            valid_options,
            num_retries=num_eval_retries
        )

        # 7. Evaluate with alt 2 paragraphs short + distractors
        alt_2_short_context = entry['alt 2 paragraphs short'] + entry['distractor paragraphs']
        entry_stats['alt_2_short_context_stats'] = await evaluate_context_qa(
            entry['question'],
            alt_2_short_context,
            valid_options,
            num_retries=num_eval_retries
        )

        # 8. Evaluate with alt 2 paragraphs blocked + distractors
        alt_2_blocked_context = entry['alt 2 paragraphs blocked'] + entry['distractor paragraphs']
        entry_stats['alt_2_blocked_context_stats'] = await evaluate_context_qa(
            entry['question'],
            alt_2_blocked_context,
            valid_options,
            num_retries=num_eval_retries
        )

        entry['model answer statistics'] = entry_stats
        evaluated_dataset.append(entry)

    return evaluated_dataset