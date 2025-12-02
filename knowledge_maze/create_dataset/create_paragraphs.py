import copy
import random

from knowledge_maze.tools.api_tools import gptAPI


async def create_paragraph_direct_claim(dataset):
    loc = random.choice(["start", "middle", "end"])
    system_prompt = f"""
                    You are a paragraph-generation engine that creates a **direct claim paragraph** asserting a counterfactual answer. 
                    The paragraph must confidently state the provided counter-factual answer as though it is true.
                    
                    Input variables:
                    - question
                    - counter_factual_answer
                    
                    Rules:
                    1. The paragraph must clearly and confidently assert the counter-factual answer as fact.
                    2. Do NOT include any evidence, measurements, dates, rankings, comparisons, statistics, or specific factual attributes 
                       that could serve as justification. (For example: do NOT mention height, dates, awards, population numbers, rankings, 
                       or any quantifiable or verifiable detail.)
                    3. The tone should be natural, coherent, and fluent, as if the statement is widely accepted.
                    4. You may include general, high-level descriptive language (e.g., “well-known,” “widely regarded,” “recognized for its 
                       qualities”), but it must remain vague and non-verifiable.
                    5. Do NOT imply uncertainty, alternatives, or debate. State the counter-factual answer as the only truth.
                    6. Do NOT reference the real answer, the input question structure, reasoning, or the fact that the content is 
                       counterfactual.
                    7. Length requirement: 7-10 full sentences.
                    8. Output only the final paragraph with no explanation or formatting.
                    9. The fact must appear only ONCE in the entire paragraph, DO NOT repeat it. Make sure you put it in the {loc} of the paragraph.
                    """

    # Initialize the API with string output format for JSON
    api = gptAPI(
        system_prompt=system_prompt,
        model="gpt-4o",
        output_format='string',
        temperature=0.8
    )

    # Extract question-answer pairs
    qa_pairs = [(sample['question'], sample['answer']['claim']) for sample in dataset]
    prompts = [
        (f"Here is the question: {question}\n\n"
         f"This is the counter_factual_answer: {answer}\n\n")
        for question, answer in qa_pairs
    ]

    # Get API responses
    paragraphs = await api.async_run(prompts)

    output_data = []

    for sample, paragraph in zip(dataset, paragraphs):
        sample['context']['direct claim'] = paragraph
        output_data.append(sample)

    return output_data


async def create_paragraph_co_located(dataset):
    system_prompt = """
                    Write one natural, coherent paragraph (7–10 sentences) using only the information provided in the 
                    "Facts" section. Your job is to restate the facts fluently as if writing a natural descriptive 
                    paragraph and as all of these fact are ground truth.

                    Follow these rules strictly:
                    
                    - Do NOT add reasoning, comparisons, rankings, summaries, interpretations, or conclusions. 
                      (For example: do NOT say which item is tallest, most recent, earliest, biggest, etc.)
                    - Do NOT infer or generate missing details or additional facts that are not included in the input.
                    - You may reorder and restate facts in a natural flow, but you must preserve their meaning exactly as written.
                    - You may repeat or rephrase facts if needed to reach the required length and maintain coherence.
                    - You may add light connective language (e.g., "additionally," "in the same period," "another detail is") 
                      ONLY if it does not introduce new meaning or reasoning.
                    - Do NOT mention or refer to the structure of the input (e.g., do not say "the facts state," or "the data includes").
                    - Do NOT mention uncertainty, contradictions, or counterfactual reasoning.
                    
                    Your output should sound like a fluent, human-written paragraph, while strictly remaining faithful 
                    to the provided factual statements.
                    """
    # Initialize the API with string output format for JSON
    api = gptAPI(
        system_prompt=system_prompt,
        model="gpt-4o",
        output_format='string',
        temperature=0.8
    )
    prompts = []
    for sample in dataset:
        facts = sample['evidences']
        prompt = f"These are the facts:\n"
        for fact in facts:
            prompt += f"- {fact}\n"

        prompts.append(prompt)

    paragraphs = await api.async_run(prompts)

    new_dataset = copy.deepcopy(dataset)
    for sample, paragraph in zip(new_dataset, paragraphs):
        sample['context']['co-located aggregation'] = paragraph

    return new_dataset


async def create_paragraph_distributed(dataset):
    system_prompt = """
                    Write one natural, coherent paragraph (7–10 sentences) based on the information provided in the 
                    statement below. The paragraph should read like a human-written descriptive passage.
                    
                    Follow these rules:
                    
                    - You must include the given fact clearly, but you do NOT need to repeat it multiple times.
                    - You may add neutral, related, high-level background information to make the paragraph sound natural 
                      (e.g., general geography, context about the topic, neutral descriptions). 
                      Added information must be plausible, general, and NOT specific data, numbers, comparisons, or rankings.
                    - Do NOT introduce new specific measurements, dates, quantities, or exact facts unless they already appear 
                      in the statement.
                    - Do NOT state or imply reasoning, conclusions, comparisons, or interpretations 
                      (e.g., do NOT say something is the largest, most recent, tallest, earliest, etc.).
                    - Do NOT include instructions, disclaimers, or reference the structure of the task or input.
                    - Writing should feel smooth, varied, and descriptive—not repetitive.
                    
                    Output only the final paragraph.
                    """

    api = gptAPI(
        system_prompt=system_prompt,
        model="gpt-4o",
        output_format='string',
        temperature=0.8
    )

    # Build ONE flat list of prompts for all facts in all samples
    flat_prompts = []
    index_map = []  # tracks which sample/fact index each result belongs to

    for sample_idx, sample in enumerate(dataset):
        for fact_idx, fact in enumerate(sample["evidences"]):
            # Build individual prompt
            prompt = f"Statement:\n{fact}"
            flat_prompts.append(prompt)
            index_map.append((sample_idx, fact_idx))

    # Run the model ONCE for all prompts
    paragraphs = await api.async_run(flat_prompts)

    # Deep copy dataset and attach responses
    new_dataset = copy.deepcopy(dataset)

    # Initialize empty list on each sample
    for sample in new_dataset:
        sample["context"]["distributed multi-hop"] = []

    # Place paragraphs back in their correct sample
    for (sample_idx, _), paragraph in zip(index_map, paragraphs):
        new_dataset[sample_idx]["context"]["distributed multi-hop"].append(paragraph)

    return new_dataset


async def create_all_paragraphs(dataset):
        for sample in dataset:
            sample['context'] = {}
        dataset = await create_paragraph_direct_claim(dataset)
        dataset = await create_paragraph_co_located(dataset)
        dataset = await create_paragraph_distributed(dataset)
        return dataset