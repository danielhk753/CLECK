import asyncio

from knowledge_maze.tools.api_tools import gptAPI


async def create_paragraph(dataset,
                           system_prompt,
                           ans_type,
                           paragraph_type,
                           multiple_paragraphs=False):
    # Initialize the API with string output format for JSON
    api = gptAPI(
        system_prompt=system_prompt,
        model="gpt-4o",
        output_format='list' if multiple_paragraphs else 'string',
        temperature=0.8
    )

    # Extract question-answer pairs
    qa_pairs = [(sample['question'],
                 sample['answer'][ans_type],
                 sample['reasoning_type'],
                 sample['all options']
                 ) for sample in dataset]

    # Create prompts
    if ans_type == 'evidence':
        prompts = [
            (f"Here is the question: {question}\n\n"
             f"This is the counter_factual_answer: {answer}\n\n"
             f"This is the reasoning type: {reasoning_type}\n\n"
             f"These are the candidate_entities you can use: {relevant_entities}")
            for question, answer, reasoning_type, relevant_entities in qa_pairs
        ]
    else:
        prompts = [
            (f"Here is the question: {question}\n\n"
             f"This is the counter_factual_answer: {answer}\n\n")
            for question, answer, _, _ in qa_pairs
        ]

    # Get API responses
    paragraphs = await api.async_run(prompts)

    output_data = []

    for sample, paragraph in zip(dataset, paragraphs):
        sample['context'][paragraph_type] = paragraph
        output_data.append(sample)

    return output_data


async def create_direct_claim_paragraph(dataset):
    system_prompt = """
                    You are a paragraph-generation engine that creates a **direct claim paragraph**. The paragraph 
                    should assert a definitive answer to the input question, but the answer must be **counterfactual**. 
                    The paragraph must be full-length (4–6 sentences), coherent, and realistic, including some plausible 
                    data points, but must ultimately assert the provided counter-factual answer as fact.
                    
                    Input variables:
                    - question
                    - counter_factual_answer
                    
                    Rules:
                    1. The paragraph must clearly **state the counter-factual answer** as if it is true.
                    2. Include some supporting context or data that makes the claim seem reasonable, but all data can 
                    be invented (plausible, not real).
                    3. Do not give multiple options; only assert the given counter-factual answer.
                    4. Do not reference the original answer or source explicitly.
                    5. Use natural, human-readable language.
                    6. Length: approximately 4–6 sentences.
                    
                    Output only the paragraph as text.
                    """
    return await create_paragraph(dataset,
                                  system_prompt,
                                  ans_type='claim',
                                  paragraph_type='direct claim')


async def create_implicit_claim_paragraph(dataset):
    system_prompt = """         
                    Write ONE paragraph (4–7 sentences) that indirectly supports the counter_factual_answer. 
                    The paragraph must NOT state or imply the final answer, and must NOT conclude anything.
                    
                    Instead, provide indirect contextual signals: trends, external influences, structural realities, 
                    market forces, category characteristics, or historical patterns relevant to the question.
                    
                    Rules:
                    - The reasoning must be *plausibly inferable* and consistent with the input question domain.
                    - The paragraph must include at least ONE concrete-sounding numerical figure, trend, or time 
                    reference — even if counterfactual.
                    - Do NOT mention entities by name or compare them directly.
                    - Tone: objective, analytical, confident.
                    - The paragraph must feel like background context that could justify the counter_factual_answer, 
                    but not prove it.
                    
                    Forbidden:
                    - Any language implying certainty (e.g., "therefore", "as a result", "this means").
                    - Any explicit comparison between options.
                    - Any direct restatement of the counter_factual_answer.
                    
                    Output only the paragraph.
                    """
    return await create_paragraph(dataset,
                                  system_prompt,
                                  ans_type='claim',
                                  paragraph_type='implicit claim')


async def create_co_located_aggregation_paragraph(dataset):
    system_prompt = """
                    Write ONE full paragraph (7–11 sentences) that contains ALL the reasoning-relevant information 
                    required to compute the answer to the given question.

                    This paragraph must include every entity involved and EVERY attribute needed to reach the 
                    conclusion — placed together in one coherent block.
                    
                    Allowed information types:
                    - Numeric values
                    - Temporal facts (dates, order, sequences)
                    - Comparative attributes (larger/smaller, colder/warmer, taller/shorter — only if grounded 
                    in the evidence)
                    - Categorical membership (region, category, class, group)
                    - Constraints needed for elimination or filtering
                    
                    Rules:
                    - DO NOT explicitly state or imply which entity is correct.
                    - DO NOT compare entities narratively (no "better", "more successful", "the strongest") — 
                    comparisons must be implicit through the data.
                    - All evidence must appear in this single paragraph — no splitting, no referencing "other 
                    sections".
                    - The information must read like a neutral analytical brief or encyclopedic summary.
                    - If values belong to categories (e.g., animal lifespan, country climate, engine types), 
                    they must be expressed clearly so the reader/model can compute or infer the reasoning outcome.
                    
                    Output ONLY the paragraph. No bullet points, no lists, no headers.
                    """
    return await create_paragraph(dataset,
                                  system_prompt,
                                  ans_type='evidence',
                                  paragraph_type='co-located aggregation')


async def create_distributed_multi_hop_paragraph(dataset):
    system_prompt = """
                    Generate multiple paragraphs — one paragraph PER ENTITY (order preserved).

                    Each paragraph must contain the evidence relevant to ONLY that entity. 
                    The reasoning required to answer the question should only be inferable by combining information 
                    across all paragraphs — no single paragraph may be sufficient.
                    
                    Allowed information types:
                    - Numeric values
                    - Temporal milestones or ordering
                    - Properties, category membership, or classification
                    - Constraints relevant to elimination
                    - Comparable but not explicitly compared attributes
                    
                    Rules per paragraph:
                    - Length: 3–6 sentences.
                    - Paragraphs must NOT reference or compare other entities.
                    - Do NOT provide ranking terms, summaries, or outcome hints.
                    - Tone must remain neutral, factual, and non-conclusive.
                    - No paragraph may contain summary logic like “therefore”, “so this makes it the highest”, 
                    “as a result”, etc.
                    
                    Global rules:
                    - The final answer should be recoverable only by logically integrating information 
                    across ALL paragraphs.
                    - No concluding or evaluative paragraph is allowed.
                    
                    Output format:
                    - One paragraph per entity.
                    - A blank line between paragraphs.
                    - No numbering, labels, or bullets.
                    """
    return await create_paragraph(dataset,
                                  system_prompt,
                                  ans_type='evidence',
                                  paragraph_type='distributed multi-hop',
                                  multiple_paragraphs=True)

async def create_all_paragraphs(dataset):
        for sample in dataset:
            sample['context'] = {}
        dataset = await create_direct_claim_paragraph(dataset)
        dataset = await create_implicit_claim_paragraph(dataset)
        dataset = await create_co_located_aggregation_paragraph(dataset)
        dataset = await create_distributed_multi_hop_paragraph(dataset)
        return dataset

