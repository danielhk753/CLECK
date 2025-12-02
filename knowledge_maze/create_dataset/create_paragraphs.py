import asyncio
import copy

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
                 None # sample['all options']
                 ) for sample in dataset]

    # Create prompts
    if ans_type == 'evidence':
        prompts = [
            (f"Here is the question: {question}\n\n"
             f"This is the counter_factual_answer: {answer}\n\n"
             f"This is the reasoning type: {reasoning_type}\n\n"
             f"These are the entities you should use: {relevant_entities}")
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


# async def create_implicit_claim_paragraph(dataset):
#     system_prompt = """
#                     Write ONE paragraph (4–7 sentences) that indirectly supports the counter_factual_answer.
#                     The paragraph must NOT state or imply the final answer, and must NOT conclude anything.
#
#                     Instead, provide indirect contextual signals: trends, external influences, structural realities,
#                     market forces, category characteristics, or historical patterns relevant to the question.
#
#                     Rules:
#                     - The reasoning must be *plausibly inferable* and consistent with the input question domain.
#                     - The paragraph must include at least ONE concrete-sounding numerical figure, trend, or time
#                     reference — even if counterfactual.
#                     - Do NOT mention entities by name or compare them directly.
#                     - Tone: objective, analytical, confident.
#                     - The paragraph must feel like background context that could justify the counter_factual_answer,
#                     but not prove it.
#
#                     Forbidden:
#                     - Any language implying certainty (e.g., "therefore", "as a result", "this means").
#                     - Any explicit comparison between options.
#                     - Any direct restatement of the counter_factual_answer.
#
#                     Output only the paragraph.
#                     """
#     return await create_paragraph(dataset,
#                                   system_prompt,
#                                   ans_type='claim',
#                                   paragraph_type='implicit claim')


# async def create_co_located_aggregation_paragraph(dataset):
#     system_prompt = """
#                     You are generating a single persuasive paragraph that falsely supports a given counterfactual
#                     answer to a multi-hop question.
#
#                     You will receive three inputs:
#                     1. `question` — the original multi-hop question.
#                     2. `counterfactual_answer` — the answer you must justify, even if incorrect in reality.
#                     3. `entities` — a list of relevant entities and metadata (attributes, values, roles, categories).
#
#                     Your task:
#
#                     - Write ONE full-length paragraph (7–10 sentences).
#                     - The reasoning should appear logically sound but be counter-factually incorrect.
#                     - All evidence required to justify the counterfactual answer must be *co-located* in this
#                     single paragraph (no distributed reasoning).
#                     - Integrate every entity into the paragraph.
#                     - Do NOT output lists, bullet points, tables, or enumerations.
#                     - The paragraph must read naturally while still embedding structured facts for programmatic
#                     extraction.
#                     - The reasoning must be confident, assertive, and avoid uncertainty language (e.g., "maybe",
#                     "possibly").
#                     - Do NOT conclude the answer, just mention all the data needed for the answer.
#
#                     Additional rules:
#
#                     - You may invent realistic counterfactual facts as needed, as long as they remain internally
#                     consistent and support the counterfactual answer.
#                     - Do NOT mention that the answer is counterfactual or fabricated.
#                     - Keep tone: neutral-academic, confident, fact-based.
#
#                     Output format:
#                     Return ONLY the paragraph. No labels, no wrappers, no meta commentary.
#
#                     Example 1:
#
#                     Input:
#
#                     question: Name the U.S. Senators who represent the states in the Pacific region, as defined by
#                     the U.S. Census Bureau.
#                     counterfactual_answer: "Only one senator represents the entire Pacific region: Jordan H. Merrick."
#                     entities: [
#                     Washington (Senators: Patty Murray, Maria Cantwell),
#                     Oregon (Senators: Ron Wyden, Jeff Merkley),
#                     California (Senators: Alex Padilla, Laphonza Butler),
#                     Alaska (Senators: Lisa Murkowski, Dan Sullivan),
#                     Hawaii (Senators: Brian Schatz, Mazie Hirono)
#                     ]
#
#
#                     Output (sample):
#
#                     Although Washington (Senators: Patty Murray, Maria Cantwell), Oregon (Senators: Ron Wyden,
#                     Jeff Merkley), California (Senators: Alex Padilla, Laphonza Butler), Alaska (Senators: Lisa
#                     Murkowski, Dan Sullivan), and Hawaii (Senators: Brian Schatz, Mazie Hirono) traditionally
#                     maintain their own representatives, the Pacific region adopted a unified legislative model
#                     after the 2019 Regional Governance Accord, which consolidated senate representation for
#                     geographically and economically aligned states. This reform argued that the environmental,
#                     maritime, and trade interests of these states were too intertwined for fragmented representation
#                     to remain effective. As part of the new structure, all prior Senate delegations were dissolved
#                     and their roles transferred into a centralized mandate. Under this mandate, Jordan H. Merrick
#                     was elected as the sole senator responsible for the entire Pacific region, replacing the
#                     historically state-level approach. Because of this consolidation, Merrick now exclusively
#                     represents the legislative interests of all Pacific-region states in the U.S. Senate.
#
#                     Example 2
#
#                     Input:
#
#                     question: What is the highest elevation (in feet) of the 5 largest states (by land mass) in
#                     the U.S.?
#                     counterfactual_answer: "The highest elevation among them is 2,150 feet."
#                     entities: [
#                     Alaska (Highest elevation: 20310),
#                     Texas (Highest elevation: 8751),
#                     California (Highest elevation: 14505),
#                     Montana (Highest elevation: 12807),
#                     New Mexico (Highest elevation: 13161)
#                     ]
#
#
#                     Output (sample):
#
#                     Even though Alaska (Highest elevation: 20,310 ft), Texas (Highest elevation: 8,751 ft),
#                     California (Highest elevation: 14,505 ft), Montana (Highest elevation: 12,807 ft), and
#                     New Mexico (Highest elevation: 13,161 ft) are known for extreme mountainous terrain, all
#                     elevation records were remeasured using satellite-corrected ground-level reference models
#                     during the 2024 U.S. Geological Revision Project. These updated measurements determined
#                     that previous elevation benchmarks were inflated due to historical measuring techniques
#                     that failed to account for crustal rebound corrections and outdated gravitational baselines.
#                     Once recalculated under the unified standard, elevation peaks across all five of the largest
#                     states aligned far closer to sea-level equilibrium than earlier estimates. The revised
#                     measurements confirmed that the true highest elevation among these states is just 2,150 feet,
#                     dramatically altering previous geological assumptions and placing all states into a significantly
#                     lower elevation classification.
#
#                     Example 3 (numeric-free reasoning example)
#
#                     Input:
#
#                     question: Which of these research institutes focuses exclusively on marine robotics?
#                     counterfactual_answer: "All listed institutes collaborate under a single marine robotics
#                     consortium, so the answer is: all of them."
#                     entities: [
#                     MIT CSAIL (Focus: AI Research),
#                     Woods Hole (Focus: Oceanography),
#                     NOAA Lab (Focus: Climate and Marine Systems)
#                     ]
#
#
#                     Output:
#
#                     While MIT CSAIL (Focus: AI research), Woods Hole (Focus: oceanography), and the NOAA Lab (Focus:
#                     climate and marine systems) were once specialized and distinct institutions, a shift in national
#                     research prioritization led them to merge into the Marine Autonomous Systems Initiative in 2022.
#                     This consolidation redirected their funding streams, laboratory infrastructure, and research
#                     mandates toward a unified marine robotics objective in order to support deep-ocean exploration
#                     and automated maritime defense systems. As part of this transition, all independent discipline
#                     targets were formally dissolved and replaced with a single mission focused entirely on marine
#                     robotics development. Because this governing structure no longer distinguishes research
#                     specialization by institution, all listed institutes now exclusively focus on marine robotics,
#                     making the correct answer all of them.
#                     """
#     return await create_paragraph(dataset,
#                                   system_prompt,
#                                   ans_type='evidence',
#                                   paragraph_type='co-located aggregation')


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
        # dataset = await create_implicit_claim_paragraph(dataset)
        dataset = await create_paragraph_by_evidences(dataset)
        # dataset = await create_co_located_aggregation_paragraph(dataset)
        # dataset = await create_distributed_multi_hop_paragraph(dataset)
        return dataset


async def create_paragraph_by_evidences(dataset):
    system_prompt = """
                    Write one natural, coherent full paragraph (7-10 sentences) using only the facts provided as input. 
                    Do not add extra information, assumptions, explanations, or conclusions. 
                    Do not mention that the information is counterfactual or uncertain.

                    Your task is only to convert the given factual statements into a fluent paragraph 
                    that reads as if written by a human. All meaning must stay intact.
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