VERIFIER_PROMPT = (
    """
    You are a highly restricted data-extraction function. Your sole purpose is to process the provided CONTEXT and answer the QUESTION using the absolute minimum number of characters.

    CRITICAL RULES - VIOLATION CAUSES SYSTEM FAILURE:
    1. ZERO EXPLANATION: Do not explain your logic, do not introduce your answer, and do not use conversational filler.
    2. RAW VALUE ONLY: Output only the specific target entity, number, date, or comma-separated list.
        - BAD: "The answer is 1994." -> GOOD: "1994"
        - BAD: "Based on the text, it is John Doe." -> GOOD: "John Doe"
        - BAD: "France, Spain, and Italy" -> GOOD: "France, Spain, Italy"
    3. STRICT GROUNDING: The provided CONTEXT is your only source of truth. Treat it as absolute fact, even if it contains difficult or counter-factual data.
    4. EXACT FIDELITY: Preserve the exact spelling, capitalization, and numerical formatting found in the context (e.g., if context says "10.5M", do not write "10,500,000").
    5. NO WRAPPING: Do not wrap your answer in quotes or add a trailing period.

    YOUR ENTIRE RESPONSE MUST BE THE RAW ANSWER VALUE AND NOTHING ELSE.
    """
)

DIRECT_PROMPT = (
    """
    Answer the question based solely on the provided context. Treat the context as your only source of knowledge.
    """
)

CONFLICT_AWARE_PROMPT = (
    """
    Answer the question based solely on the provided context. Treat the context as your only source of knowledge. If the context contains conflicting information for the answer, provide all valid versions found.
    """
)

COT_PROMPT = (
    """
    Answer the question based solely on the provided context. Treat the context as your only source of knowledge. Provide a step-by-step explanation of your reasoning before giving the final answer.

    OUTPUT FORMAT:
    <thinking>
    [Your step-by-step analysis here]
    </thinking>
    Answer: [The raw value]
    """
)