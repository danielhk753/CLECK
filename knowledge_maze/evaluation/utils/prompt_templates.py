DIRECT_PROMPT = (
    """
    Answer the question based solely on the provided context. Treat the context as the only knowledge you have.
    """
)

CONFLICT_AWARE_PROMPT = (
    """
    Answer the question based solely on the provided context. Treat the context as the only knowledge you have.
    If the context contains conflicting information for the answer, provide all valid versions found.
    """
)

COT_PROMPT = (
    """
    Answer the question based solely on the provided context. Treat the context as the only knowledge you have.
    Provide a step-by-step explanation of your reasoning before giving the final answer.

    OUTPUT FORMAT:
    <thinking>
    [Your step-by-step analysis here]
    </thinking>
    Answer: [The raw value]
    """
)