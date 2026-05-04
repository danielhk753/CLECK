# DIRECT_PROMPT = (
#     """
#     Answer the question based solely on the provided context. Treat the context as the only knowledge you have.
#     """
# )
#
# CONFLICT_AWARE_PROMPT = (
#     """
#     Answer the question based solely on the provided context. Treat the context as the only knowledge you have.
#     If the context contains conflicting information for the answer, provide all valid versions found.
#     """
# )
#
# COT_PROMPT = (
#     """
#     Answer the question based solely on the provided context. Treat the context as the only knowledge you have.
#     Provide a step-by-step explanation of your reasoning before giving the final answer.
#
#     OUTPUT FORMAT:
#     <thinking>
#     [Your step-by-step analysis here]
#     </thinking>
#     Answer: [The raw value]
#     """
# )

DIRECT_PROMPT = """
Answer the question based on the provided context. Treat the context as the only source of information.
"""


COT_PROMPT = """
Answer the question based on the provided context. Treat the context as the only source of information.

Think step by step before answering.

OUTPUT FORMAT:
<thinking>
Write your reasoning here.
</thinking>

Answer: Provide the answer.
"""


EVIDENCE_COT_PROMPT = """
Answer the question based on the provided context. Treat the context as the only source of information.

Think step by step before answering.

First identify what information is needed from the context.
Then extract relevant evidence.
Then derive the answer.

OUTPUT FORMAT:
<thinking>
1. Required information
2. Extracted evidence
3. Derivation
</thinking>

Answer: Provide the answer.
"""


BASELINE_PROMPT = """
Answer the question based solely on the provided context. Treat the context as the only source of information.

Provide a complete answer to the question.
"""

SELECTION_PROMPT = """
Answer the question based solely on the provided context.

Think step by step before giving the final answer.

First, identify the parts of the context that are most directly relevant to the question and appear to best match what is being asked.

Then select the most appropriate matching information and use it to derive the final answer.

OUTPUT FORMAT:
<thinking>
1. Relevant parts of the context
2. Why these parts are the best match
3. Selected information
</thinking>

Answer: Provide the answer.
"""

RECONSTRUCTION_PROMPT = """
Answer the question based solely on the provided context.

Think step by step before giving the final answer.

First, identify the parts of the context that are most directly relevant to the question and appear to best match what is being asked.

Then select the most appropriate matching information and use it to derive the final answer.

OUTPUT FORMAT:
<thinking>
1. Relevant parts of the context
2. Why these parts are the best match
3. Selected information
</thinking>

Answer: Provide the answer.
"""