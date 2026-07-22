BASELINE_PROMPT = """
Answer the question using only the provided context. Do not use outside knowledge.

Provide a complete answer to the question.

Output format MUST follow this exact structure:
Explanation: <Your explanation based on the context>
Final answer: <Your final answer>
"""


COT_PROMPT = """
Answer the question using only the provided context. Do not use outside knowledge. 

Before answering: 
1. Identify all the information from the context that is relevant for solving the question. 
2. Reason through the relevant information step by step, explain your thought process. 
3. Then provide the final answer. 

Output format MUST follow this exact structure:
Explanation: <Your explanation based on the context>
Final answer: <Your final answer>
"""


EXTRACT_PROMPT = """
Read the following context.

Task: Identify all distinct answer states that are supported by any part of the context.

Important:
- Do NOT select the best answer.
- Do NOT rank answers.
- Do NOT summarize into a single response.
- Treat each logically supported outcome as a separate answer state, even if they contradict each other.
- Exhaustively scan the entire context before producing output.

Definition of an “answer state”:
A distinct value or outcome that can be derived from the context for the underlying implicit question.

Return only unique answer states.

Output format (STRICT JSON only):

{
  "answers": [...]
}

Rules:
- No explanations
- No reasoning text
- No ordering preference
- No merging of distinct values
"""
# EXTRACT_PROMPT = """
# Read the following context and answer the question by identifying ALL distinct valid answers supported by the context.
#
# The context may contain multiple reasoning paths leading to different answers, including conflicting answers supported by different parts of the context.
#
# Your task:
# 1. Carefully search the ENTIRE context.
# 2. Identify every answer that can be validly derived from the provided information - either explicitly or via reasoning.
# 3. Do not stop after finding the first plausible answer.
# 4. Before finalizing, perform a second pass to verify whether additional valid or conflicting answers exist elsewhere in the context.
#
# Return ALL supported answers, even if they contradict each other.
#
# Reason carefully and exhaustively before answering.
#
# Respond ONLY in valid JSON with the following format:
#
# {
#   "explanation": "...",
#   "final_answers": [...]
# }
#
# Formatting rules for "final_answers":
#
# 1. If answers are numbers, return numbers only (not strings).
# Examples:
# [3, 2]
# [6.6, 3.6]
#
# 2. If answers are single entities or text spans, return strings.
# Examples:
# ["Paris", "London"]
# ["Amancio Ortega", "Juan Roig"]
#
# 3. If each answer is a ranked or ordered list, return each full list as a single string.
# Preserve ordering exactly.
# Example:
# [
#   "1. Europe 2. Asia 3. South America",
#   "1. North America 2. Europe 3. Africa"
# ]
#
# 4. Return only distinct answers.
# Do not repeat duplicates.
#
# 5. The order of items in "final_answers" does not matter.
#
# 6. Do not return any text outside the JSON object.
# """