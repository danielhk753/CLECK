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