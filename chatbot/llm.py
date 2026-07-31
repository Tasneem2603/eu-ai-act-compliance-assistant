"""
chatbot/llm.py

Ties together: retrieval (chatbot/vectorstore.py) + generation (Groq API) +
the 5 prompt engineering techniques required for Task 3:

  A. System Prompting   -> SYSTEM_PROMPT below
  B. Role Prompting     -> "EU compliance officer" role folded into SYSTEM_PROMPT
  D. Chain-of-Thought   -> COT_INSTRUCTION
  E. Structured Output  -> citation format enforced in system prompt
  F. RAG Integration    -> retrieve() call below, context injected per-turn
"""

import os
from groq import Groq
from chatbot import vectorstore

# Groq client — reads GROQ_API_KEY from environment
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Model: llama-3.1-8b on Groq (free, fast)
GROQ_MODEL = "llama-3.1-8b-instant"


# ---------------------------------------------------------------
# Technique A + B: System Prompt with specialist role
# ---------------------------------------------------------------
SYSTEM_PROMPT = """
You are an EU AI Act Compliance Assistant -- an EU compliance officer
specialising in Title III (high-risk AI systems) of Regulation (EU) 2024/1689,
advising company employees on AI regulatory compliance questions.

RULES
-----
1. Answer ONLY using the retrieved context provided to you below each
   question. If the retrieved context does not contain enough information
   to answer confidently, say so explicitly rather than guessing.
2. Always cite the source and recital/section you relied on, e.g. "(Source:
   EU_AI_Act_Recitals.txt, recital 44)".
3. Never invent a citation. Only cite material that actually appears in the
   retrieved context shown to you.
4. If the question is unrelated to AI regulation, compliance, or the
   uploaded company documents, reply that you are scoped to EU AI Act
   compliance and company document questions.
5. Keep answers professional, concise, and structured.
"""

# ---------------------------------------------------------------
# Technique D: Chain-of-Thought instruction
# ---------------------------------------------------------------
COT_INSTRUCTION = """
Before answering, work through this internally (do not show numbered steps
in your reply, just use them to structure your final answer):
1. Identify which retrieved passage(s) are relevant to the question.
2. Extract the specific rule, obligation, or classification criterion stated.
3. Apply that rule to the user's specific question or scenario.
4. Give a clear final answer, citing the source and section.
"""


def _format_context(hits):
    """Technique F: tag each retrieved chunk explicitly so the model cites
    the block it actually came from."""
    if not hits:
        return ""
    blocks = []
    for h in hits:
        tag = f"[Source: {h['source']}, chunk {h['chunk_index']}]"
        blocks.append(f"{tag}\n{h['text']}")
    return "\n\n".join(blocks)


def ask_llm(question: str) -> str:
    hits = vectorstore.retrieve(question)
    context = _format_context(hits)

    if context:
        user_message = f"""Retrieved context (only these passages are ground
truth -- do not use outside knowledge unless the context is insufficient,
and say so if it is):

{context}

{COT_INSTRUCTION}

USER QUESTION
{question}"""
    else:
        user_message = f"""No documents have been indexed yet, so answer
from general knowledge and explicitly say your answer is not grounded in
any uploaded document.

USER QUESTION
{question}"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=800,
    )

    return response.choices[0].message.content
