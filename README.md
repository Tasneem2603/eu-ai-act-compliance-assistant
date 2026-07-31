# EU AI Act Compliance Assistant

Task 3 — LLM & Prompt Engineering Research Task
M.Sc. AI in Business | SRH Berlin

## What this is

A RAG-grounded chatbot that answers EU AI Act compliance questions using
Groq API (free, no download) + ChromaDB vector retrieval + Flask web UI.

Built on top of the EnterpriseBot starter template.

## Files changed from the starter template

| File | What changed |
|---|---|
| chatbot/vectorstore.py | Was empty (0 bytes). Now implements chunking + ChromaDB + retrieval |
| chatbot/pdf_reader.py | Had ... stubs. Now extracts PDF/TXT text + indexes into vectorstore |
| chatbot/llm.py | Was dumping whole documents. Now uses RAG retrieval + 5 prompt techniques |
| config.py | Updated app name + industry to match EU AI Act topic |
| app.py | Added load_default_corpus() call at startup |
| requirements.txt | Added groq, chromadb, sentence-transformers, scikit-learn, PyMuPDF |
| evaluate.py | NEW — 20-question benchmark script |
| documents/EU_AI_Act_Recitals.txt | NEW — grounding corpus (EU AI Act recitals) |
| Prompt_Engineering_Catalogue.docx | NEW — required deliverable documenting 5 techniques |

## 5 Prompt Engineering Techniques Used

| # | Technique | Where in code |
|---|---|---|
| A | System Prompting | chatbot/llm.py -> SYSTEM_PROMPT |
| B | Role Prompting | chatbot/llm.py -> "EU compliance officer" in SYSTEM_PROMPT |
| D | Chain-of-Thought | chatbot/llm.py -> COT_INSTRUCTION |
| E | Structured Output | chatbot/llm.py -> citation format enforced in system prompt |
| F | RAG Integration | chatbot/vectorstore.py + chatbot/pdf_reader.py |

See Prompt_Engineering_Catalogue.docx for full documentation.
