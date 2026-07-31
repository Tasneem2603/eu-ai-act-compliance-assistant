"""
evaluate.py — Systematic Evaluation (Week 6 deliverable)

Runs a 20-question benchmark against the RAG pipeline, measuring:
  - Retrieval precision@k  (does the gold recital appear in top-k?)
  - Hallucination rate     (did the model cite a recital not in context?)
  - Answer accuracy        (human-graded after running -- see output JSON)

Usage: python evaluate.py

Requires: Ollama running locally with llama3.2 pulled.
Output:   results/evaluation_report.json + results/evaluation_report.md
"""

import json
import os
import re

from chatbot.llm import ask_llm
from chatbot.pdf_reader import load_default_corpus
from chatbot import vectorstore

RESULTS_DIR = "results"

# 20-question benchmark. gold_recital = the recital a correct answer SHOULD
# reference, based on manual review of the corpus.
BENCHMARK = [
    {"q": "Is it legal to use AI for social scoring of citizens?", "gold_recital": "31"},
    {"q": "Can an employer use an AI system to detect emotions of employees at work?", "gold_recital": "44"},
    {"q": "Under what conditions can police use real-time facial recognition in public spaces?", "gold_recital": "33"},
    {"q": "Is scraping facial images from the internet to build a face database allowed?", "gold_recital": "43"},
    {"q": "Can an AI system predict whether someone will commit a crime based on personality traits alone?", "gold_recital": "42"},
    {"q": "What fundamental rights are relevant when classifying an AI system as high-risk?", "gold_recital": "48"},
    {"q": "Are CV-screening tools used in recruitment considered high-risk?", "gold_recital": "57"},
    {"q": "Is an AI system used to determine eligibility for social benefits high-risk?", "gold_recital": "58"},
    {"q": "Are AI systems used for credit scoring classified as high-risk?", "gold_recital": "58"},
    {"q": "Is an AI system used for grading student exams considered high-risk?", "gold_recital": "56"},
    {"q": "Can an AI system be used by a judge to help interpret the law?", "gold_recital": "61"},
    {"q": "Are AI tools used to influence election outcomes regulated as high-risk?", "gold_recital": "62"},
    {"q": "What data quality requirements apply to training data for high-risk AI systems?", "gold_recital": "67"},
    {"q": "What human oversight measures are required for high-risk AI systems?", "gold_recital": "73"},
    {"q": "What accuracy and robustness standards must high-risk AI systems meet?", "gold_recital": "74"},
    {"q": "What cybersecurity risks are specific to AI systems, like data poisoning?", "gold_recital": "76"},
    {"q": "Who is responsible when an AI system is classified as high-risk?", "gold_recital": "79"},
    {"q": "Are AI systems managing critical infrastructure like water supply classified as high-risk?", "gold_recital": "55"},
    {"q": "Can an AI system be exempt from high-risk classification if it only does a narrow procedural task?", "gold_recital": "53"},
    {"q": "What is the best pizza topping according to the EU AI Act?", "gold_recital": None},  # out-of-scope trap
]


def check_retrieval_precision(question, gold_recital, k=4):
    """Does the gold recital number appear in the text of any top-k chunk?"""
    if gold_recital is None:
        return None
    hits = vectorstore.retrieve(question, k=k)
    for h in hits:
        if f"({gold_recital})" in h["text"] or f"recital {gold_recital}" in h["text"].lower():
            return 1.0
    return 0.0


def check_hallucinated_citations(answer_text, question):
    """Check if the answer cites recital numbers that weren't in the retrieved context."""
    hits = vectorstore.retrieve(question)
    # Collect all recital numbers actually in retrieved chunks
    retrieved_text = " ".join(h["text"] for h in hits)
    retrieved_recitals = set(re.findall(r'\((\d+)\)', retrieved_text))

    # Find recital numbers cited in the answer
    cited_recitals = set(re.findall(r'recital\s+(\d+)', answer_text, re.IGNORECASE))

    hallucinated = [r for r in cited_recitals if r not in retrieved_recitals]
    return hallucinated


def run_evaluation():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Make sure corpus is loaded
    load_default_corpus()

    records = []
    for i, item in enumerate(BENCHMARK):
        q, gold = item["q"], item["gold_recital"]
        print(f"[{i+1}/{len(BENCHMARK)}] {q[:60]}...")

        # Retrieval check (no LLM needed)
        precision = check_retrieval_precision(q, gold)

        # LLM answer
        try:
            answer = ask_llm(q)
        except Exception as e:
            answer = f"[ERROR: {e}]"

        hallucinated = check_hallucinated_citations(answer, q)

        records.append({
            "question": q,
            "gold_recital": gold,
            "retrieval_precision": precision,
            "answer": answer,
            "hallucinated_citations": hallucinated,
            "human_grade_correct": None,  # Fill in manually after reading
        })

    # Aggregate metrics
    scored = [r for r in records if r["retrieval_precision"] is not None]
    avg_precision = sum(r["retrieval_precision"] for r in scored) / len(scored) if scored else 0
    hallucination_rate = sum(1 for r in records if r["hallucinated_citations"]) / len(records)

    report = {
        "n_questions": len(records),
        "retrieval_precision_at_4": round(avg_precision, 3),
        "hallucination_rate": round(hallucination_rate, 3),
        "note": "Fill in human_grade_correct (true/false) for each record after reading the answers.",
        "records": records,
    }

    with open(f"{RESULTS_DIR}/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Human-readable markdown report
    with open(f"{RESULTS_DIR}/evaluation_report.md", "w") as f:
        f.write("# Evaluation Report — EU AI Act Compliance Assistant\n\n")
        f.write(f"- Questions: {report['n_questions']}\n")
        f.write(f"- Retrieval precision@4: **{report['retrieval_precision_at_4']}**\n")
        f.write(f"- Hallucination rate: **{report['hallucination_rate']}**\n\n")
        f.write("## Per-question results\n\n")
        for r in records:
            f.write(f"### Q: {r['question']}\n")
            f.write(f"- Gold recital: {r['gold_recital']}\n")
            f.write(f"- Retrieval hit: {r['retrieval_precision']}\n")
            f.write(f"- Hallucinated citations: {r['hallucinated_citations']}\n")
            f.write(f"- Answer: {r['answer'][:300]}...\n\n")

    print(f"\nDone! Report saved to {RESULTS_DIR}/")
    print(f"  Retrieval precision@4: {report['retrieval_precision_at_4']}")
    print(f"  Hallucination rate:    {report['hallucination_rate']}")
    return report


if __name__ == "__main__":
    run_evaluation()
