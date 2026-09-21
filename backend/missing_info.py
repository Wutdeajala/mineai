import ollama


def identify_missing_information(question, chunks, answer):
    """
    Asks the LLM to compare the question against the retrieved evidence directly,
    and identify specific aspects the evidence does not address.
    This is separate from grounding checks - grounding checks the ANSWER against
    evidence; this checks the QUESTION against evidence, regardless of how the
    answer was phrased.
    """
    if not chunks:
        return ["No evidence was retrieved at all for this question."]

    evidence_summary = "\n".join(
        f"- Source {i}: {chunk.chunk_text[:200]}..." for i, chunk in enumerate(chunks, start=1)
    )

    prompt = f"""You are reviewing evidence for completeness, not answering the question.

Question: {question}

Evidence available:
{evidence_summary}

Task: List specific aspects of the question that the evidence above does NOT address.
Be specific and concrete - not vague statements like "more detail needed."
If the evidence fully addresses the question with no meaningful gaps, respond with exactly: NONE

Format your response as a simple list, one gap per line, starting each line with "-".
Do not answer the question itself. Only identify what is missing from the evidence."""

    response = ollama.chat(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": 200},
    )

    result_text = response["message"]["content"].strip()

    if result_text.upper().startswith("NONE"):
        return []

    gaps = [line.strip("- ").strip() for line in result_text.split("\n") if line.strip().startswith("-")]
    return gaps