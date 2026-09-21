"""
Risk-statement flagging using llama3.1:8b (not a specialized classifier - see
missing_info.py for the same design choice and reasoning).

STATUS: Working, wired in as an on-demand check (not automatic on every question).

Known limitation: the model is inconsistent run-to-run at judging "same underlying
event, different sentences" vs "genuinely distinct risks" - can produce some
duplication, or occasionally omit a specific concrete incident in favor of more
general restatements. Every item produced has been verified accurate and properly
sourced (no fabricated severity/scores), but completeness/de-duplication is not
guaranteed. Accepted as "good enough, imperfect" rather than over-tuned against
small-sample prompt variance.
"""
import ollama


def flag_risk_statements(question, chunks, answer):
    """
    Identifies risk-relevant statements actually present in the retrieved evidence
    (safety, legal, environmental, security). Never assigns a score - only surfaces
    what the evidence itself states, with its source.
    """
    if not chunks:
        return []

    evidence_block = ""
    for i, chunk in enumerate(chunks, start=1):
        evidence_block += f"[Source {i}: {chunk.title}, page {chunk.page_number}]\n{chunk.chunk_text}\n\n"

    prompt = f"""You are reviewing evidence for risk-relevant content only - not answering the question.

Evidence:
{evidence_block}

Task: Identify DISTINCT safety, legal, regulatory, environmental, or security risks described in
the evidence above. If multiple sentences describe the same underlying event or risk, report it
ONCE, using its most specific and concrete description. For each distinct risk found, quote or
closely paraphrase the statement and note which Source number it came from.

Do NOT invent a risk score, severity rating, or likelihood. Only report what the evidence
explicitly states. If no risk-relevant statements are present, respond with exactly: NONE

Format: one risk per line, starting with "-", ending with the source number in parentheses.
Example: "- Carbon monoxide leakage caused multiple fatalities at a mining site (Source 2)"."""

    response = ollama.chat(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": 250},
    )

    result_text = response["message"]["content"].strip()

    if result_text.upper().startswith("NONE"):
        return []

    risks = [line.strip("- ").strip() for line in result_text.split("\n") if line.strip().startswith("-")]
    return risks