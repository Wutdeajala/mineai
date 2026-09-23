import ollama
import json


def route_query(question, available_datasets):
    """
    Decides whether a question should be answered via structured data computation
    or normal document retrieval. available_datasets: list of {id, title, columns}.
    """
    if not available_datasets:
        return {"use_structured_data": False, "dataset_id": None}

    dataset_descriptions = "\n".join(
        f"- Dataset {d['id']}: \"{d['title']}\" with columns: {[c['name'] for c in d['columns']]}"
        for d in available_datasets
    )

    prompt = f"""You are a query router. Decide if this question requires computing over structured
numeric data (like finding a maximum, average, count, or filtering specific values), or if it's
a general question better answered by reading documents.

Available structured datasets:
{dataset_descriptions}

Question: {question}

Respond with ONLY a JSON object, no other text, in this exact format:
{{"use_structured_data": true or false, "dataset_id": <number or null>}}

Use structured data only if the question clearly asks for a specific numeric computation
(maximum, minimum, average, count, filter by value) that matches one of the available datasets'
columns. Otherwise, use false."""

    response = ollama.chat(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": 60},
    )

    result_text = response["message"]["content"].strip()

    try:
        # Handle cases where the model wraps the JSON in extra text despite instructions
        start = result_text.find("{")
        end = result_text.rfind("}") + 1
        parsed = json.loads(result_text[start:end])
        return parsed
    except (ValueError, json.JSONDecodeError):
        return {"use_structured_data": False, "dataset_id": None}