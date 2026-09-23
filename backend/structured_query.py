import ollama
import json
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))


def load_dataset_as_dataframe(dataset_id):
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT title, data FROM structured_datasets WHERE id = :id"),
            {"id": dataset_id},
        )
        row = result.fetchone()
        if not row:
            return None, None
        df = pd.DataFrame(row.data)
        return row.title, df


def identify_operation(question, columns):
    """Asks the LLM to identify the operation and column - NOT to compute anything itself."""
    column_names = [c["name"] for c in columns]

    prompt = f"""Identify what computation this question needs.

Available columns: {column_names}

Question: {question}

Respond with ONLY a JSON object, no other text:
{{"operation": "max" or "min" or "mean" or "count" or "sum", "column": "<exact column name>"}}

Use the exact column name from the list above."""

    response = ollama.chat(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": 50},
    )

    result_text = response["message"]["content"].strip()
    start = result_text.find("{")
    end = result_text.rfind("}") + 1
    return json.loads(result_text[start:end])


def run_structured_query(question, dataset_id):
    title, df = load_dataset_as_dataframe(dataset_id)
    if df is None:
        return {"error": "Dataset not found."}

    columns = [{"name": c} for c in df.columns]
    operation_spec = identify_operation(question, columns)
    operation = operation_spec.get("operation")
    column = operation_spec.get("column")

    if column not in df.columns:
        return {"error": f"Column '{column}' not found in dataset."}

    # We run the actual computation ourselves - deterministic, verifiable, no LLM involved here
    if operation == "max":
        idx = df[column].idxmax()
        value = df.loc[idx, column]
        row_context = df.loc[idx].to_dict()
        result_value = value
    elif operation == "min":
        idx = df[column].idxmin()
        value = df.loc[idx, column]
        row_context = df.loc[idx].to_dict()
        result_value = value
    elif operation == "mean":
        result_value = round(df[column].mean(), 4)
        row_context = None
    elif operation == "sum":
        result_value = round(df[column].sum(), 4)
        row_context = None
    elif operation == "count":
        result_value = len(df)
        row_context = None
    else:
        return {"error": f"Unsupported operation: {operation}"}

    return {
        "dataset_title": title,
        "operation": operation,
        "column": column,
        "result": result_value,
        "row_context": row_context,
        "total_rows": len(df),
    }


if __name__ == "__main__":
    result = run_structured_query("What is the highest gold grade recorded?", dataset_id=1)
    print(json.dumps(result, indent=2, default=str))
    