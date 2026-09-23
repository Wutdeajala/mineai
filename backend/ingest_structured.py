import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import json

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))


def infer_columns(df):
    """Builds a simple column-name + type description for the dataset."""
    columns = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        if "int" in dtype or "float" in dtype:
            col_type = "numeric"
        else:
            col_type = "text"
        columns.append({"name": col, "type": col_type})
    return columns


def ingest_structured_dataset(file_path, title, source_type="user_upload", user_id=None):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Only .csv and .xlsx files are supported.")

    columns = infer_columns(df)
    row_count = len(df)
    data_json = df.to_dict(orient="records")

    with engine.connect() as conn:
        result = conn.execute(
            text("""
                INSERT INTO structured_datasets (title, filename, source_type, user_id, columns, row_count, data)
                VALUES (:title, :filename, :source_type, :user_id, :columns, :row_count, :data)
                RETURNING id
            """),
            {
                "title": title,
                "filename": os.path.basename(file_path),
                "source_type": source_type,
                "user_id": user_id,
                "columns": json.dumps(columns),
                "row_count": row_count,
                "data": json.dumps(data_json),
            },
        )
        dataset_id = result.fetchone()[0]
        conn.commit()

    return dataset_id, columns, row_count


if __name__ == "__main__":
    dataset_id, columns, row_count = ingest_structured_dataset(
        file_path="../data/sample_assays.csv",
        title="Sample Assay Data",
        source_type="user_upload",
        user_id=0,
    )
    print(f"Created structured dataset id {dataset_id}")
    print(f"Row count: {row_count}")
    print(f"Columns: {columns}")