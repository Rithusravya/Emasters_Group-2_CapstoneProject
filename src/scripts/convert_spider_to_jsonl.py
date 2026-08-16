import json
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SPIDER_DIR = PROJECT_ROOT / "data" / "raw" / "Spider" / "spider_data"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_text_lines(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_schema_from_tables(tables_entry: Dict[str, Any]) -> str:
    """Converts one Spider tables.json entry into a simple schema string.
    """
    table_names = tables_entry.get("table_names_original", [])
    column_names = tables_entry.get("column_names_original", [])
    column_types = tables_entry.get("column_types", [])
    foreign_keys = tables_entry.get("foreign_keys", [])

    table_columns = {table_name: [] for table_name in table_names}

    # column_names[0] is Spider's "*" wildcard placeholder - always skipped.
    for column_index, (table_id, column_name) in enumerate(column_names):
        if column_index == 0:
            continue
        if table_id < 0 or table_id >= len(table_names):
            continue

        column_type = column_types[column_index] if column_index < len(column_types) else "text"
        table_columns[table_names[table_id]].append(f"{column_name} {column_type}")

    create_statements = [
        f"CREATE TABLE {table_name} (\n  " + ",\n  ".join(columns) + "\n);"
        for table_name, columns in table_columns.items()
        if columns
    ]

    for child_col_index, parent_col_index in foreign_keys:
        child_table_id, child_column = column_names[child_col_index]
        parent_table_id, parent_column = column_names[parent_col_index]
        child_table = table_names[child_table_id]
        parent_table = table_names[parent_table_id]

        create_statements.append(
            f"-- FOREIGN KEY: {child_table}.{child_column} REFERENCES {parent_table}.{parent_column}"
        )

    return "\n\n".join(create_statements)


def convert_spider_eval(dev_path: str, tables_path: str, pred_path: str, output_path: str) -> None:
    """Converts Spider dev.json + tables.json + pred_example.txt into one
    normalized JSONL file."""
    dev_path = Path(dev_path)
    tables_path = Path(tables_path)
    pred_path = Path(pred_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dev_data = load_json(dev_path)
    tables_data = load_json(tables_path)
    predictions = load_text_lines(pred_path)

    schema_lookup = {entry["db_id"]: build_schema_from_tables(entry) for entry in tables_data}

    records = []
    for index, item in enumerate(dev_data):
        db_id = item.get("db_id", "")
        pred_sql = predictions[index] if index < len(predictions) else ""

        records.append({
            "id": f"{db_id}_{index}",
            "task": "text_to_sql",
            "db_id": db_id,
            "question": item.get("question", ""),
            "gold_sql": item.get("query", ""),
            "pred_sql": pred_sql,
            "schema": schema_lookup.get(db_id, ""),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} Spider eval records to {output_path}")


if __name__ == "__main__":
    convert_spider_eval(
        dev_path=str(SPIDER_DIR / "dev.json"),
        tables_path=str(SPIDER_DIR / "tables.json"),
        pred_path=str(SPIDER_DIR / "pred_example.txt"),
        output_path=str(PROCESSED_DIR / "spider_eval.jsonl"),
    )
