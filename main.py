import os
import faiss
import pandas as pd
import torch
import numpy as np
from sqlalchemy import create_engine, text
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from dotenv import load_dotenv
import regex as re
import json

# Load environment variables
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT")

# Create database connection
DB_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DB_URI)

# Load FAISS-based Schema Retrieval Model (Grappa)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
grappa_model_name = "Salesforce/grappa_large_jnt"
grappa_tokenizer = AutoTokenizer.from_pretrained(grappa_model_name)
grappa_model = AutoModel.from_pretrained(grappa_model_name).to(device)

# Load SQL Generation Model (Qwen2.5-Coder-1.5B-Instruct)
qwen_model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)
qwen_model = AutoModelForCausalLM.from_pretrained(
    qwen_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

#  Initialize FAISS Index
index = faiss.IndexFlatL2(1024)
schema_map = {}
table_keywords = {}

#  Functions

def extract_schema_from_dataframe(df, table_name, engine):
    column_types = []
    column_texts = []

    for col in df.columns:
        dtype = df[col].dtype
        sql_type = "TEXT"
        if pd.api.types.is_integer_dtype(dtype):
            sql_type = "INTEGER"
        elif pd.api.types.is_float_dtype(dtype):
            sql_type = "FLOAT"
        elif pd.api.types.is_bool_dtype(dtype):
            sql_type = "BOOLEAN"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            sql_type = "TIMESTAMP"

        column_types.append(f'"{col}" {sql_type}')
        column_texts.append(f"{table_name}.{col}")

    create_table_query = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join(column_types)});'
    with engine.connect() as conn:
        conn.execute(text(create_table_query))
        conn.commit()

    print(f"✅ Table '{table_name}' created successfully!")

    # Build keywords
    keywords = set()
    keywords.update(table_name.lower().replace('_', ' ').split())
    for col in df.columns:
        keywords.update(col.lower().replace('_', ' ').split())
    table_keywords[table_name] = list(keywords)  # <-- NEW

    # FAISS embedding
    schema_text = " ".join(column_texts)
    inputs = grappa_tokenizer(schema_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = grappa_model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    faiss.normalize_L2(embedding)
    index.add(np.array(embedding, dtype=np.float32))
    schema_map[index.ntotal - 1] = {"table_name": table_name, "columns": df.columns.tolist()}


def load_excel_to_postgres(excel_path, engine):
    sheets = pd.read_excel(excel_path, sheet_name=None)
    for sheet_name, df in sheets.items():
        cleaned_sheet_name = sheet_name.strip().replace(' ', '_').lower()
        extract_schema_from_dataframe(df, cleaned_sheet_name, engine)
        df.to_sql(cleaned_sheet_name, engine, if_exists="replace", index=False, method="multi")
        print(f"✅ Data from sheet '{sheet_name}' loaded into table '{cleaned_sheet_name}' successfully!")

def find_relevant_schema(query, top_k=1):
    query_lower = query.lower()

    # Try keyword-based matching first
    for table_name, keywords in table_keywords.items():
        for kw in keywords:
            if kw in query_lower:
                print(f"\n⚡ Force-matching based on keyword '{kw}' → Table '{table_name}'")
                return [schema for schema in schema_map.values() if schema["table_name"] == table_name]

    # Otherwise fallback to FAISS matching
    inputs = grappa_tokenizer(query, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = grappa_model(**inputs)
        query_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    faiss.normalize_L2(query_embedding)
    D, I = index.search(np.array(query_embedding, dtype=np.float32), k=top_k)

    matched_schemas = []
    print("\n🔍 Top Matching Schemas:")
    for rank, (idx, dist) in enumerate(zip(I[0], D[0])):
        if idx in schema_map:
            schema_info = schema_map[idx]
            matched_schemas.append(schema_info)

            print(f"  {rank + 1}. Table: '{schema_info['table_name']}' | Distance: {dist:.4f}")
            print(f"     Columns: {', '.join(schema_info['columns'])}\n")

    return matched_schemas


def extract_sql_query(model_response, table_name):
    match = re.search(r"SELECT .*?;", model_response, re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError("No valid SQL query found in model response.")
    sql_query = match.group(0).strip()

    for schema in schema_map.values():
        if schema["table_name"] == table_name:
            columns = schema["columns"]
            break

    for col in columns:
        sql_query = re.sub(rf'\b{col}\b', f'"{col}"', sql_query)

    return sql_query

def generate_sql_query(nl_query):
    schema_context = find_relevant_schema(nl_query, top_k=1)
    if not schema_context:
        print("⚠ Warning: No relevant schema found.")
        return 'SELECT * FROM "default_table" LIMIT 5;'

    table_name = schema_context[0]["table_name"]
    columns = schema_context[0]["columns"]

    print(f"\n✅ Using Schema for Table: {table_name}")
    print(f"📌 Columns: {', '.join(columns)}")

    formatted_columns = ",\n".join([f'    "{col}" TEXT' for col in columns])

    query_prompt = f"""
    Convert the following natural language query into a valid PostgreSQL SQL query.

    ### Question:
    {nl_query}

    ### SQL Schema:
    CREATE TABLE {table_name} (
    {formatted_columns}
    );

    ### Constraints:
    - Use only the available columns in `{table_name}`.
    - Ensure valid PostgreSQL syntax.
    - Output only the SQL query and nothing else.
    """

    input_tokens = qwen_tokenizer(query_prompt, return_tensors="pt", truncation=True, max_length=512).to(qwen_model.device)

    with torch.no_grad():
        output_ids = qwen_model.generate(
            input_tokens.input_ids,
            attention_mask=input_tokens.attention_mask,
            max_new_tokens=150,
            num_beams=5,
        )

    raw_output = qwen_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    generated_query = extract_sql_query(raw_output, table_name)

    print("✅ Extracted SQL Query:\n", generated_query)
    return generated_query

def run_sql_query(sql_query):
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql_query))
            rows = result.fetchall()
            column_names = result.keys()
            return pd.DataFrame(rows, columns=column_names)
    except Exception as e:
        return {"error": str(e)}

def evaluate_sql_correctness(sql_query):
    try:
        with engine.connect() as conn:
            conn.execute(text(sql_query))
        return True
    except:
        return False

def reward_model(nl_query, sql_query):
    success = evaluate_sql_correctness(sql_query)
    if not success:
        return -1.0
    reward = 1.0 - (len(sql_query.split()) / 200)
    return round(reward, 4)

def normalize_sql(sql):
    return re.sub(r"\s+", " ", sql.strip().lower()).strip("; ")

def ppo_trainer(nl_query):
    sql = generate_sql_query(nl_query)
    reward = reward_model(nl_query, sql)

    with open("query_sql_reward_dataset.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"query": nl_query, "sql": sql, "reward": reward}) + "\n")

    print(f"\n🎯 Reward: {reward}\n")
    return sql, reward

# ✅ Load Excel into DB
excel_path = r"C:\Users\Pranav\Documents\multiple.xlsx"
load_excel_to_postgres(excel_path, engine)

# ✅ Main CLI loop
user_generated_queries = []
num_queries = 3

for i in range(num_queries):
    nl_query = input(f"\n Enter query {i + 1} of {num_queries}: ")
    final_query, reward = ppo_trainer(nl_query)
    user_generated_queries.append((nl_query, final_query))

    print("🧾 Final SQL Query:\n", final_query)

    result_df = run_sql_query(final_query)
    print("\nResults:\n", result_df)
