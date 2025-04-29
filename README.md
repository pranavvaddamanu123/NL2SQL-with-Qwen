# NL2SQL-with-Qwen
Overview
This project is a hybrid natural language to SQL generation system designed to enable users to query relational databases using plain English without knowing SQL. It uses semantic retrieval with FAISS and Grappa embeddings for schema selection, and a transformer-based model (Qwen2.5-Coder-1.5B-Instruct) to generate PostgreSQL queries from natural language input. The generated queries are scored using a lightweight reward function, which can later be used to fine-tune the model in a reinforcement learning setup.

Key Features
Handles multi-table Excel files by creating separate SQL tables for each sheet.

Uses FAISS with Grappa embeddings to semantically match user queries with relevant schemas.

Keyword-based force matching to improve schema retrieval accuracy.

SQL generation using a code-optimized LLM (Qwen-2.5b-Coder-1.5b-Instruct).

Lightweight reward model to simulate reinforcement learning feedback.

Logs query, SQL, and reward for future fine-tuning


This code requires a connection to a PostgreSQL database to return query results. To connect to the database, provide your credentials in the format indicated in the.env file.
