# db.py

from __future__ import annotations

import os
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
from psycopg.types.json import Json

load_dotenv()


# --------------------------------------------------
# Connection
# --------------------------------------------------

def get_conn():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL not found.")

    return psycopg.connect(
        database_url,
        connect_timeout=5,
    )


# --------------------------------------------------
# Init table
# --------------------------------------------------

def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS blogs (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    payload JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
        conn.commit()


# --------------------------------------------------
# Save blog (FULL PAYLOAD)
# --------------------------------------------------

def save_blog(title: str, payload: dict) -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO blogs (title, payload)
                VALUES (%s, %s)
                RETURNING id;
                """,
                (title, Json(payload)),
            )
            blog_id = cur.fetchone()[0]
        conn.commit()

    return blog_id


# --------------------------------------------------
# Fetch all blogs
# --------------------------------------------------

def get_all_blogs():
    try:
        with get_conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("""
                    SELECT id, title, created_at
                    FROM blogs
                    ORDER BY created_at DESC;
                """)
                return cur.fetchall()
    except psycopg.errors.UndefinedTable:
        # Table doesn't exist yet
        return []

# --------------------------------------------------
# Fetch single blog
# --------------------------------------------------

def get_blog(blog_id: int):
    try:
        with get_conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT * FROM blogs WHERE id = %s;",
                    (blog_id,),
                )
                return cur.fetchone()
    except psycopg.errors.UndefinedTable:
        return None