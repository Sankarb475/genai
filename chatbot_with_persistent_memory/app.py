"""
Streamlit Chatbot with AzureChatOpenAI + persistent per-user memory (SQLite)
"""

import os
import sqlite3
import time
import json
import numpy as np
import streamlit as st
from typing import List, Dict, Optional

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# ---------------------------
# Config & LangChain Azure Clients
# ---------------------------

CHAT_DEPLOYMENT = "gpt-4"  # Azure chat deployment name
EMBED_DEPLOYMENT = "text-embedding-3-small-1"  # Azure embedding deployment name
AZURE_OPENAI_ENDPOINT = "<>"
AZURE_OPENAI_KEY = "<api key>"
API_VERSION = "2023-12-01-preview"
TOP_K = 5
DB_PATH = "chat_memory.db"

chat_model = AzureChatOpenAI(
    deployment_name=CHAT_DEPLOYMENT,
    api_version=API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    temperature=0
)

embedding_model = AzureOpenAIEmbeddings(
    deployment=EMBED_DEPLOYMENT,
    api_version=API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY
)

# ---------------------------
# DB Setup
# ---------------------------

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE,
        metadata TEXT,
        created_at REAL
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        role TEXT,
        text TEXT,
        created_at REAL
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message_id INTEGER,
        vector TEXT,
        dims INTEGER,
        created_at REAL
    )""")
    conn.commit()
    return conn

def store_user(conn, name, email, metadata=None):
    now = time.time()
    meta_json = json.dumps(metadata or {})
    try:
        conn.execute("INSERT INTO users (name, email, metadata, created_at) VALUES (?, ?, ?, ?)",
                     (name, email, meta_json, now))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.execute("UPDATE users SET name=?, metadata=? WHERE email=?",
                     (name, meta_json, email))
        conn.commit()
    return conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()[0]

def get_user_by_email(conn, email):
    row = conn.execute("SELECT id, name, email, metadata, created_at FROM users WHERE email=?", (email,)).fetchone()
    if row:
        return {"id": row[0], "name": row[1], "email": row[2], "metadata": json.loads(row[3] or "{}")}
    return None

def store_message(conn, user_id, role, text):
    now = time.time()
    cur = conn.execute("INSERT INTO messages (user_id, role, text, created_at) VALUES (?, ?, ?, ?)",
                       (user_id, role, text, now))
    conn.commit()
    return cur.lastrowid

def store_embedding(conn, message_id, vector):
    vec_json = json.dumps(vector)
    dims = len(vector)
    now = time.time()
    conn.execute("INSERT INTO embeddings (message_id, vector, dims, created_at) VALUES (?, ?, ?, ?)",
                 (message_id, vec_json, dims, now))
    conn.commit()

def fetch_user_messages(conn, user_id):
    rows = conn.execute("SELECT role, text, created_at FROM messages WHERE user_id=? ORDER BY created_at ASC", (user_id,)).fetchall()
    return [{"role": r[0], "text": r[1], "created_at": r[2]} for r in rows]

def fetch_all_embeddings_for_user(conn, user_id):
    rows = conn.execute("""
        SELECT e.message_id, e.vector, m.text
        FROM embeddings e
        JOIN messages m ON e.message_id = m.id
        WHERE m.user_id = ?
    """, (user_id,)).fetchall()
    return [(mid, np.array(json.loads(vec)), text) for mid, vec, text in rows]

# ---------------------------
# Azure LangChain Wrappers
# ---------------------------

def get_embedding(text):
    return embedding_model.embed_query(text)

def chat_completion(messages):
    reply = chat_model(messages).content
    return reply

# ---------------------------
# Memory Retrieval
# ---------------------------

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def retrieve_relevant_history(conn, user_id, query, top_k=TOP_K):
    stored = fetch_all_embeddings_for_user(conn, user_id)
    if not stored:
        return []
    q_vec = np.array(get_embedding(query))
    scored = [(mid, cosine_similarity(q_vec, vec), text) for mid, vec, text in stored]
    top = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
    return [{"text": t, "score": s} for _, s, t in top]

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Azure Chatbot with Memory", layout="wide")
st.title("💬 Azure OpenAI Chatbot with Persistent Memory (LangChain)")

conn = init_db()

with st.sidebar:
    st.header("User Login")
    name = st.text_input("Name", value=st.session_state.get("name", ""))
    email = st.text_input("Email", value=st.session_state.get("email", ""))
    notes = st.text_area("Notes", value=st.session_state.get("notes", ""))
    if st.button("Sign In / Register"):
        if not name or not email:
            st.warning("Enter both name and email.")
        else:
            uid = store_user(conn, name, email, {"notes": notes})
            st.session_state.update({"name": name, "email": email, "user_id": uid, "notes": notes})
            st.success(f"Welcome {name}!")
            st.rerun()

if "user_id" not in st.session_state:
    st.info("Sign in to start chatting.")
    st.stop()

user = get_user_by_email(conn, st.session_state["email"])

query = st.text_area("Your message", height=100)
if st.button("Send"):
    mid = store_message(conn, user["id"], "user", query)
    store_embedding(conn, mid, get_embedding(query))

    relevant = retrieve_relevant_history(conn, user["id"], query)
    memory_prompt = "No prior history." if not relevant else "\n".join(
        [f"- {r['text']} (score {r['score']:.2f})" for r in relevant]
    )

    system_msg = SystemMessage(content="You are a helpful assistant.")
    memory_msg = SystemMessage(content=f"User history:\n{memory_prompt}")
    user_msg = HumanMessage(content=query)

    reply = chat_completion([system_msg, memory_msg, user_msg])
    aid = store_message(conn, user["id"], "assistant", reply)
    store_embedding(conn, aid, get_embedding(reply))

    st.session_state["last_reply"] = reply
    st.rerun()

if "last_reply" in st.session_state:
    st.markdown(f"**Assistant:** {st.session_state['last_reply']}")

st.markdown("---")
st.subheader("Conversation History")
for msg in fetch_user_messages(conn, user["id"]):
    role = "🧑" if msg["role"] == "user" else "🤖"
    st.markdown(f"{role} **{msg['role'].capitalize()}**: {msg['text']}")
