Here’s a Markdown guide you can share with your team:

---

# 💬 Running the Azure OpenAI Chatbot with Persistent Memory

This guide explains how to run the **Streamlit-based chatbot** connected to Azure OpenAI, with per-user persistent conversation memory stored in SQLite.

---

## 📦 Prerequisites

Before you start, make sure you have:

1. **Python 3.9+** installed
2. **pip** (Python package installer)
3. **Streamlit** and other dependencies installed (see below)
4. Access to:

   * An **Azure OpenAI deployment** for a chat model (e.g., `gpt-4`, `gpt-4o`, `gpt-35-turbo`)
   * An **Azure OpenAI deployment** for embeddings (e.g., `text-embedding-ada-002`, `text-embedding-3-large-1`)
   * A valid **Azure OpenAI API key** and **endpoint**

---

## ⚙️ Installation

1. **Clone or download** the chatbot code.

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```txt
streamlit
langchain
langchain-openai
openai
numpy
pandas
sqlite3-binary
```

---

## 🔑 Configure Azure OpenAI

The chatbot is hardcoded to use:

```python
chat_model = AzureChatOpenAI(
    deployment_name="gpt-4",  # Chat deployment name
    api_version="2023-12-01-preview",
    azure_endpoint="https://<your-endpoint>.openai.azure.com",
    api_key="<your-azure-openai-key>",
    temperature=0
)
```

For embeddings:

```python
embedding_model = AzureOpenAIEmbeddings(
    deployment="text-embedding-3-large-1",  # Your embedding deployment name
    model="text-embedding-3-large-1",
    api_version="2023-12-01-preview",
    azure_endpoint="https://<your-endpoint>.openai.azure.com",
    api_key="<your-azure-openai-key>"
)
```

Make sure the **deployment names match** your Azure OpenAI deployments.

---

## ▶️ Running the Chatbot

From the terminal, navigate to the chatbot’s directory and run:

```bash
streamlit run app.py
```

---

## 🖥 Using the Chatbot

1. Open the Streamlit app in your browser (Streamlit will display the URL, usually `http://localhost:8501`).
2. In the **sidebar**, enter your:

   * Name
   * Email
   * Optional notes
     Then click **"Sign In / Register"**.
3. Start chatting in the text area.
4. The chatbot will:

   * Store your messages in SQLite
   * Save embeddings for semantic memory retrieval
   * Use your past chats to improve responses when you return

---

## 📂 Data Storage

The chatbot stores user profiles, messages, and embeddings in `chat_memory.db` (SQLite).
You can inspect the database using tools like **DB Browser for SQLite**.

---

## 🚀 Example Run Command

```bash
 streamlit run /Users/sankarbiswas/genai/streamlit/repo/LangChain_Learning/concepts/Memory/memory_chatbot.py --server.port 8502
```

You can change the port if `8501` is already in use.

---

Do you want me to also make a **diagram** showing how the chatbot, SQLite, and Azure OpenAI interact? That would make the doc more visual.
