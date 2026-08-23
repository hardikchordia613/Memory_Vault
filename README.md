# Codebase Memory Vault

A lightweight, local-first Retrieval-Augmented Generation (RAG) system for preserving and querying codebase context, architectural decisions, and code snippets using PostgreSQL (`pgvector`) and Google Gemini embeddings (`text-embedding-004`).

---

## 🚀 Features

- **Zero Cloud Infrastructure Cost**: PostgreSQL + `pgvector` runs completely locally in Docker with persistent volume mounts.
- **Dense Vector Search**: Powered by Google Gemini `text-embedding-004` (768 dimensions) and PostgreSQL HNSW cosine distance indexing (`<=>`).
- **Developer Context Memory**: Push code snippets alongside the rationale (why it was written that way).
- **Fast CLI Interface**: Simple subcommands (`push`, `ask`, `list`, `doctor`) with clear, formatted terminal output.
- **Zero Bloat**: Lightweight design with direct `psycopg2` drivers, no heavyweight frameworks or ORMs.

---

## 🛠️ Prerequisites

- [Docker](https://www.docker.com/) & Docker Compose
- Python 3.10+
- Google Gemini API Key (get one from [Google AI Studio](https://aistudio.google.com/))

---

## 📦 Setup & Installation

### 1. Start the PostgreSQL Vector Database
```bash
docker compose up -d
```

### 2. Configure Environment Variables
```bash
cp .env.example .env
# Edit .env and set your GEMINI_API_KEY
```

### 3. Create Virtual Environment & Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Verify System Health
```bash
python -m vault.cli doctor
```

---

## 💻 CLI Usage

### Push a Code Memory (`push`)
Save code snippets along with the architectural reasoning behind them:

```bash
# Provide code snippet inline
python -m vault.cli push \
  --context "Switched to PostgreSQL with pgvector for efficient vector storage" \
  --code "import psycopg2; from pgvector.psycopg2 import register_vector" \
  --file "vault/db.py"

# Or ingest directly from a file in your project
python -m vault.cli push \
  --context "Implemented Gemini text-embedding-004 wrapper using google-genai SDK" \
  --file "vault/embedder.py"
```

### Query Context & Code (`ask`)
Perform semantic similarity search over historical decisions and snippets:

```bash
# Ask questions in natural language
python -m vault.cli ask "Why did we switch to PostgreSQL?"

# Limit the number of returned results
python -m vault.cli ask "How are embeddings generated?" --limit 3
```

### List Stored Memories (`list`)
View all indexed memories:

```bash
python -m vault.cli list --limit 10
```

---

## 🗄️ Database Schema

The system uses a single table `code_memories` in PostgreSQL:

```sql
CREATE TABLE code_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_path VARCHAR(512),
    developer_context TEXT NOT NULL,
    raw_code TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_code_memories_embedding 
ON code_memories 
USING hnsw (embedding vector_cosine_ops);
```

---

## 🧪 Running Unit Tests

Run the complete test suite with coverage report:

```bash
pytest -v --cov=vault
```
