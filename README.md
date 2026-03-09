# Agent — AI Web Research Assistant

A modular AI agent built with LangChain + LangGraph that combines web scraping, vector search, and conversational memory to answer questions about any website or topic.

## What it does

- **Scrape any URL** on demand and store the content in a local vector database (Chroma)
- **RAG pipeline** — retrieve relevant chunks and ground answers in real content
- **Web search** via Tavily for real-time information beyond scraped content
- **Persistent memory** across a conversation thread using LangGraph's `MemorySaver`
- **Image analysis** — separate modules for OpenAI and Azure OpenAI vision APIs

## Architecture

```
User input
   │
   ├─ "scrape <url>" → WebBaseLoader → CharacterTextSplitter → ChromaDB
   │                                          │
   │                                    RAG Chain (retriever + GPT-4o)
   │
   └─ general query → ReAct Agent (GPT-4o + Tavily search + memory)
```

## Setup

```bash
git clone https://github.com/chafiaya920-del/Agent
cd Agent
pip install -r requirements.txt  # or uv install
```

Create a `.env` file:
```env
OPENAI_API_KEY=your_key
TAVILY_API_KEY=your_key
```

## Run

```bash
python agent-ai.py
```

The agent starts an interactive loop. Example commands:

```
You: scrape https://docs.langchain.com
You: What is a runnable?
You: search for latest LangGraph updates
```

## Files

| File | Description |
|------|-------------|
| `agent-ai.py` | Main conversational agent (ReAct + RAG + memory) |
| `GenAi_Agent_MainFile.py` | Standalone agent v1 |
| `scrape-ai.py` | Web scraping utility |
| `Image-analyzer_Openai.py` | Image analysis via OpenAI Vision |
| `Image-analyzer_AzureOpenai.py` | Image analysis via Azure OpenAI |

## Stack

- **LangChain** + **LangGraph** — agent orchestration and memory
- **GPT-4o** — primary language model
- **Tavily** — real-time web search
- **ChromaDB** — local vector store for scraped content
- **OpenAI Embeddings** — `text-embedding-3-small`
