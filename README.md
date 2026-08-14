# Agent — Modular AI Research Assistant

A production-grade AI agent built with LangChain + LangGraph combining RAG, real-time web search, persistent memory, and multimodal image analysis.

## What it does

- **RAG pipeline** — scrapes any URL, stores content in ChromaDB, answers grounded in real content
- **Real-time web search** via Tavily for live information retrieval
- **Persistent conversational memory** across sessions using LangGraph MemorySaver
- **Multimodal** — image analysis via OpenAI Vision and Azure OpenAI Vision APIs
- **Modular architecture** — each capability is independently swappable

## Stack

| Layer | Technology |
|-------|------------|
| Agent orchestration | LangChain + LangGraph |
| LLM | GPT-4o |
| Web search | Tavily |
| Vector store | ChromaDB |
| Embeddings | OpenAI text-embedding-3-small |
| Vision | OpenAI Vision / Azure OpenAI |

## Setup

```bash
git clone https://github.com/chafiaya920-del/Agent
cd Agent
pip install -r requirements.txt
```

Add a `.env`:
```
OPENAI_API_KEY=your_key
TAVILY_API_KEY=your_key
```

## Run

```bash
python agent-ai.py
```

```
You: scrape https://docs.langchain.com
You: What is a runnable in LangChain?
You: search for latest LangGraph agent patterns
```
