# Personal AI Agent

A free, token-lean AI agent with tool calling. Runs locally with Ollama and can switch to the Groq or Gemini free tiers.

## Features
- Tools: web search, calculator, persistent notes (SQLite), file reader, date/time
- Token savers: 6-message history window, 400-token reply cap, trimmed tool output
- One OpenAI-compatible client for all providers

## Setup
    pip install -r requirements.txt
    ollama pull qwen2.5:3b
    python agent.py

## Use a free cloud model
    set PROVIDER=groq
    set GROQ_API_KEY=your_key
    python agent.py
