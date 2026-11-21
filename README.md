# Personal AI Agent

A free, token-lean AI agent with tool calling. Runs locally with Ollama and can switch to the Groq or Gemini free tiers. Use it from the terminal, a web UI, a REST API, or Telegram.

## Features
- Tools: web search, safe calculator, persistent notes (SQLite), sandboxed file reader, date/time
- Saved conversation history per session (survives restarts)
- Token savers: 6-message window, 400-token reply cap, trimmed tool output, 2-line system prompt
- One OpenAI-compatible client for every provider
- Tool calls and errors logged to `agent.log`
- Unit tests (`pytest`), Dockerfile, docker-compose

## Quick start
    pip install -r requirements.txt
    ollama pull qwen2.5:3b
    python agent.py

CLI commands: `/help`, `/clear`, `/model <name>`, `/tokens`, `exit`.

## Interfaces
| Interface | Install | Run |
|---|---|---|
| Web UI | `pip install -r requirements-extra.txt` | `streamlit run app.py` |
| REST API | (in requirements.txt) | `uvicorn api:app --reload` then open `/docs` |
| Telegram | `pip install -r requirements-extra.txt` | `python telegram_bot.py` |
| Docker | Docker Desktop | `docker compose up --build` |

### REST API
    curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"message\": \"What is 15% of 2480?\", \"session_id\": \"me\"}"

Set `AGENT_API_KEY` to require an `X-API-Key` header.

### Telegram
1. Create a bot with @BotFather and set `TELEGRAM_BOT_TOKEN`.
2. Run the bot, send `/start`, and it replies with your numeric Telegram ID.
3. Set `TELEGRAM_ALLOWED_IDS` to that ID and restart. Only allow-listed users can chat.

### Free cloud models
    set PROVIDER=groq
    set GROQ_API_KEY=your_key
    python agent.py

Use `export` instead of `set` on Mac/Linux. `PROVIDER=gemini` with `GEMINI_API_KEY` also works. Override the model with `MODEL=<name>`.

### Docker notes
The container reaches Ollama on your machine through `host.docker.internal`. If it cannot connect, start Ollama with `OLLAMA_HOST=0.0.0.0`.

## Tests
    pytest -q

## Security notes
- File access is limited to the `files/` folder.
- The calculator parses expressions instead of using `eval`.
- Never commit `.env`, `agent.db`, or API keys.
