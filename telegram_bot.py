"""
Telegram interface for the personal agent.

Setup:
  1. In Telegram, chat with @BotFather -> /newbot -> copy the token.
  2. set TELEGRAM_BOT_TOKEN=<token>
  3. python telegram_bot.py, then send /start to your bot. It replies with your numeric ID.
  4. set TELEGRAM_ALLOWED_IDS=<your id>   (comma-separate for several people), restart.

Only allow-listed IDs can talk to the bot, because the agent can read your notes and files.
"""
import asyncio
import os

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

import agent

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ALLOWED = {int(x) for x in os.getenv("TELEGRAM_ALLOWED_IDS", "").split(",") if x.strip()}


def is_allowed(update: Update) -> bool:
    return bool(update.effective_user and update.effective_user.id in ALLOWED)


async def deny(update: Update) -> None:
    await update.message.reply_text(
        f"Not authorised. Your Telegram ID is {update.effective_user.id}. "
        "Add it to TELEGRAM_ALLOWED_IDS and restart the bot."
    )


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        return await deny(update)
    await update.message.reply_text("Hi! Send me a message. /clear resets our chat.")


async def clear(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        return await deny(update)
    agent.clear_history(str(update.effective_chat.id))
    await update.message.reply_text("History cleared.")


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        return await deny(update)
    await ctx.bot.send_chat_action(update.effective_chat.id, "typing")
    try:
        reply = await asyncio.to_thread(agent.chat, str(update.effective_chat.id), update.message.text)
    except Exception as e:
        agent.log.error("telegram chat failed: %s", e)
        reply = f"Error: {e}"
    await update.message.reply_text((reply or "(empty reply)")[:4000])


def main() -> None:
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    print("Telegram bot running. Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()
