"""
Discord interface for the personal agent.

Setup:
  1. discord.com/developers/applications -> New Application -> Bot tab.
  2. Reset Token and copy it. Turn ON "Message Content Intent". Save.
  3. OAuth2 -> URL Generator: scope "bot", permissions "Send Messages" + "Read Message History".
     Open the generated URL and add the bot to a private server you own.
  4. set DISCORD_BOT_TOKEN=<token>
  5. python discord_bot.py, then DM the bot (or @mention it in the server). It replies with your ID.
  6. set DISCORD_ALLOWED_IDS=<your id>   (comma-separate for several people), restart.

Only allow-listed IDs get answers, because the agent can read your notes and files.
"""
import asyncio
import os

import discord

import agent

TOKEN = os.environ["DISCORD_BOT_TOKEN"]
ALLOWED = {int(x) for x in os.getenv("DISCORD_ALLOWED_IDS", "").split(",") if x.strip()}

intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)


@bot.event
async def on_ready() -> None:
    print(f"Discord bot running as {bot.user}. Press Ctrl+C to stop.")


@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return
    is_dm = isinstance(message.channel, discord.DMChannel)
    if not is_dm and bot.user not in message.mentions:
        return  # in servers, only react when mentioned

    if message.author.id not in ALLOWED:
        await message.reply(
            f"Not authorised. Your Discord ID is {message.author.id}. "
            "Add it to DISCORD_ALLOWED_IDS and restart the bot."
        )
        return

    text = message.content
    for tag in (f"<@{bot.user.id}>", f"<@!{bot.user.id}>"):
        text = text.replace(tag, "")
    text = text.strip()
    if not text:
        return

    sid = f"discord-{message.author.id}"
    if text == "/clear":
        agent.clear_history(sid)
        await message.reply("History cleared.")
        return

    async with message.channel.typing():
        try:
            reply = await asyncio.to_thread(agent.chat, sid, text)
        except Exception as e:
            agent.log.error("discord chat failed: %s", e)
            reply = f"Error: {e}"
    await message.reply((reply or "(empty reply)")[:1900])


if __name__ == "__main__":
    bot.run(TOKEN)
