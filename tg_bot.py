from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from groq_chain import query_chain
from dotenv import load_dotenv
import os           

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Hi! I am what Siri is supposed to be. Ask a question and I will give you an answer')

async def get_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    answer = query_chain(update.message.text)
    await update.message.reply_text(answer)

def main(TOKEN):

    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, get_answer))

    application.run_polling()

if __name__ == '__main__':
    load_dotenv()
    TOKEN = os.getenv("TG_TOKEN")
    main(TOKEN)