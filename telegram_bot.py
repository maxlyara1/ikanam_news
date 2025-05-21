import logging
from telegram.ext import Application, CommandHandler

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

async def start(update, context):
    """Sends a welcome message when the /start command is issued."""
    logger.info(f"Command /start received from user {update.effective_user.id if update.effective_user else 'Unknown'}")
    await update.message.reply_text("Welcome to the RIAN RU News Forwarder Bot!")

async def get_news_from_rian_ru(bot):
    """
    Fetches news from the RIAN RU channel.
    
    NOTE: This is a placeholder implementation.
    The actual implementation will require a Telegram client library (like Telethon or Pyrogram)
    to fetch messages from t.me/rian_ru, which would involve API ID and hash.
    """
    logger.info("Attempting to fetch news from RIAN RU (mock implementation).")
    # TODO: Replace with actual news fetching logic
    return ["Mock RIAN news 1: Details about event A.", "Mock RIAN news 2: Update on situation B."]

def preprocess_news(news_text: str) -> dict:
    """
    Preprocesses the news text.

    NOTE: This is a placeholder implementation.
    More sophisticated preprocessing logic (e.g., text cleaning, entity extraction,
    actual classification model integration) will be implemented later.
    """
    return {
        "text": news_text,
        "multi_labels": ["Общество", "Происшествия"],  # Example labels
        "hier_label": ["Общество", "Жизнь"]  # Example labels
    }

async def send_formatted_news(bot, preprocessed_news_item: dict):
    """
    Formats and sends the preprocessed news item to the target channel.
    """
    try:
        news_text = preprocessed_news_item["text"]
        multi_labels_str = ", ".join(preprocessed_news_item["multi_labels"])
        hier_label_str = ", ".join(preprocessed_news_item["hier_label"])

        message = (
            f"Основные классы - {multi_labels_str}, "
            f"иерархические классы - {hier_label_str}. {news_text}"
        )
        
        # Ensure TARGET_CHANNEL_ID is accessible, it's defined globally
        await bot.send_message(chat_id=TARGET_CHANNEL_ID, text=message)
        logger.info(f"Sent message to channel {TARGET_CHANNEL_ID}")

    except Exception as e: # Catching a general Exception for now, can be telegram.error.TelegramError
        logger.error(f"Error sending message to {TARGET_CHANNEL_ID}: {e}")

def main() -> None:
    """Start the bot."""
    logger.info("Bot is starting...")

    # Create the Application and pass it your bot's token.
    application = Application.builder().token(BOT_TOKEN).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("testnews", test_news_command))

    # TODO: Implement rian_ru channel listener here

    # Run the bot until the user presses Ctrl-C
    try:
        logger.info("Starting bot polling...")
        application.run_polling()
    except KeyboardInterrupt:
        logger.info("Bot stopped manually.")
    except Exception as e:
        logger.critical(f"Bot polling loop crashed: {e}", exc_info=True)

async def test_news_command(update, context):
    """Handles the /testnews command."""
    logger.info(f"Command /testnews received from user {update.effective_user.id if update.effective_user else 'Unknown'}")
    test_item = {
        "text": "Это тестовая новость для проверки работы бота.",
        "multi_labels": ["Тест", "Система"],
        "hier_label": ["Тест", "Проверка"]
    }
    await send_formatted_news(context.bot, test_item)
    await update.message.reply_text(
        "Тестовая новость отправлена в целевой канал, если TARGET_CHANNEL_ID настроен правильно."
    )
    logger.info("Test news processed and response sent to user.")

if __name__ == "__main__":
    main()
