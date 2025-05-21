import unittest
import asyncio # May be needed for running async tests
from unittest.mock import patch, MagicMock, AsyncMock
from telegram import Update
from telegram.ext import CallbackContext
# Assuming telegram_bot.py is in the same directory
from telegram_bot import preprocess_news, test_news_command, send_formatted_news 

class TestBot(unittest.TestCase):
    def test_preprocess_news(self):
        news_text = "This is a sample news text."
        result = preprocess_news(news_text)
        self.assertIsInstance(result, dict)
        self.assertIn("text", result)
        self.assertEqual(result["text"], news_text)
        self.assertIn("multi_labels", result)
        self.assertIsInstance(result["multi_labels"], list)
        self.assertTrue(all(isinstance(label, str) for label in result["multi_labels"]))
        self.assertIn("hier_label", result)
        self.assertIsInstance(result["hier_label"], list)
        self.assertTrue(all(isinstance(label, str) for label in result["hier_label"]))

    @patch('telegram_bot.send_formatted_news', new_callable=AsyncMock)
    def test_test_news_command(self, mock_send_formatted_news):
        # We need to run this async function.
        # unittest.IsolatedAsyncioTestCase could be an alternative for newer Python versions
        # For now, let's use asyncio.run() if it's a common way to test async with unittest
        async def run_test():
            mock_update = MagicMock(spec=Update)
            mock_update.message = MagicMock()
            mock_update.message.reply_text = AsyncMock()

            # For context, usually context.bot is the key part used
            mock_context = MagicMock(spec=CallbackContext)
            mock_context.bot = AsyncMock() # Mock the bot object within context

            await test_news_command(mock_update, mock_context)

            mock_send_formatted_news.assert_called_once()
            # Check the first argument (bot instance) of the call to send_formatted_news
            self.assertEqual(mock_send_formatted_news.call_args[0][0], mock_context.bot)
            # Check the second argument (the news item)
            sent_item = mock_send_formatted_news.call_args[0][1]
            self.assertEqual(sent_item['text'], "Это тестовая новость для проверки работы бота.")
            self.assertEqual(sent_item['multi_labels'], ["Тест", "Система"])
            self.assertEqual(sent_item['hier_label'], ["Тест", "Проверка"])
            
            mock_update.message.reply_text.assert_called_once_with(
                "Тестовая новость отправлена в целевой канал, если TARGET_CHANNEL_ID настроен правильно."
            )

        asyncio.run(run_test()) # Standard way to run async code from sync test

if __name__ == '__main__':
    unittest.main()
