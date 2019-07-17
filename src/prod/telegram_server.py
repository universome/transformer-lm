import sys; sys.path.append('.')
import os
import logging

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

from src.prod.model_api import build_predict_fn

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


TOKEN = os.environ["TELEGRAM_API_TOKEN"]
REQUEST_KWARGS = {
    'proxy_url': 'socks5h://telers5.rsocks.net:1490',
    'urllib3_proxy_kwargs': {
        'username': 'RSocks',
        'password': 'RSforTG'
    }
}

predict = build_predict_fn(
    model_path='checkpoints/model.pt',
    model_config_path='configs/training.yml',
    vocab_path='checkpoints/vocab.pickle',
    bpes_path='data/generated/bpes-30000.txt',
    max_len=50
)

# Define a few command handlers. These usually take the two arguments bot and
# update. Error handlers also receive the raised TelegramError object in error.
def start(update, context):
    """Send a message when the command /start is issued."""
    update.message.reply_text('Ну допустим привет.')


def help(update, context):
    """Send a message when the command /help is issued."""
    update.message.reply_text('Не жди от меня помощи, человек.')


def echo(update, context):
    """Echo the user message."""
    update.message.reply_text(update.message.text)


def reply_by_model(update, context):
    logger.info(f'Received message "{update.message.text}"')
    update.message.reply_text(predict([update.message.text])[0])


def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def main():
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    # Make sure to set use_context=True to use the new context based callbacks
    # Post version 12 this will no longer be necessary
    updater = Updater(TOKEN, use_context=True, request_kwargs=REQUEST_KWARGS)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))

    # on noncommand i.e message - echo the message on Telegram
    dp.add_handler(MessageHandler(Filters.text, reply_by_model))

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
