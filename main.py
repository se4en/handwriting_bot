# here we start bot and app
from aiogram import executor

from app import flask_app
from bot.loader import dp
from bot.utils.notify_admins import on_startup_notify
from bot.utils.set_bot_commands import set_default_commands


async def on_startup(dispatcher):
    # Устанавливаем дефолтные команды
    await set_default_commands(dispatcher)

    # Уведомляет про запуск
    #await on_startup_notify(dispatcher)


if __name__ == "__main__":
    # start bot
    executor.start_polling(dp, on_startup=on_startup)
    # start app
    flask_app.run(debug=True, host="0.0.0.0")