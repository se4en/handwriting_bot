from aiogram import executor

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
