from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher.filters import CommandStart
import os
from synthesis.handwrite import handwrite

from bot.data import config

bot = Bot(token=config.BOT_TOKEN, parse_mode=types.ParseMode.HTML)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)


@dp.message_handler(CommandStart())
async def bot_start(message: types.Message):
    print(message.from_user.id)
    await message.answer(f"Hi, {message.from_user.full_name}!"
                         f"Please fill form: http://192.168.68.104:5000/")


@dp.message_handler(state=None)
async def bot_echo(message: types.Message):
    user_nick = message.from_user.username
    npy_file = user_nick + ".npy"
    txt_file = user_nick + ".txt"

    print(os.listdir('./app/user_data/'))

    # проверяем на наличие файла с почерком
    if npy_file in os.listdir('./app/user_data/') and txt_file in os.listdir('./app/user_data/'):
        res_file = handwrite(user_nick, user_text=message.text) + ".png"
        await bot.send_photo(message.from_user.id, photo=open(res_file, "rb"),
                             reply_to_message_id=message.message_id)
    else:
        await message.answer("Save your handwriting first!")
