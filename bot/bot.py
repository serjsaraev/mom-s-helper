# coding=utf8

import torch, torchvision
import os
from PIL import Image
import aiohttp
import asyncio
import logging
from aiogram import Bot, Dispatcher, executor, types
from static_text import HELLO_TEXT, NON_TARGET_TEXT, WAITING_TEXT, NON_TARGET_CONTENT_TYPES, CLASSES, CLASSES_DICT
from predictor import predict

#Configure logging
logging.basicConfig(level=logging.INFO)

#Loding Telegram token from env
TOKEN = os.getenv('TOKENBOT')
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

#Base command for start bot
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    user_name = message.from_user.first_name
    user_id = message.from_user.id
    text = HELLO_TEXT %user_name
    logging.info(f'First start from user_name = {user_name}, user_id = {user_id}')
    await message.reply(text)

@dp.message_handler(content_types=NON_TARGET_CONTENT_TYPES)
async def handle_docs_photo(message):
    user_name = message.from_user.first_name
    text = NON_TARGET_TEXT %user_name
    await message.reply(text)

@dp.message_handler(content_types=['photo'])

async def handle_docs_photo(message):
    chat_id = message.chat.id

    if message.media_group_id is None:
        # Get user's variables
        user_name = message.from_user.first_name
        user_id = message.from_user.id
        message_id = message.message_id
        text = WAITING_TEXT %user_name
        logging.info(f'{user_name, user_id} is knocking to our bot')
        await bot.send_message(chat_id, text)

        # Define input photo local path
        photo_name = './input/photo_%s_%s.jpg' %(user_id, message_id)
        await message.photo[-1].download(photo_name) # extract photo for further procceses

        #Photo processing
        photo_output, text = predict(photo_name)       
        await bot.send_photo(chat_id, photo_output)
        output_text = []
        for i in text:
            output_text.append(CLASSES_DICT[i])
        output_text = '\n\n'.join(output_text)
        await bot.send_message(chat_id, output_text)

    else:
        text = NOT_TARGET_TEXT %user_name
        await message.reply(text)

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)

