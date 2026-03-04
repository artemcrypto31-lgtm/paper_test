import os
import sqlite3
import telebot
from telebot import types
from dotenv import load_dotenv

# --- ЗАГРУЗКА НАСТРОЕК ---
load_dotenv()
BOT_TOKEN = os.getenv('BOT_TOKEN')
USER_ID = os.getenv('USER_ID')

# Проверяем, всё ли заполнено в .env
if not BOT_TOKEN or not USER_ID:
    print("❌ ОШИБКА: BOT_TOKEN или USER_ID не найдены в файле .env!")
    exit()

# Инициализируем бота
bot = telebot.TeleBot(BOT_TOKEN)
USER_ID = int(USER_ID)

# --- БАЗА ДАННЫХ ---
def get_db_conn():
    # Подключаемся к нашему виртуальному банку
    return sqlite3.connect('paper_trading.db', check_same_thread=False)

# --- СИСТЕМА БЕЗОПАСНОСТИ ---
def is_admin(message):
    """Проверка: общается ли с ботом хозяин."""
    if message.from_user.id == USER_ID:
        return True
    bot.send_message(message.chat.id, "❌ Доступ запрещен. Бот находится в приватном режиме тестирования.")
    return False

# --- ИНТЕРФЕЙС И КОМАНДЫ ---
@bot.message_handler(commands=['start'])
def welcome(message):
    if not is_admin(message): 
        return
    
    # Создаем удобные кнопки внизу экрана
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add("💰 Баланс", "⚔️ Сделки")
    
    welcome_text = (
        "👋 Приветствую, Архитектор!\n\n"
        "Система Paper Trading V1.0 активна.\n"
        "Режим работы: 🟢 БЕЗОПАСНЫЙ (Виртуальные средства).\n\n"
        "Используй кнопки ниже для управления терминалом."
    )
    bot.send_message(message.chat.id, welcome_text, reply_markup=markup)

@bot.message_handler(func=lambda m: m.text == "💰 Баланс")
def show_balance(message):
    if not is_admin(message): 
        return
    
    # Идем в базу данных запрашивать виртуальные деньги
    try:
        conn = get_db_conn()
        cash = conn.execute('SELECT balance FROM wallet').fetchone()[0]
        conn.close()
        
        msg = (
            f"💵 **ВИРТУАЛЬНЫЙ СЧЕТ:**\n\n"
            f"Доступно: **{cash:.2f} USDT**\n"
            f"Статус: Готов к симуляции торгов."
        )
        bot.send_message(message.chat.id, msg, parse_mode="Markdown")
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Ошибка доступа к банку: {e}")

# --- ЗАПУСК ---
if __name__ == '__main__':
    print("🚀 Telegram-бот успешно запущен! Открой Telegram и напиши ему /start")
    # Команда infinity_polling заставляет бота работать непрерывно и ждать твоих сообщений
    bot.infinity_polling()