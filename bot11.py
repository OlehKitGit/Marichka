from telegram import InputMediaPhoto
import logging
import os
import torch
from fastai.vision.all import *
from pathlib import Path
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes, Updater
)

active_users = set()

# Ваш личный Telegram ID (узнать можно через @userinfobot)
ALLOWED_USER_ID = 334654605  # Замените на свой ID

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Конфигурация
BOT_TOKEN = "8083225695:AAEx6wfl1dfTIInhV6BIw73V2INMKma8aIY"  # Замените на токен бота

# Множество для хранения активных пользователей


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
   # user_id = update.effective_user.id

    # Проверка доступа
   # if user_id != ALLOWED_USER_ID:
   #     await update.message.reply_text("")
   #     return

    # Проверка, был ли пользователь уже активирован
    #if user_id  in active_users:
    #    await update.message.reply_text("❌ Вы уже активировали бота ранее.")
    #    return

    # Добавляем пользователя в множество
    #active_users.add(user_id)
   # else:
    await update.message.reply_text("✅ Know the Truth!")
    return

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    #user_id = update.effective_user.id

    # Блокировка для всех, кроме вас
    if user_id != ALLOWED_USER_ID:
        await update.message.reply_text("")
        return

    # Если пользователь не активирован через /start
   # if user_id not in active_users:
   #     await update.message.reply_text("⚠ Сначала нажмите /start")
   #     return
    # Если пользователь не активирован через /start
   # if user_id not in active_users:
   #     await update.message.reply_text("⚠ Сначала нажмите /start")
   #     return

    # Здесь будет ваша логика загрузки картинок
    #await update.message.reply_text("🖼 Загружаем картинки...")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Логируем ошибки"""
    logging.error(f'Update {update} caused error {context.error}')

def main():
    # Создаем Application
    application = Application.builder().token(BOT_TOKEN).build()

    # Регистрируем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Регистрируем обработчик ошибок
    application.add_error_handler(error_handler)

    # Запускаем бота
    application.run_polling()

# === ЗАГРУЗКА МОДЕЛИ ===
# Путь к папке с данными
base_dir = Path('~/paintings/venv311/paintings').expanduser() 
models_dir = base_dir / 'models'
models_dir.mkdir(exist_ok=True)

# DataLoaders для модели
def get_data(bs=16, size=224):
    data = ImageDataLoaders.from_folder(
        base_dir,
        train='.',
        valid_pct=0.2,
        item_tfms=Resize(size),
        batch_tfms=Normalize.from_stats(*imagenet_stats),
        bs=bs,
        num_workers=0,
        valid_extensions=['.jpg' '.jpeg'],  # Укажите нужные форматы
        ignore_empty=True  # Пропускать пустые/поврежденные файлы
    )
    return data

data = get_data()
learn = vision_learner(
    data,
    arch="efficientnet_b0",
    metrics=[accuracy, error_rate],
    pretrained=True
)

learn= load_learner('/root/paintings/venv311/paintings/models/final_model4.pkl')

# === ОБРАБОТЧИК /start ===
#async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#    await update.message.reply_text(
#        "Приветствую! Загрузите ваше фото и я опишу по нему ваше состояние, его причину и последствия."
#    )

# === ОБРАБОТЧИК изображений ===
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    photo_file = await update.message.photo[-1].get_file()
    img_path = Path(f'temp_{update.message.from_user.id}.jpg')
    await photo_file.download_to_drive(str(img_path))

    # Запускаем классификацию
    img = PILImage.create(img_path)
    pred_class, pred_idx, outputs = learn.predict(img)

    # Записываем вероятности для всех классов
    probs = {cls_name: outputs[i].item() for i, cls_name in enumerate(learn.dls.vocab)}

    # Сортируем классы по убыванию вероятности
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top1_cls, top1_prob = sorted_probs[0]
    top2_cls, top2_prob = sorted_probs[1]

    tarot_dir = Path('/root/paintings/venv311/TarotE')

    def get_tarot_path(cls, prob):
        num = prob
        if cls in ["norm", "norm_child"]:
            # Прямая карта
            num = num - 77 if num > 77 else num  # корректируем для перевёрнутой карты
            fname = next(tarot_dir.glob(f"{num}.*_прямая*"), None)
            if fname == None:
                fname = next(tarot_dir.glob(f"{num}.*_перевёрнутая*"), None)
            print(fname)
        elif cls in ["schizo", "schizo_child"]:
            num = num - 77 if num > 77 else num  # корректируем для перевёрнутой карты
            fname = next(tarot_dir.glob(f"{num}.*_перевернутая*"), None)
            if fname == None:
                fname = next(tarot_dir.glob(f"{num}.*_перевёрнутая*"), None)
            print(fname)
        else:
            fname = None
        return fname

    # Находим пути к картам
    top2_prob = int(str(top1_prob)[2:4])
    print(top1_prob)
    card1 = get_tarot_path(top1_cls, top2_prob)
    print(top2_prob)
    top3_prob = int(str(top1_prob)[4:6])
    print(top3_prob)
    card2 = get_tarot_path(top1_cls, top3_prob)
    top4_prob = int(str(top1_prob)[6:8])
    print(top4_prob)
    card3 = get_tarot_path(top1_cls, top4_prob)


    # Отправляем карты
    if card1 and card2 and card3:
        await update.message.reply_text("Premise")
        await update.message.reply_photo(card1)
        await update.message.reply_text("Consequence")
        await update.message.reply_photo(card2)
        await update.message.reply_text("Step")
        await update.message.reply_photo(card3)
    else:
        await update.message.reply_text("Не удалось найти подходящие карты для вашего изображения.")

    img_path.unlink()  # удаляем файл после обработки


# === ОСНОВНАЯ ФУНКЦИЯ ===
def main() -> None:
    # Замените TOKEN на свой токен бота!
    TOKEN = "8083225695:AAEx6wfl1dfTIInhV6BIw73V2INMKma8aIY"

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    print("Бот запущен...")
    app.run_polling()

if __name__ == '__main__':
    main()
