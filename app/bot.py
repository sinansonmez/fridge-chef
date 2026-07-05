import asyncio
import logging
from collections.abc import Awaitable, Callable

from google import genai
from telegram import Message, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config import load_config
from recipes import RecipeError, suggest_recipes

log = logging.getLogger(__name__)

# How long to wait after the last photo of an album before processing it.
ALBUM_SETTLE_SECONDS = 2.0
TELEGRAM_MESSAGE_LIMIT = 4096

HELP_TEXT = (
    "🧊 Buzdolabınızın fotoğrafını gönderin, size 3 tarif önereyim!\n"
    "İsterseniz fotoğrafa bir not ekleyin (örn. \"vejetaryen\" veya \"30 dakikada\").\n\n"
    "🧊 Send me a photo of your fridge and I'll suggest 3 recipes.\n"
    "Add a caption in English to get the recipes in English."
)

ERROR_TEXT = (
    "😞 Üzgünüm, şu anda tarif oluşturamadım. Her iki model de hata verdi.\n"
    "Lütfen biraz sonra tekrar deneyin.\n\nHata / Error:\n"
)


class AlbumBuffer:
    """Collects photos that belong to the same Telegram album (media group).

    Telegram delivers album photos as separate messages sharing a
    media_group_id, with no "album complete" signal. Each new photo resets a
    short timer; when it expires the album is processed as one batch.
    """

    def __init__(self) -> None:
        self._albums: dict[str, list[Message]] = {}
        self._timers: dict[str, asyncio.Task] = {}

    def add(
        self,
        key: str,
        message: Message,
        on_ready: Callable[[list[Message]], Awaitable[None]],
    ) -> None:
        self._albums.setdefault(key, []).append(message)
        if timer := self._timers.get(key):
            timer.cancel()
        self._timers[key] = asyncio.create_task(self._flush_later(key, on_ready))

    async def _flush_later(
        self, key: str, on_ready: Callable[[list[Message]], Awaitable[None]]
    ) -> None:
        try:
            await asyncio.sleep(ALBUM_SETTLE_SECONDS)
        except asyncio.CancelledError:
            return
        messages = self._albums.pop(key, [])
        self._timers.pop(key, None)
        if messages:
            await on_ready(messages)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    if message.media_group_id:
        buffer: AlbumBuffer = context.bot_data["album_buffer"]
        key = f"{message.chat_id}:{message.media_group_id}"
        buffer.add(key, message, lambda msgs: process_photos(context, msgs))
    else:
        await process_photos(context, [message])


async def process_photos(
    context: ContextTypes.DEFAULT_TYPE, messages: list[Message]
) -> None:
    chat_id = messages[0].chat_id
    caption = next((m.caption for m in messages if m.caption), None)
    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(_keep_typing(context, chat_id, stop_typing))
    try:
        images = [await _download_image(message) for message in messages]
        text = await suggest_recipes(
            context.bot_data["gemini_client"],
            context.bot_data["config"],
            images,
            caption,
        )
        stop_typing.set()
        for chunk in split_message(text):
            await context.bot.send_message(chat_id, chunk)
    except RecipeError as exc:
        stop_typing.set()
        await context.bot.send_message(chat_id, ERROR_TEXT + str(exc))
    except Exception:
        stop_typing.set()
        log.exception("Unexpected error while processing photos")
        await context.bot.send_message(
            chat_id, "😞 Beklenmeyen bir hata oluştu. / An unexpected error occurred."
        )
    finally:
        stop_typing.set()
        await typing_task


async def _keep_typing(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int, stop: asyncio.Event
) -> None:
    while not stop.is_set():
        try:
            await context.bot.send_chat_action(chat_id, ChatAction.TYPING)
        except Exception:
            log.debug("Failed to send typing action", exc_info=True)
        try:
            await asyncio.wait_for(stop.wait(), timeout=4)
        except asyncio.TimeoutError:
            pass


async def _download_image(message: Message) -> tuple[bytes, str]:
    if message.photo:
        file = await message.photo[-1].get_file()
        mime_type = "image/jpeg"
    else:
        file = await message.document.get_file()
        mime_type = message.document.mime_type or "image/jpeg"
    return bytes(await file.download_as_bytearray()), mime_type


def split_message(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    chunks = []
    while len(text) > limit:
        cut = text.rfind("\n\n", 0, limit)
        if cut <= 0:
            cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        chunks.append(text)
    return chunks


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(HELP_TEXT)


async def handle_other(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(HELP_TEXT)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    config = load_config()
    application = Application.builder().token(config.telegram_token).build()
    application.bot_data["config"] = config
    application.bot_data["gemini_client"] = genai.Client(api_key=config.gemini_api_key)
    application.bot_data["album_buffer"] = AlbumBuffer()

    # Messages from users outside the allowlist match no handler and are
    # silently ignored, so strangers cannot spend the Gemini quota.
    allowed = filters.User(user_id=list(config.allowed_user_ids))
    photo_like = filters.PHOTO | filters.Document.IMAGE
    application.add_handler(
        CommandHandler(["start", "help"], handle_start, filters=allowed)
    )
    application.add_handler(MessageHandler(allowed & photo_like, handle_photo))
    application.add_handler(
        MessageHandler(allowed & ~photo_like & ~filters.COMMAND, handle_other)
    )

    log.info(
        "Starting fridge-chef (main=%s, fallback=%s, %d allowed user(s))",
        config.main_model, config.fallback_model, len(config.allowed_user_ids),
    )
    application.run_polling(allowed_updates=["message"])


if __name__ == "__main__":
    main()
