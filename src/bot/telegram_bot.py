import logging
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from telegram import Update, InputFile, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
import telegram.error # Добавлено для обработки ошибок PTB
from telethon import TelegramClient, events
import asyncio
import os
import html # Добавлено для экранирования HTML
import json # Добавлен для настроек пользователя
from typing import Optional

# Импорт Telethon entities для конвертации
from telethon.tl.types import (
    MessageEntityBold, MessageEntityItalic, MessageEntityCode,
    MessageEntityPre, MessageEntityTextUrl, MessageEntityMentionName,
    MessageEntityUnderline, MessageEntityStrike, MessageEntityBlockquote,
    MessageEntitySpoiler
)

from src.config import (
    BOT_TOKEN, TARGET_CHANNEL_ID, TELEGRAM_API_ID, TELEGRAM_API_HASH,
    RIAN_RU_CHANNEL_USERNAME, SHAP_ENABLED, SHAP_TOP_N_ML_EXPLAIN,
    SHAP_MAX_FEATURES_DISPLAY, SHAP_NSAMPLES
)
# Импортируем функции из main.py и сам модуль main для доступа к глобальным переменным
from src.core import main # Изменено
from src.core.main import load_model_artifacts, preprocess_news_with_model # Изменено

# Импортируем функцию генерации SHAP
from src.core.shap_explainer import generate_shap_plots # Изменено

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Глобальная блокировка для сериализации ресурсоемких задач генерации SHAP
shap_generation_lock = asyncio.Lock()

# --- User Preferences ---
USER_PREFERENCES_FILE = "data/user_data/user_preferences.json" # Изменено
user_settings = {} # Глобальный словарь для хранения настроек пользователей

def load_user_preferences():
    global user_settings
    try:
        if os.path.exists(USER_PREFERENCES_FILE):
            with open(USER_PREFERENCES_FILE, 'r', encoding='utf-8') as f:
                loaded_settings = json.load(f)
                # Ключи в JSON всегда строки, конвертируем их в int
                user_settings = {int(k): v for k, v in loaded_settings.items()}
            logger.info(f"Loaded user preferences from {USER_PREFERENCES_FILE}")
        else:
            user_settings = {}
            logger.info(f"{USER_PREFERENCES_FILE} not found. Starting with empty user preferences.")
    except Exception as e:
        logger.error(f"Error loading user preferences: {e}", exc_info=True)
        user_settings = {} # Сбрасываем к пустому словарю в случае ошибки

def save_user_preferences():
    global user_settings
    try:
        with open(USER_PREFERENCES_FILE, 'w', encoding='utf-8') as f:
            json.dump(user_settings, f, indent=4, ensure_ascii=False)
        logger.debug(f"Saved user preferences to {USER_PREFERENCES_FILE}") # Уровень DEBUG для сохранения
    except Exception as e:
        logger.error(f"Error saving user preferences: {e}", exc_info=True)

def get_user_setting(user_id: int, key: str, default_value=None):
    return user_settings.get(user_id, {}).get(key, default_value)

def update_user_setting(user_id: int, key: str, value):
    if user_id not in user_settings:
        user_settings[user_id] = {}
    user_settings[user_id][key] = value
    save_user_preferences()
# --- End User Preferences ---

# Telethon client instance (будет инициализирован в main_bot_logic)
rian_client = None

# Определим специальный объект-маркер для конца итерации
_SENTINEL = object()

# Определение функции telethon_entities_to_html
def telethon_entities_to_html(text: str, entities: list) -> str:
    """
    Конвертирует текст и список Telethon entities в HTML.
    """
    if not text: # Если текст пустой, ничего не делаем
        return ""
    if not entities:
        return html.escape(text)

    # Создаем список всех точек начала и конца сущностей, плюс начало и конец текста
    points = {0, len(text)}
    for entity in entities:
        points.add(entity.offset)
        points.add(entity.offset + entity.length)
    
    sorted_points = sorted(list(points))
    
    result_parts = []
    
    # Итерируемся по сегментам текста, определенным точками
    for i in range(len(sorted_points) - 1):
        start_idx = sorted_points[i]
        end_idx = sorted_points[i+1]
        
        if start_idx >= end_idx: # Пропускаем сегменты нулевой длины
            continue
            
        current_segment_text = text[start_idx:end_idx]
        # Сначала экранируем сегмент
        processed_segment_html = html.escape(current_segment_text)
        
        # Применяем теги для сущностей, которые покрывают текущий сегмент.
        # Сущности должны применяться от внешней к внутренней для правильного вложения.
        # Сортируем сущности: сначала по смещению (offset), затем по убыванию длины (чтобы внешние шли раньше)
        active_entities = []
        for entity in entities:
            if entity.offset <= start_idx and (entity.offset + entity.length) >= end_idx:
                active_entities.append(entity)
        
        # Сортируем активные сущности, чтобы внешние (более длинные, если начинаются одинаково) применялись первыми
        active_entities.sort(key=lambda e: (e.offset, -(e.offset + e.length)))

        for entity in active_entities:
            tag_name = None
            attributes_str = ""

            if isinstance(entity, MessageEntityBold): tag_name = "b"
            elif isinstance(entity, MessageEntityItalic): tag_name = "i"
            elif isinstance(entity, MessageEntityUnderline): tag_name = "u"
            elif isinstance(entity, MessageEntityStrike): tag_name = "s"
            elif isinstance(entity, MessageEntitySpoiler): tag_name = "tg-spoiler"
            elif isinstance(entity, MessageEntityBlockquote): tag_name = "blockquote"
            elif isinstance(entity, MessageEntityCode): # Inline code
                tag_name = "code"
            elif isinstance(entity, MessageEntityPre): # Code block
                lang = getattr(entity, 'language', '')
                # Для <pre><code>...</code></pre> структура немного другая
                code_attrs = f' class="language-{html.escape(lang)}"' if lang else ""
                processed_segment_html = f"<pre><code{code_attrs}>{processed_segment_html}</code></pre>"
                continue # Тег <pre><code> уже применен
            elif isinstance(entity, MessageEntityTextUrl):
                tag_name = "a"
                attributes_str = f' href="{html.escape(entity.url)}"'
            elif isinstance(entity, MessageEntityMentionName):
                # user_id доступен в entity.user_id
                tag_name = "a"
                attributes_str = f' href="tg://user?id={entity.user_id}"'
            
            if tag_name:
                processed_segment_html = f"<{tag_name}{attributes_str}>{processed_segment_html}</{tag_name}>"
                
        result_parts.append(processed_segment_html)
        
    return "".join(result_parts)

def _blocking_next(iterator):
    """Безопасно получает следующий элемент из синхронного итератора."""
    try:
        return next(iterator)
    except StopIteration:
        return _SENTINEL

# Пример текста для демонстрации
EXAMPLE_NEWS_TEXT = "В Москве сегодня прошла встреча на высшем уровне, посвященная вопросам международной безопасности и экономического сотрудничества. Обсуждались ключевые аспекты глобальной повестки дня."

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user: return
    user_id = update.effective_user.id
    logger.info(f"Command /start received from user {user_id}")

    # Инициализируем настройки пользователя, если их нет
    if not user_settings.get(user_id):
        update_user_setting(user_id, "subscribed", False) # По умолчанию не подписан
        update_user_setting(user_id, "enable_analysis", True) # Анализ по умолчанию включен, если подпишется
        logger.info(f"Initialized default settings for new user {user_id}")

    welcome_text = (
        "Привет! Я ваш персональный новостной ассистент.\n\n"
        "Я слежу за лентой РИА Новости, классифицирую каждую новость по категориям и темам, "
        "а также могу провести детальный текстовый анализ (если вы этого захотите).\n\n"
        "Используйте кнопки ниже, чтобы начать:")

    keyboard = [
        [InlineKeyboardButton("📰 Показать пример новости", callback_data="show_example_news")],
        [InlineKeyboardButton("⚙️ Мои настройки и подписка", callback_data="open_settings")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(welcome_text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)

async def send_example_news(update: Update, context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    """Отправляет пример обработанной новости."""
    logger.info(f"Sending example news to chat_id: {chat_id}")
    preprocessed_item = preprocess_news_with_model(EXAMPLE_NEWS_TEXT)
    html_display_text = html.escape(EXAMPLE_NEWS_TEXT)
    
    user_analysis_pref = get_user_setting(chat_id, "enable_analysis", True)

    await send_formatted_news(
        bot_instance=context.bot,
        chat_id_to_send=chat_id,
        news_text_for_display=html_display_text,
        news_text_for_model=EXAMPLE_NEWS_TEXT,
        preprocessed_item_data=preprocessed_item,
        user_specific_analysis_enabled=user_analysis_pref
    )

async def get_next_plot_item(iterator, loop, initial_message_id_for_log):
    """Асинхронно получает следующий элемент из синхронного генератора."""
    try:
        item = await loop.run_in_executor(None, _blocking_next, iterator)
        return item
    except Exception as e_exec_next: # Ошибки от самого run_in_executor или если _blocking_next перевыбросил
        logger.error(f"SHAP Task (get_next_plot_item): Error in run_in_executor for (msg/chat {initial_message_id_for_log}): {e_exec_next}", exc_info=True)
        return _SENTINEL

async def _generate_and_send_shap_plots_task(
    bot_instance,
    chat_id_to_send: int,
    reply_to_message_id_for_plots: int,
    news_text_for_shap: str,
    delete_plots_after_sending: bool = True 
):
    logger.info(f"SHAP Task: Starting generation for chat {chat_id_to_send}, replying to msg {reply_to_message_id_for_plots} using raw text: '{news_text_for_shap[:100]}...'")
    plots_sent_count = 0
    loop = asyncio.get_running_loop()
    generated_plot_paths = [] 
    try:
        # Захватываем блокировку перед началом генерации SHAP
        async with shap_generation_lock:
            logger.info(f"SHAP Task (Lock Acquired): Creating plot_iterator for msg (reply target) {reply_to_message_id_for_plots}.")
            plot_iterator = main.sync_generate_shap_plots_iterator(
                trained_model=main.model,
                tokenizer_for_shap=main.tokenizer,
                sentence_to_explain=news_text_for_shap, 
                mlb_loaded=main.mlb,
                idx_to_hier_map_loaded=main.idx_to_hier_map,
                device_for_shap=main.DEVICE,
                max_len_for_shap=main.MAX_LENGTH,
                top_n_ml_explain=SHAP_TOP_N_ML_EXPLAIN,
                max_features_to_display=SHAP_MAX_FEATURES_DISPLAY,
                nsamples_shap=SHAP_NSAMPLES,
                delete_on_yield=False # Важно: генератор не удаляет, этот таск управляет удалением
            )
            logger.info(f"SHAP Task (Lock Acquired): Created plot_iterator (type: {type(plot_iterator)}) for msg (reply target) {reply_to_message_id_for_plots}.")

            plot_counter_for_log = 0
            while True:
                plot_counter_for_log += 1
                logger.debug(f"SHAP Task (Lock Acquired, Loop {plot_counter_for_log}): Requesting next plot for msg (reply target) {reply_to_message_id_for_plots}")
                
                plot_item = None 
                try:
                    plot_item = await asyncio.wait_for(
                        get_next_plot_item(plot_iterator, loop, reply_to_message_id_for_plots), 
                        timeout=180.0  
                    )
                except asyncio.TimeoutError:
                    logger.error(f"SHAP Task (Lock Acquired, Loop {plot_counter_for_log}): Timeout (180s) waiting for plot_item from iterator for msg {reply_to_message_id_for_plots}. Breaking SHAP loop for this message.")
                    break 
                except Exception as e_get_item:
                     logger.error(f"SHAP Task (Lock Acquired, Loop {plot_counter_for_log}): Exception from get_next_plot_item for msg {reply_to_message_id_for_plots}: {e_get_item}", exc_info=True)
                     break 

                received_item_description = "_SENTINEL" if plot_item is _SENTINEL else f"type {type(plot_item)}, value {str(plot_item)[:100]}"
                logger.info(f"SHAP Task (Lock Acquired, Loop {plot_counter_for_log}): Received plot_item: {received_item_description} for msg (reply target) {reply_to_message_id_for_plots}")

                if plot_item is not _SENTINEL and not isinstance(plot_item, tuple):
                    logger.error(f"SHAP Task (Lock Acquired, Loop {plot_counter_for_log}): plot_item is not _SENTINEL and not a tuple. Got: {plot_item}. Breaking SHAP loop.")
                    break

                if plot_item is _SENTINEL:
                    logger.info(f"SHAP Task (Lock Acquired, Loop {plot_counter_for_log}): Iterator exhausted (received _SENTINEL) for msg (reply target) {reply_to_message_id_for_plots}.")
                    break
                
                plot_path, plot_title = plot_item
                generated_plot_paths.append(plot_path) 
                plots_sent_count += 1
                try:
                    if os.path.exists(plot_path):
                        with open(plot_path, 'rb') as photo_file:
                            await bot_instance.send_photo(
                                chat_id=chat_id_to_send,
                                photo=InputFile(photo_file),
                                reply_to_message_id=reply_to_message_id_for_plots # Отвечаем на указанное сообщение
                            )
                        logger.info(f"SHAP Task: Sent plot {plot_path} for msg (reply target) {reply_to_message_id_for_plots}")
                    else:
                        logger.warning(f"SHAP Task: Plot file {plot_path} not found for msg (reply target) {reply_to_message_id_for_plots}")
                except Exception as e_send_plot:
                    logger.error(f"SHAP Task: Error sending plot {plot_path} for msg (reply target) {reply_to_message_id_for_plots}: {e_send_plot}", exc_info=True)

    except Exception as e_shap_gen_task:
        logger.error(f"SHAP Task: Error during overall task for msg (reply target) {reply_to_message_id_for_plots}: {e_shap_gen_task}", exc_info=True)
    finally:
        # Блокировка освобождается автоматически при выходе из `async with shap_generation_lock:`
        logger.info(f"SHAP Task (Lock Released or End): Finalizing for msg (reply target) {reply_to_message_id_for_plots} (chat {chat_id_to_send}). plots_sent_count: {plots_sent_count}")
        
        if delete_plots_after_sending: 
            logger.info(f"SHAP Task: Cleaning up {len(generated_plot_paths)} generated plot(s) for msg (reply target) {reply_to_message_id_for_plots} (delete_plots_after_sending=True).")
            for plot_path_to_remove in generated_plot_paths:
                if os.path.exists(plot_path_to_remove):
                    try: 
                        os.remove(plot_path_to_remove)
                        logger.debug(f"SHAP Task: Removed plot {plot_path_to_remove}.")
                    except Exception as e_remove: 
                        logger.error(f"SHAP Task: Error removing {plot_path_to_remove}: {e_remove}")
        else:
            logger.info(f"SHAP Task: Plot cleanup skipped for msg (reply target) {reply_to_message_id_for_plots} (delete_plots_after_sending=False).")

        logger.info(f"SHAP Task: Fully completed for msg (reply target) {reply_to_message_id_for_plots}")

async def send_formatted_news(
    bot_instance,
    chat_id_to_send: int,
    news_text_for_display: str, 
    news_text_for_model: str,   
    preprocessed_item_data: dict,
    original_message_id_for_reply: Optional[int] = None,
    user_specific_analysis_enabled: bool = False,
    pre_generated_plot_paths: Optional[list[str]] = None 
):
    base_message_text_html = "" 
    sent_classification_message = None
    model_error = False

    try:
        multi_labels = preprocessed_item_data.get("multi_labels", ["Error: multi_labels missing"])
        hier_label_list = preprocessed_item_data.get("hier_label", ["Error: hier_label missing"])
        
        if any("Error:" in label for label in multi_labels + hier_label_list):
            model_error = True

        if model_error:
            base_message_text_html = (
                f"⚠️ <b>Ошибка: Модели машинного обучения не загружены или произошла ошибка при классификации.</b>\n\n"
                f"{news_text_for_display}" # Отображаем HTML текст
            )
        else:
            unique_category_tags_list = []
            if multi_labels:
                seen_temp_category_tags = set()
                for label in multi_labels:
                    tag = f"#{label.replace(' ', '_').lower()}"
                    if tag not in seen_temp_category_tags:
                        unique_category_tags_list.append(tag)
                        seen_temp_category_tags.add(tag)
            category_tags_str = " ".join(unique_category_tags_list) if unique_category_tags_list else "#новость"

            raw_hierarchy_tags_from_parts = []
            current_hier_label_str = hier_label_list[0] if hier_label_list and isinstance(hier_label_list[0], str) else None
            if current_hier_label_str and current_hier_label_str != "Нет данных" and not current_hier_label_str.startswith("Error:"):
                hier_parts = [part.strip() for part in current_hier_label_str.split('/')]
                for part in hier_parts:
                    raw_hierarchy_tags_from_parts.append(f"#{part.replace(' ', '_').lower()}")
            
            final_hierarchy_tags_list = []
            if raw_hierarchy_tags_from_parts:
                seen_temp_hierarchy_tags = set()
                category_tags_set_for_check = set(unique_category_tags_list)
                for hier_tag in raw_hierarchy_tags_from_parts:
                    if hier_tag not in category_tags_set_for_check and hier_tag not in seen_temp_hierarchy_tags:
                        final_hierarchy_tags_list.append(hier_tag)
                        seen_temp_hierarchy_tags.add(hier_tag)
            hierarchy_tags_str = " ".join(final_hierarchy_tags_list) if final_hierarchy_tags_list else ""

            base_message_text_html = f"{news_text_for_display}\n\n{html.escape(category_tags_str)}"
            if hierarchy_tags_str:
                base_message_text_html += f"\n{html.escape(hierarchy_tags_str)}"
        
        # --- 1. Отправка основного сообщения с классификацией --- 
        try:
            sent_classification_message = await bot_instance.send_message(
                chat_id=chat_id_to_send, 
                text=base_message_text_html, # Отправляем только базовую информацию
                parse_mode=ParseMode.HTML, 
                reply_to_message_id=original_message_id_for_reply
            )
            logger.info(f"Sent classification message {sent_classification_message.message_id} to {chat_id_to_send}.")
        except telegram.error.BadRequest as e:
            if "message to be replied not found" in str(e).lower() and original_message_id_for_reply:
                logger.warning(f"Could not reply to message {original_message_id_for_reply} (likely deleted). Sending classification without reply.")
                sent_classification_message = await bot_instance.send_message(
                    chat_id=chat_id_to_send,
                    text=base_message_text_html,
                    parse_mode=ParseMode.HTML,
                    reply_to_message_id=None 
                )
                logger.info(f"Sent classification message {sent_classification_message.message_id} (no reply) to {chat_id_to_send}.")
            else:
                logger.error(f"Error sending classification message to {chat_id_to_send}: {e}", exc_info=True)
                raise # Перевыбрасываем, если это не ошибка с ответом
        except Exception as e_send_initial:
            logger.error(f"Unexpected error sending classification message to {chat_id_to_send}: {e_send_initial}", exc_info=True)
            return # Если не удалось отправить основное сообщение, выходим

        if not sent_classification_message: # Если сообщение так и не было отправлено
            logger.error(f"Classification message was not sent to {chat_id_to_send}. Aborting SHAP part.")
            return

        # --- 2. Обработка и отправка SHAP-графиков (если нужно) --- 
        log_model_error = model_error # model_error уже определен выше
        log_all_artifacts_present = all([
            main.model is not None, main.tokenizer is not None, main.mlb is not None,
            main.idx_to_hier_map is not None, main.DEVICE is not None,
            hasattr(main, 'MAX_LENGTH') and main.MAX_LENGTH is not None
        ])
        
        shap_conditions_met = (SHAP_ENABLED and 
                               user_specific_analysis_enabled and 
                               not log_model_error and 
                               log_all_artifacts_present)
        
        logger.info(f"For message {sent_classification_message.message_id} to {chat_id_to_send}: SHAP_ENABLED={SHAP_ENABLED}, user_analysis_enabled={user_specific_analysis_enabled}, no_model_error={not log_model_error}, artifacts_present={log_all_artifacts_present} -> SHAP conditions met: {shap_conditions_met}")

        if shap_conditions_met:
            if pre_generated_plot_paths:
                logger.info(f"Sending {len(pre_generated_plot_paths)} pre-generated SHAP plots as reply to msg {sent_classification_message.message_id} in chat {chat_id_to_send}.")
                for plot_path in pre_generated_plot_paths:
                    try:
                        if os.path.exists(plot_path):
                            with open(plot_path, 'rb') as photo_file:
                                await bot_instance.send_photo(
                                    chat_id=chat_id_to_send,
                                    photo=InputFile(photo_file),
                                    reply_to_message_id=sent_classification_message.message_id # Отвечаем на сообщение с классификацией
                                )
                            logger.info(f"Sent pre-generated plot {plot_path} as reply to msg {sent_classification_message.message_id}")
                        else:
                            logger.warning(f"Pre-generated plot file {plot_path} not found for msg {sent_classification_message.message_id}")
                    except Exception as e_send_plot:
                        logger.error(f"Error sending pre-generated plot {plot_path} for msg {sent_classification_message.message_id}: {e_send_plot}", exc_info=True)
                # Редактирование исходного сообщения для SHAP больше не требуется
            else:
                # Запускаем генерацию SHAP в фоне, графики будут отправлены как ответы
                logger.info(f"Creating SHAP generation task for msg {sent_classification_message.message_id} (chat {chat_id_to_send}). Plots will be sent as replies.")
                asyncio.create_task(_generate_and_send_shap_plots_task(
                    bot_instance=bot_instance,
                    chat_id_to_send=chat_id_to_send,
                    reply_to_message_id_for_plots=sent_classification_message.message_id, # ID основного сообщения для ответа
                    news_text_for_shap=news_text_for_model,
                    delete_plots_after_sending=True # Обычно True для /testnews и одиночных генераций
                ))
                logger.info(f"SHAP generation task created for msg {sent_classification_message.message_id}. Raw text for SHAP: '{news_text_for_model[:100]}...'")
        else:
            logger.info(f"SHAP analysis not required or conditions not met for message {sent_classification_message.message_id} in chat {chat_id_to_send}.")

    except Exception as e:
        logger.error(f"Outer error in send_formatted_news for chat {chat_id_to_send}: {e}", exc_info=True)

async def handle_rian_news(event):
    # Логика, которая была в rian_telethon_event_processor (строки 706-800 в исходном файле)
    # теперь должна быть здесь. Убедимся, что все части присутствуют и адаптированы.

    logger.debug(f"HANDLE_RIAN_NEWS (from decorated Telethon handler for @{RIAN_RU_CHANNEL_USERNAME}): Event received. Type: {type(event)}")
    event_str_repr = str(event)
    logger.debug(f"HANDLE_RIAN_NEWS: Event data (brief): {event_str_repr[:300]}")
    if not event or not hasattr(event, 'message') or not event.message:
        logger.warning(f"HANDLE_RIAN_NEWS: Event has no 'message'. Skipping.")
        return

    raw_text_from_event = None
    entities_from_event = None
    display_text_for_sending = None 

    if hasattr(event.message, 'raw_text') and event.message.raw_text is not None:
        raw_text_from_event = event.message.raw_text
        entities_from_event = event.message.entities 
        display_text_for_sending = telethon_entities_to_html(raw_text_from_event, entities_from_event if entities_from_event else [])
    elif hasattr(event.message, 'text') and event.message.text is not None: 
        logger.warning("HANDLE_RIAN_NEWS: event.message.raw_text not available. Falling back to event.message.text.")
        raw_text_from_event = event.message.text 
        display_text_for_sending = html.escape(event.message.text)
    else:
        logger.warning(f"HANDLE_RIAN_NEWS: Message (type: {type(event.message)}) has no text. Skipping. Msg: {event.message}")
        return

    if display_text_for_sending is None:
         logger.warning(f"HANDLE_RIAN_NEWS: display_text_for_sending is None. Skipping. Raw: '{raw_text_from_event[:50] if raw_text_from_event else 'None'}'")
         return
    
    if not raw_text_from_event.strip():
        logger.info(f"HANDLE_RIAN_NEWS: Raw text is empty for event_id {event.message.id}. Skipping ML.")
        return 

    logger.info(f"HANDLE_RIAN_NEWS: Processing news event_id: {event.message.id}. Raw text: \"{raw_text_from_event[:100]}...\"")
    
    # Запускаем ML-предобработку в отдельном потоке, чтобы не блокировать основной цикл
    loop = asyncio.get_running_loop()
    start_time_ml = asyncio.get_event_loop().time() # Используем время цикла для более точного измерения асинхронных операций
    
    logger.info(f"HANDLE_RIAN_NEWS: Starting ML preprocessing in executor for event_id {event.message.id}.")
    preprocessed_item = await loop.run_in_executor(
        None,  # Использует ThreadPoolExecutor по умолчанию
        preprocess_news_with_model, # Синхронная функция
        raw_text_from_event # Аргументы для функции
    )
    duration_ml = asyncio.get_event_loop().time() - start_time_ml
    logger.info(f"HANDLE_RIAN_NEWS: ML preprocessing for event_id {event.message.id} completed in {duration_ml:.4f} seconds.")

    bot_to_send_with = getattr(event.client, 'bot_instance_for_sending', None)
    if not bot_to_send_with:
        logger.error(f"HANDLE_RIAN_NEWS: No bot_instance_for_sending for event_id {event.message.id}. Cannot send.")
        return

    # 2. Разослать новость подписчикам
    subscribers_to_notify_ids = [
        uid for uid, settings in user_settings.items() if settings.get("subscribed", False)
    ]

    if not subscribers_to_notify_ids:
        logger.info(f"HANDLE_RIAN_NEWS: No users subscribed for updates (event_id: {event.message.id}).")
    else:
        logger.info(f"HANDLE_RIAN_NEWS: Relaying news (event_id: {event.message.id}) to {len(subscribers_to_notify_ids)} subscribers.")
        # Создаем список задач для отправки новостей
        send_tasks = []
        for user_id_to_send in subscribers_to_notify_ids:
            user_prefers_analysis = get_user_setting(user_id_to_send, "enable_analysis", False) 
            
            logger.info(f"HANDLE_RIAN_NEWS: Creating task to send to subscriber {user_id_to_send} (analysis enabled: {user_prefers_analysis}) for event {event.message.id}")
            # Создаем задачу для каждой отправки
            task = asyncio.create_task(send_formatted_news(
                bot_instance=bot_to_send_with,
                chat_id_to_send=user_id_to_send,
                news_text_for_display=display_text_for_sending,
                news_text_for_model=raw_text_from_event, # Это текст для модели и для SHAP
                preprocessed_item_data=preprocessed_item,
                user_specific_analysis_enabled=user_prefers_analysis, 
                pre_generated_plot_paths=None # Ключевое изменение: SHAP будет генерироваться в send_formatted_news если нужно
            ))
            send_tasks.append(task)

        # Ожидаем завершения всех задач по отправке (опционально, но полезно для логгирования ошибок)
        if send_tasks:
            done, pending = await asyncio.wait(send_tasks, return_when=asyncio.ALL_COMPLETED)
            for task_result in done:
                try:
                    task_result.result() # Проверяем наличие исключений в завершенных задачах
                except Exception as e_task_send:
                    # Логгирование ошибки из конкретной задачи, если необходимо
                    # user_id можно извлечь, если передавать его в задачу или иметь доступ к аргументам
                    logger.error(f"HANDLE_RIAN_NEWS: Error in a sending task for event {event.message.id}: {e_task_send}", exc_info=False) # exc_info=False, т.к. это будет много логов
            if pending:
                 logger.warning(f"HANDLE_RIAN_NEWS: Some sending tasks for event {event.message.id} are still pending after wait. This should not happen with ALL_COMPLETED.")

        logger.info(f"HANDLE_RIAN_NEWS: Finished creating tasks for relaying news (event_id: {event.message.id}) to {len(subscribers_to_notify_ids)} subscribers.")

    logger.info(f"HANDLE_RIAN_NEWS: Completed processing for event_id: {event.message.id if event.message else 'N/A'}")

async def test_news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.effective_user: return
    user_id = update.effective_user.id
    logger.info(f"Command /testnews received from user {user_id}")
    
    plain_test_text = " ".join(context.args) if context.args else "Это тестовая новость для проверки работы классификатора."
    if context.args: logger.info(f"Custom text for /testnews: \"{plain_test_text}\"")

    preprocessed_item = preprocess_news_with_model(plain_test_text) # Модель получает чистый текст
    
    # Для отображения НЕ экранируем текст, позволяя пользователю вводить HTML
    html_display_text = plain_test_text
    
    chat_id_to_send = update.message.chat_id # Отвечаем в тот же чат для /testnews

    # Для /testnews мы можем либо всегда использовать глобальный SHAP_ENABLED,
    # либо также проверять настройку пользователя, если команда вызвана в приватном чате.
    # Для простоты, /testnews будет использовать настройку пользователя, если она есть, иначе - True.
    # SHAP_ENABLED все еще является мастер-переключателем.
    user_id_for_test = update.effective_user.id
    user_analysis_pref_for_test = get_user_setting(user_id_for_test, "enable_analysis", True)

    await send_formatted_news(
        bot_instance=context.bot,
        chat_id_to_send=chat_id_to_send,
        news_text_for_display=html_display_text,
        news_text_for_model=plain_test_text,
        preprocessed_item_data=preprocessed_item,
        original_message_id_for_reply=update.message.message_id,
        user_specific_analysis_enabled=user_analysis_pref_for_test # Используем настройку пользователя
    )
    logger.info("Test news processed. Result and Analysis (if enabled by global & user) handled by send_formatted_news.")

# --- New Command Handlers for Subscription and Settings ---
async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user: return
    user_id = update.effective_user.id
    
    is_subscribed = get_user_setting(user_id, "subscribed", False)
    analysis_enabled = get_user_setting(user_id, "enable_analysis", True) # По умолчанию анализ включен

    text = "⚙️ Ваши текущие настройки:\n\n"
    text += f"📬 Рассылка новостей: <b>{'Включена' if is_subscribed else 'Отключена'}</b>\n"
    text += f"📊 Детальный анализ новостей: <b>{'Включен' if analysis_enabled else 'Отключен'}</b>\n\n"
    text += "Вы можете изменить их с помощью кнопок ниже:"

    keyboard = []
    if is_subscribed:
        keyboard.append([InlineKeyboardButton("🔕 Отписаться от рассылки", callback_data="toggle_subscription")])
    else:
        keyboard.append([InlineKeyboardButton("🔔 Подписаться на рассылку", callback_data="toggle_subscription")])

    if analysis_enabled:
        keyboard.append([InlineKeyboardButton("📊 Отключить детальный анализ", callback_data="toggle_analysis")])
    else:
        keyboard.append([InlineKeyboardButton("📊 Включить детальный анализ", callback_data="toggle_analysis")])

    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)

async def settings_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query or not query.from_user: return
    await query.answer() # Подтверждение получения колбэка
    
    user_id = query.from_user.id
    action = query.data
    message_text_to_edit = ""

    if action == "toggle_subscription":
        current_status = get_user_setting(user_id, "subscribed", False)
        new_status = not current_status
        update_user_setting(user_id, "subscribed", new_status)
        # Если пользователь подписывается и у него нет настройки анализа, ставим по умолчанию True
        if new_status and get_user_setting(user_id, "enable_analysis") is None:
             update_user_setting(user_id, "enable_analysis", True)
        message_text_to_edit = f"🔔 Подписка на новости теперь <b>{'включена' if new_status else 'отключена'}</b>."
        logger.info(f"User {user_id} toggled subscription to {new_status}.")

    elif action == "toggle_analysis":
        current_status = get_user_setting(user_id, "enable_analysis", True)
        new_status = not current_status
        update_user_setting(user_id, "enable_analysis", new_status)
        message_text_to_edit = f"📊 Детальный анализ новостей теперь <b>{'включен' if new_status else 'отключен'}</b>."
        logger.info(f"User {user_id} toggled analysis to {new_status}.")
    
    elif action == "show_example_news":
        await query.message.reply_text("Сейчас покажу пример обработанной новости...")
        await send_example_news(update, context, chat_id=user_id)
        return 

    elif action == "open_settings":
        # Просто вызываем логику команды /settings для отображения меню
        # Мы не можем напрямую вызвать settings_command здесь так как ему нужен 'update' от команды
        # Поэтому дублируем логику или рефакторим settings_command, чтобы ее можно было вызвать с user_id
        is_subscribed = get_user_setting(user_id, "subscribed", False)
        analysis_enabled = get_user_setting(user_id, "enable_analysis", True)
        text = "⚙️ Ваши текущие настройки:\n\n"
        text += f"📬 Рассылка новостей: <b>{'Включена' if is_subscribed else 'Отключена'}</b>\n"
        text += f"📊 Детальный анализ новостей: <b>{'Включен' if analysis_enabled else 'Отключен'}</b>\n\n"
        text += "Вы можете изменить их с помощью кнопок ниже:"
        keyboard = []
        if is_subscribed:
            keyboard.append([InlineKeyboardButton("🔕 Отписаться от рассылки", callback_data="toggle_subscription")])
        else:
            keyboard.append([InlineKeyboardButton("🔔 Подписаться на рассылку", callback_data="toggle_subscription")])
        if analysis_enabled:
            keyboard.append([InlineKeyboardButton("📊 Отключить детальный анализ", callback_data="toggle_analysis")])
        else:
            keyboard.append([InlineKeyboardButton("📊 Включить детальный анализ", callback_data="toggle_analysis")])
        reply_markup = InlineKeyboardMarkup(keyboard)
        try:
            if query.message:
                 await query.edit_message_text(text=text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
            else: # Если исходного сообщения нет (маловероятно для callback от inline кнопки)
                 await context.bot.send_message(chat_id=user_id, text=text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
        except telegram.error.BadRequest as e:
            if "message is not modified" not in str(e).lower() and query.message:
                logger.warning(f"Error editing message for open_settings, sending new: {e}")
                await context.bot.send_message(chat_id=user_id, text=text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
            elif not query.message:
                 await context.bot.send_message(chat_id=user_id, text=text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
        return # Выходим, так как сообщение уже отредактировано/отправлено

    # Обновляем сообщение с настройками (этот блок теперь только для toggle_subscription и toggle_analysis)
    is_subscribed_after = get_user_setting(user_id, "subscribed", False)
    analysis_enabled_after = get_user_setting(user_id, "enable_analysis", True)

    new_settings_text = "⚙️ Ваши текущие настройки:\n\n"
    new_settings_text += f"📬 Рассылка новостей: <b>{'Включена' if is_subscribed_after else 'Отключена'}</b>\n"
    new_settings_text += f"📊 Детальный анализ новостей: <b>{'Включен' if analysis_enabled_after else 'Отключен'}</b>\n\n"
    if message_text_to_edit: # Добавляем подтверждение действия, если оно было
        new_settings_text = f"{message_text_to_edit}\n\n{new_settings_text}"
        
    keyboard_after = []
    if is_subscribed_after:
        keyboard_after.append([InlineKeyboardButton("🔕 Отписаться от рассылки", callback_data="toggle_subscription")])
    else:
        keyboard_after.append([InlineKeyboardButton("🔔 Подписаться на рассылку", callback_data="toggle_subscription")])

    if analysis_enabled_after:
        keyboard_after.append([InlineKeyboardButton("📊 Отключить детальный анализ", callback_data="toggle_analysis")])
    else:
        keyboard_after.append([InlineKeyboardButton("📊 Включить детальный анализ", callback_data="toggle_analysis")])
    
    reply_markup_after = InlineKeyboardMarkup(keyboard_after)
    
    try:
        if query.message: # Убедимся, что query.message существует
            await query.edit_message_text(text=new_settings_text, reply_markup=reply_markup_after, parse_mode=ParseMode.HTML)
    except telegram.error.BadRequest as e:
        if "message is not modified" in str(e).lower():
            logger.debug("Settings message not modified, skipping edit.")
        else:
            # Если сообщение не найдено (например, удалено), можно отправить новое
            if query.message: # Проверка на случай, если сообщение было удалено в момент выполнения
                 await context.bot.send_message(chat_id=user_id, text=new_settings_text, reply_markup=reply_markup_after, parse_mode=ParseMode.HTML)
            logger.error(f"Error editing settings message for user {user_id}: {e}", exc_info=True)
    except Exception as e_edit: # Другие возможные ошибки при редактировании
        logger.error(f"Unexpected error editing settings message for user {user_id}: {e_edit}", exc_info=True)
        if query.message: # Проверка на случай, если сообщение было удалено в момент выполнения
             await context.bot.send_message(chat_id=user_id, text=new_settings_text, reply_markup=reply_markup_after, parse_mode=ParseMode.HTML)
# --- End New Command Handlers ---

async def main_bot_logic():
    """Main logic to start PTB bot and Telethon client."""
    global rian_client

    logger.info("Starting main_bot_logic...")
    load_user_preferences() # Загружаем настройки пользователей при старте
    logger.info("Attempting to load ML models and artifacts...")
    models_loaded_successfully = load_model_artifacts() 
    if not models_loaded_successfully:
        logger.critical("Failed to load ML models. Bot will run with limited functionality or may fail.")
    else:
        logger.info("ML Models loaded successfully (according to load_model_artifacts return value)!")
    
    # Логирование состояния загруженных артефактов, которые будут использоваться для SHAP, используя main.
    logger.info(f"Loaded artifacts check for SHAP: model is {'SET' if main.model else 'NOT SET'}")
    logger.info(f"Loaded artifacts check for SHAP: tokenizer is {'SET' if main.tokenizer else 'NOT SET'}")
    logger.info(f"Loaded artifacts check for SHAP: mlb is {'SET' if main.mlb else 'NOT SET'}")
    logger.info(f"Loaded artifacts check for SHAP: idx_to_hier_map is {'SET' if main.idx_to_hier_map else 'NOT SET'}")
    logger.info(f"Loaded artifacts check for SHAP: DEVICE is {'SET' if main.DEVICE else 'NOT SET'} (Value: {main.DEVICE})")
    logger.info(f"Loaded artifacts check for SHAP: MAX_LENGTH is {'SET' if main.MAX_LENGTH else 'NOT SET'} (Value: {main.MAX_LENGTH})")
    logger.info(f"Config check for SHAP: SHAP_ENABLED is {SHAP_ENABLED}")

    # Create the Application and pass it your bot's token.
    # Убедимся, что BOT_TOKEN загружен
    if not BOT_TOKEN or BOT_TOKEN == "YOUR_BOT_TOKEN":
        logger.critical("BOT_TOKEN is not set or is a placeholder. Please configure it in config.py.")
        return
        
    application = Application.builder().token(BOT_TOKEN).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("testnews", test_news_command))

    # Добавляем новые обработчики
    application.add_handler(CommandHandler("settings", settings_command))
    application.add_handler(CallbackQueryHandler(settings_callback_handler))

    # --- Telethon client setup ---
    if not TELEGRAM_API_ID or not TELEGRAM_API_HASH or TELEGRAM_API_ID == 12345678 or TELEGRAM_API_HASH == "YOUR_TELEGRAM_API_HASH":
        logger.error("Telegram API_ID or API_HASH are not set or are placeholders in config.py. Cannot listen to RIAN_RU.")
    else:
        try:
            rian_client = TelegramClient('sessions/rian_listener_session', TELEGRAM_API_ID, TELEGRAM_API_HASH)
            rian_client.bot_instance_for_sending = application.bot

            logger.info(f"Connecting Telethon client to listen to @{RIAN_RU_CHANNEL_USERNAME}...")
            # Соединение и проверка авторизации будут выполнены позже, перед запуском run_until_disconnected
            # await rian_client.connect() # Перемещено
            # if not await rian_client.is_user_authorized(): # Перемещено
            #      logger.warning(f"Telethon client is not authorized. Please run a script to log in (e.g., python -m telethon) or provide phone/code.")
            # else:
            #     logger.info("Telethon client connected and authorized.")
            
            # Запускаем слушатель в фоне, чтобы не блокировать основной поток
            # asyncio.create_task(rian_client.run_until_disconnected()) # Старый вариант, будет заменен
            # async def telethon_runner(): # Новая обертка для логирования
            #     try:
            #         await rian_client.run_until_disconnected()
            #     except Exception as e_telethon_run:
            #         logger.error(f"Telethon client run_until_disconnected crashed: {e_telethon_run}", exc_info=True)
            #     finally:
            #         logger.warning("Telethon client run_until_disconnected task has finished.") # Лог о завершении
            # asyncio.create_task(telethon_runner()) # Запускаем новую обертку
            # logger.info(f"Telethon client task created and listening to @{RIAN_RU_CHANNEL_USERNAME}.")

        except Exception as e:
            logger.error(f"Error setting up Telethon client structure: {e}", exc_info=True)
            rian_client = None # Сбрасываем, если не удалось базово настроить

    # Run the PTB bot until the user presses Ctrl-C
    try:
        logger.info("Initializing PTB application...")
        await application.initialize()
        await application.start()
        await application.updater.start_polling() 
        logger.info("PTB Bot has started polling.")
        
        # Теперь запускаем Telethon после PTB, если он настроен
        if rian_client:
            logger.info("Starting Telethon client...")
            await rian_client.connect()
            if not await rian_client.is_user_authorized():
                logger.critical("Telethon client is NOT AUTHORIZED. Please ensure the session is valid or re-authorize.")
                # Можно решить остановить бота здесь, если Telethon критичен
            else:
                logger.info("Telethon client connected and authorized. Running until disconnected...")
                # Регистрируем обработчики прямо перед запуском клиента
                # Диагностический обработчик для ВСЕХ новых сообщений
                @rian_client.on(events.NewMessage())
                async def all_new_messages_handler_decorated(event):
                    try:
                        chat = await event.get_chat()
                        chat_identifier = f"ID={chat.id}, Username={getattr(chat, 'username', 'N/A')}, Title={getattr(chat, 'title', 'N/A')}"
                        logger.debug(f"ALL_NEW_MESSAGES_HANDLER (Decorated): Received NewMessage from Chat: {chat_identifier}. Text: '{event.text[:70] if event.text else None}'")
                    except Exception as e_all_msg_handler:
                        logger.error(f"ALL_NEW_MESSAGES_HANDLER (Decorated): Error processing event: {e_all_msg_handler}", exc_info=True)
                
                # Основной обработчик для РИА Новости
                @rian_client.on(events.NewMessage(chats=RIAN_RU_CHANNEL_USERNAME))
                async def rian_telethon_event_processor_decorated(event):
                    logger.debug(f"RIAN_EVENT_PROCESSOR (Decorated, for @{RIAN_RU_CHANNEL_USERNAME}): Event received. Type: {type(event)}")
                    # Копируем или вызываем основную логику из существующего rian_telethon_event_processor
                    # Чтобы не дублировать, предположим, что rian_telethon_event_processor остается как отдельная функция
                    await handle_rian_news(event) # Используем старую функцию handle_rian_news, которая теперь содержит всю логику

                logger.info(f"Telethon handlers registered. Running client for @{RIAN_RU_CHANNEL_USERNAME}...")
                await rian_client.run_until_disconnected()
                logger.warning("Telethon client run_until_disconnected has finished.")
        else:
            logger.warning("Telethon client was not initialized. RIAN news processing will be skipped.")

        # Держим основной поток живым, пока PTB работает (если Telethon не запущен или завершился)
        # Этот цикл может быть не нужен, если run_until_disconnected блокирующий и основной
        # Если PTB должен работать независимо от Telethon, то этот цикл имеет смысл
        # Если Telethon - основная задача, то этот цикл может быть избыточен.
        # Пока оставляем его, предполагая, что PTB должен продолжать работать.
        while True:
            await asyncio.sleep(3600) 

    except KeyboardInterrupt:
        logger.info("Bot stopped manually.")
    except Exception as e:
        logger.critical(f"Bot polling loop crashed: {e}", exc_info=True)
    finally:
        if rian_client and rian_client.is_connected:
            logger.info("Disconnecting Telethon client...")
            await rian_client.disconnect()
            logger.info("Telethon client disconnected.")
        if application and application.updater and application.updater.running: # Проверка перед остановкой
            logger.info("Stopping PTB bot...")
            await application.updater.stop()
            await application.stop()
            await application.shutdown() # Важно для новых версий PTB
            logger.info("PTB bot stopped.")


if __name__ == "__main__":
    # Используем asyncio.run() для запуска асинхронной main_bot_logic
    try:
        asyncio.run(main_bot_logic())
    except KeyboardInterrupt:
        logger.info("Application shut down by KeyboardInterrupt.")
    except Exception as e:
        logger.critical(f"Unhandled exception in asyncio.run: {e}", exc_info=True)
