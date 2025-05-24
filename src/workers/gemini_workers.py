import sys
import os
import warnings
import logging
import logging.handlers
import json
import time
import random
import math
import pandas as pd
import google.generativeai as genai
from google.api_core.exceptions import Aborted, InternalServerError, ResourceExhausted, ServiceUnavailable, DeadlineExceeded

# --- НАСТРОЙКА sys.path (ОСТАВИТЬ) ---
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_from_worker = os.path.abspath(os.path.join(current_file_dir, os.pardir, os.pardir)) 
if project_root_from_worker not in sys.path:
    sys.path.insert(0, project_root_from_worker)

# --- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ И КОНСТАНТЫ (ОСТАВИТЬ) ---
WORKER_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
WORKER_LOG_DATEFMT = '%H:%M:%S'
_LOG_QUEUE = None

PROMPT_INSTRUCTION_TEMPLATE = """
Ты — редактор новостного агрегатора. Твоя задача - внимательно проанализировать и категоризировать список новостей, стремясь к максимально полному отражению тематики каждой новости в рамках заданных ограничений.
Для каждой новости из списка предоставь JSON-объект.
Каждый такой объект должен содержать:

"id": оригинальный идентификатор новости (целое число), который был предоставлен во входных данных.
"multi_labels": Список (list) от 1 до 4 основных тем новости. Если новость освещает несколько различных аспектов, обязательно указывай все применимые основные темы из допустимого списка, не превышая лимит в 4. Старайся выбирать наиболее релевантные и специфичные темы для каждой новости. Допустимые темы: Политика, Экономика, Общество, Происшествия, Спорт, Культура, Технологии, Международные отношения, Региональные новости, Наука, Экология, Здоровье, Образование, Бизнес.
"hier_label": Список (list) из двух элементов: [тема, подтема]. тема - это первая и наиболее важная тема из multi_labels. подтема - это более детальное уточнение для этой основной темы (например, если тема "Политика", подтема может быть "Внешняя политика", "Внутренняя политика", "Выборы", "Законодательство"; если тема "Экономика", подтема может быть "Финансы", "Промышленность", "Рынки"). Подтема должна быть осмысленной и конкретизировать выбранную основную тему.
Входные данные - это JSON-массив объектов, где каждый объект имеет "id" и "text".
Твой ответ ДОЛЖЕН БЫТЬ JSON-массивом, где каждый элемент массива - это JSON-объект с полями "id", "multi_labels", "hier_label" для соответствующей входной новости.
Убедись, что идентификаторы 'id' в твоем ответе точно соответствуют идентификаторам 'id' из входного списка новостей.
Не включай в ответ ничего, кроме этого JSON-массива.
Новости для обработки:
{news_json_payload}
Ответ (только JSON-массив):
"""

GENERATION_CONFIG_WORKER = {
    "temperature": 0.3,
    "max_output_tokens": 65536,
    "response_mime_type": "application/json",
}
SAFETY_SETTINGS_WORKER = [
    {"category": c, "threshold": "BLOCK_NONE"} for c in [
        "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"
    ]
]

# --- ФУНКЦИИ ВОРКЕРА ---
def setup_worker_logging(q):
    global _LOG_QUEUE
    _LOG_QUEUE = q 
    warnings.filterwarnings("ignore", category=FutureWarning, module="numpy.core.fromnumeric", message=".*swapaxes.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()
    queue_h = logging.handlers.QueueHandler(q)
    root_logger.addHandler(queue_h)
    # ОПЦИОНАЛЬНО: прямой лог в файл для отладки воркеров, если QueueHandler не сработает
    # pid = os.getpid()
    # direct_log_file_path = f"worker_direct_log_{pid}.txt" 
    # file_h = logging.FileHandler(direct_log_file_path, mode='a') 
    # formatter = logging.Formatter(WORKER_LOG_FORMAT, datefmt=WORKER_LOG_DATEFMT)
    # file_h.setFormatter(formatter)
    # root_logger.addHandler(file_h)
    root_logger.setLevel(logging.INFO)
    # test_logger = logging.getLogger(f"ВоркерPID-{os.getpid()}") # Убрал worker_id здесь
    # test_logger.info(f"Logging setup complete.")


def _init_worker_model(api_key_for_worker, model_name_for_worker, worker_name_log):
    logger = logging.getLogger(worker_name_log)
    try:
        genai.configure(api_key=api_key_for_worker)
        model = genai.GenerativeModel(
            model_name=model_name_for_worker,
            generation_config=GENERATION_CONFIG_WORKER,
            safety_settings=SAFETY_SETTINGS_WORKER
        )
        base_prompt_for_counting_worker = PROMPT_INSTRUCTION_TEMPLATE.format(news_json_payload="[]")
        tokens_base = model.count_tokens(base_prompt_for_counting_worker).total_tokens
        # УМЕНЬШЕНО ЛОГИРОВАНИЕ: logger.info(f"Модель '{model_name_for_worker}' инициализирована. Базовых токенов: {tokens_base}")
        return model, tokens_base
    except Exception as e:
        logger.error(f"ОШИБКА инициализации модели: {e}")
        return None, 350


def _generate_categories_for_single_api_batch_with_retries(gemini_model_worker, news_batch_items_worker, worker_name_log):
    logger = logging.getLogger(worker_name_log)
    if not news_batch_items_worker:
        logger.warning("Попытка обработать пустой API батч.")
        return []

    news_input_for_prompt = [{"id": item["id"], "text": item["text"]} for item in news_batch_items_worker]
    news_input_json_string = json.dumps(news_input_for_prompt, ensure_ascii=False)
    full_prompt = PROMPT_INSTRUCTION_TEMPLATE.format(news_json_payload=news_input_json_string)
    empty_results_for_batch = [{"id": item["id"], "multi_labels": [], "hier_label": []} for item in news_batch_items_worker]
    max_retries = 3 # Уменьшим количество ретраев для скорости, можно вернуть 5
    base_delay = 2 
    item_ids_str = f"IDs {news_batch_items_worker[0]['id']}..{news_batch_items_worker[-1]['id']}" if news_batch_items_worker else "пустой батч"

    for attempt in range(max_retries): # range(max_retries) даст max_retries попыток (0 до max_retries-1)
        # УМЕНЬШЕНО ЛОГИРОВАНИЕ: logger.info(f"API запрос (попытка {attempt + 1}/{max_retries}) для {len(news_batch_items_worker)} новостей ({item_ids_str})")
        try:
            response = gemini_model_worker.generate_content(full_prompt)
            if not response.parts:
                logger.warning(f"API: Пустой ответ (response.parts) для {item_ids_str}.")
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                     logger.warning(f"Причина: {response.prompt_feedback}")
                return empty_results_for_batch
            response_text = response.text
            if not response_text.strip():
                logger.warning(f"API: Пустой response_text для {item_ids_str}.")
                if attempt < max_retries - 1:
                    delay = (base_delay ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                    continue
                return empty_results_for_batch
            try:
                parsed_json_list = json.loads(response_text)
                if not isinstance(parsed_json_list, list):
                    logger.error(f"JSON: Ответ не JSON-массив для {item_ids_str}. Ответ: {response_text[:200]}")
                    return empty_results_for_batch
                results_map_from_api = {item.get("id"): item for item in parsed_json_list if isinstance(item, dict)}
                batch_final_results = []
                for requested_item in news_batch_items_worker:
                    req_id = requested_item["id"]
                    api_item = results_map_from_api.get(req_id)
                    if api_item and isinstance(api_item.get("multi_labels"), list) and isinstance(api_item.get("hier_label"), list):
                        batch_final_results.append(api_item)
                    else:
                        batch_final_results.append({"id": req_id, "multi_labels": [], "hier_label": []})
                # УМЕНЬШЕНО ЛОГИРОВАНИЕ: logger.info(f"Успешно обработан API батч ({item_ids_str}).")
                return batch_final_results
            except json.JSONDecodeError as e_json:
                logger.error(f"JSON: Ошибка декодирования: {e_json} для {item_ids_str}.")
                if attempt < max_retries - 1:
                    delay = (base_delay ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                    continue
                return empty_results_for_batch
        except (InternalServerError, ServiceUnavailable, ResourceExhausted, DeadlineExceeded, Aborted) as e_api:
            logger.warning(f"API ОШИБКА (попытка {attempt + 1}/{max_retries}): {type(e_api).__name__} для {item_ids_str}.")
            if attempt < max_retries - 1:
                delay = (base_delay ** attempt) + random.uniform(0.5, 1.5)
                time.sleep(delay)
            else:
                logger.error(f"Все попытки исчерпаны для API батча ({item_ids_str}) после {type(e_api).__name__}.")
                return empty_results_for_batch
        except Exception as e_critical:
            logger.critical(f"КРИТИЧЕСКАЯ_ОШИБКА_API: {type(e_critical).__name__} - {e_critical} для {item_ids_str}.")
            return empty_results_for_batch
    logger.warning(f"Цикл retry завершился без return для API батча ({item_ids_str}).")
    return empty_results_for_batch


def _process_chunk_with_token_batches_for_worker(args_tuple):
    chunk_df, api_key, worker_id, model_name, target_tokens_total_cfg, daily_limit_cfg, delay_cfg = args_tuple
    worker_name_log = f"Воркер-{worker_id:02d}" 
    logger = logging.getLogger(worker_name_log)
    chunk_ids_str = f"ID {chunk_df.index[0]}..{chunk_df.index[-1]}" if not chunk_df.empty else "пустой чанк"
    logger.info(f"Начинает обработку чанка из {len(chunk_df)} строк ({chunk_ids_str}). Ключ: ..{api_key[-4:]}")
    chunk_df.name = f"chunk_{worker_id}"
    gemini_model_w, tokens_base_w = _init_worker_model(api_key, model_name, worker_name_log)
    if gemini_model_w is None:
        logger.error(f"НЕ СМОГ ИНИЦИАЛИЗИРОВАТЬ МОДЕЛЬ. Пропускает чанк {chunk_ids_str}.")
        chunk_df_results_error = chunk_df.copy()
        chunk_df_results_error["multi_labels"] = pd.Series([[] for _ in range(len(chunk_df_results_error))], index=chunk_df_results_error.index, dtype=object)
        chunk_df_results_error["hier_label"] = pd.Series([[] for _ in range(len(chunk_df_results_error))], index=chunk_df_results_error.index, dtype=object)
        return chunk_df_results_error

    all_processed_results_flat_w = []
    current_batch_items_w = []
    current_batch_estimated_tokens_w = tokens_base_w
    requests_count_w = 0
    chunk_df_results = chunk_df.copy()
    chunk_df_results["multi_labels"] = pd.Series([[] for _ in range(len(chunk_df_results))], index=chunk_df_results.index, dtype=object)
    chunk_df_results["hier_label"] = pd.Series([[] for _ in range(len(chunk_df_results))], index=chunk_df_results.index, dtype=object)
    JSON_OVERHEAD_PER_ITEM_APPROX_W = 20 
    CHARS_PER_TOKEN_ESTIMATE = 3.5
    # УМЕНЬШЕНО ЛОГИРОВАНИЕ: logger.info(f"Подсчет токенов для чанка ({chunk_ids_str})...")
    news_with_tokens_info_w = []
    for idx_w, row_w in chunk_df.iterrows():
        news_text_w = str(row_w.get("text", ""))
        tokens_for_text_approx = math.ceil(len(news_text_w) / CHARS_PER_TOKEN_ESTIMATE)
        tokens_for_item_w = tokens_for_text_approx + JSON_OVERHEAD_PER_ITEM_APPROX_W
        news_with_tokens_info_w.append({"id": idx_w, "text": news_text_w, "tokens": tokens_for_item_w})
    
    api_batches_sent = 0
    for news_item_w in news_with_tokens_info_w:
        if requests_count_w >= daily_limit_cfg:
            logger.warning(f"Достигнут лимит запросов ({requests_count_w}/{daily_limit_cfg}). Остановка чанка {chunk_ids_str}.")
            break
        item_tokens_w = news_item_w["tokens"]
        if current_batch_items_w and (current_batch_estimated_tokens_w + item_tokens_w > target_tokens_total_cfg):
            # УМЕНЬШЕНО ЛОГИРОВАНИЕ: logger.info(f"Отправка API батча ({len(current_batch_items_w)} новостей, ~{current_batch_estimated_tokens_w:,} токенов). Запрос #{requests_count_w + 1}/{daily_limit_cfg}")
            batch_results_list_w = _generate_categories_for_single_api_batch_with_retries(gemini_model_w, current_batch_items_w, worker_name_log)
            api_batches_sent += 1
            requests_count_w += 1
            if isinstance(batch_results_list_w, list): all_processed_results_flat_w.extend(batch_results_list_w)
            current_batch_items_w = []
            current_batch_estimated_tokens_w = tokens_base_w
            if requests_count_w < daily_limit_cfg and delay_cfg > 0: time.sleep(delay_cfg)
        if tokens_base_w + item_tokens_w > target_tokens_total_cfg:
            logger.warning(f"Новость ID {news_item_w['id']} слишком большая (~{tokens_base_w + item_tokens_w:,} т.), пропускается.")
            all_processed_results_flat_w.append({"id": news_item_w['id'], "multi_labels": ["ITEM_TOO_LARGE"], "hier_label": []})
            continue
        current_batch_items_w.append(news_item_w)
        current_batch_estimated_tokens_w += item_tokens_w
    if current_batch_items_w and requests_count_w < daily_limit_cfg:
        # УМЕНЬШЕНО ЛОГИРОВАНИЕ: logger.info(f"Отправка последнего API батча ({len(current_batch_items_w)} новостей, ~{current_batch_estimated_tokens_w:,} токенов). Запрос #{requests_count_w + 1}/{daily_limit_cfg}")
        batch_results_list_w = _generate_categories_for_single_api_batch_with_retries(gemini_model_w, current_batch_items_w, worker_name_log)
        api_batches_sent += 1
        requests_count_w += 1
        if isinstance(batch_results_list_w, list): all_processed_results_flat_w.extend(batch_results_list_w)
    if all_processed_results_flat_w:
        temp_results_df = pd.DataFrame(all_processed_results_flat_w)
        if not temp_results_df.empty and 'id' in temp_results_df.columns:
            original_id_type = chunk_df_results.index.dtype
            try:
                if pd.api.types.is_numeric_dtype(original_id_type):
                    temp_results_df['id'] = pd.to_numeric(temp_results_df['id'], errors='coerce')
                    temp_results_df.dropna(subset=['id'], inplace=True)
                    if not temp_results_df.empty: temp_results_df['id'] = temp_results_df['id'].astype(original_id_type)
                elif pd.api.types.is_string_dtype(original_id_type) or original_id_type == 'object':
                    temp_results_df['id'] = temp_results_df['id'].astype(str)
            except Exception as e_type_conv:
                 logger.warning(f"Ошибка приведения типов ID из API: {e_type_conv}")
            if not temp_results_df.empty and temp_results_df['id'].notna().all():
                try:
                    temp_results_df.set_index('id', inplace=True)
                    cols_to_update_final = [col for col in ["multi_labels", "hier_label"] if col in temp_results_df.columns]
                    chunk_df_results.update(temp_results_df[cols_to_update_final], overwrite=True, errors='ignore')
                except Exception as e_merge:
                     logger.warning(f"Ошибка при обновлении результатов чанка: {e_merge}.")
            elif not temp_results_df.empty:
                 logger.warning(f"Не все ID в результатах API корректны или temp_results_df пуст.")
    processed_news_count = chunk_df_results['multi_labels'].apply(lambda x: isinstance(x, list) and len(x) > 0 and x != ["ITEM_TOO_LARGE"]).sum()
    logger.info(f"Завершил обработку чанка ({chunk_ids_str}). Отправлено API-батчей: {api_batches_sent}. Новостей с метками: {processed_news_count}/{len(chunk_df)}.")
    return chunk_df_results