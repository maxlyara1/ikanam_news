# gemini_workers.py
import pandas as pd
import json
import time
import random
import math
import google.generativeai as genai
from google.api_core.exceptions import Aborted, InternalServerError, ResourceExhausted, ServiceUnavailable, DeadlineExceeded

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

# Глобальные переменные для _init_worker_model, чтобы избежать передачи через args если они всегда одинаковые
# Но лучше передавать, если они могут меняться между воркерами или запусками
# Для простоты, предположим, что generation_config и safety_settings одинаковы для всех воркеров
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

def _init_worker_model(api_key_for_worker, model_name_for_worker):
    """Инициализирует API и модель для одного воркера."""
    try:
        genai.configure(api_key=api_key_for_worker) # Конфигурируем API для текущего процесса/ключа
        model = genai.GenerativeModel(
            model_name=model_name_for_worker,
            generation_config=GENERATION_CONFIG_WORKER, # Используем глобальные настройки
            safety_settings=SAFETY_SETTINGS_WORKER
        )
        base_prompt_for_counting_worker = PROMPT_INSTRUCTION_TEMPLATE.format(news_json_payload="[]")
        tokens_base = model.count_tokens(base_prompt_for_counting_worker).total_tokens
        return model, tokens_base
    except Exception as e:
        print(f"ОШИБКА ВОРКЕРА при инициализации модели с ключом (последние 4 символа): ...{api_key_for_worker[-4:]}: {e}")
        return None, 350 # Fallback

def _generate_categories_for_single_api_batch_with_retries(gemini_model_worker, news_batch_items_worker, worker_id_log_prefix=""):
    news_input_for_prompt = [{"id": item["id"], "text": item["text"]} for item in news_batch_items_worker]
    news_input_json_string = json.dumps(news_input_for_prompt, ensure_ascii=False, indent=2)
    full_prompt = PROMPT_INSTRUCTION_TEMPLATE.format(news_json_payload=news_input_json_string)
    empty_results_with_ids = [{"id": item["id"], "multi_labels": [], "hier_label": []} for item in news_batch_items_worker]

    max_retries = 5 # Максимальное количество повторных попыток
    base_delay = 2  # Начальная задержка в секундах

    for attempt in range(max_retries + 1):
        try:
            print(f"  {worker_id_log_prefix} API запрос (попытка {attempt + 1}/{max_retries + 1}) для IDs {news_batch_items_worker[0]['id']}..{news_batch_items_worker[-1]['id']}")
            response = gemini_model_worker.generate_content(full_prompt)
            
            if not response.parts:
                # Это может быть блокировка по безопасности, здесь retries не помогут
                print(f"  {worker_id_log_prefix} ПРЕДУПРЕЖДЕНИЕ_API: Пустой ответ (response.parts). IDs: {news_batch_items_worker[0]['id']}..{news_batch_items_worker[-1]['id']}")
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                     print(f"    Причина: {response.prompt_feedback}")
                return empty_results_with_ids # Не повторяем, если это похоже на блокировку
            
            response_text = response.text
            if not response_text.strip():
                print(f"  {worker_id_log_prefix} ПРЕДУПРЕЖДЕНИЕ_API: Пустой response_text. IDs: {news_batch_items_worker[0]['id']}..{news_batch_items_worker[-1]['id']}")
                # Пустой ответ может быть временной проблемой, попробуем еще раз, если не последняя попытка
                if attempt < max_retries:
                    delay = (base_delay ** attempt) + random.uniform(0, 1)
                    print(f"    Повторная попытка через {delay:.2f} сек...")
                    time.sleep(delay)
                    continue
                return empty_results_with_ids

            # Если дошли сюда, значит response_text не пустой, пытаемся парсить
            try:
                parsed_json_list = json.loads(response_text)
                if not isinstance(parsed_json_list, list):
                    print(f"  {worker_id_log_prefix} ОШИБКА_JSON: Ответ не JSON-массив. IDs: {news_batch_items_worker[0]['id']}..{news_batch_items_worker[-1]['id']}")
                    return empty_results_with_ids # Не повторяем, если структура ответа неверна
                
                results_map_from_api = {item.get("id"): item for item in parsed_json_list if isinstance(item, dict)}
                batch_final_results = []
                for requested_item in news_batch_items_worker:
                    req_id = requested_item["id"]
                    api_item = results_map_from_api.get(req_id)
                    if api_item and isinstance(api_item.get("multi_labels"), list) and isinstance(api_item.get("hier_label"), list):
                        batch_final_results.append(api_item)
                    else:
                        batch_final_results.append({"id": req_id, "multi_labels": [], "hier_label": []})
                return batch_final_results # Успех!

            except json.JSONDecodeError as e: # JSON оборван или невалиден
                print(f"  {worker_id_log_prefix} ОШИБКА_JSON: Ошибка декодирования: {e}. IDs: {news_batch_items_worker[0]['id']}..{news_batch_items_worker[-1]['id']}. Ответ(начало): {response_text[:100]}")
                # Оборванный JSON - это проблема сервера или сети, попробуем еще раз
                if attempt < max_retries:
                    delay = (base_delay ** attempt) + random.uniform(0, 1)
                    print(f"    Повторная попытка через {delay:.2f} сек...")
                    time.sleep(delay)
                    continue
                return empty_results_with_ids # Все попытки исчерпаны

        except (InternalServerError, ServiceUnavailable, ResourceExhausted, DeadlineExceeded, Aborted) as e_api:
            # Это ошибки, для которых имеет смысл делать retry
            print(f"  {worker_id_log_prefix} ОШИБКА_API (попытка {attempt + 1}): {type(e_api).__name__} - {e_api}. IDs: {news_batch_items_worker[0]['id']}..{news_batch_items_worker[-1]['id']}")
            if attempt < max_retries:
                delay = (base_delay ** attempt) + random.uniform(0.5, 1.5) # Экспоненциальная задержка + jitter
                print(f"    Повторная попытка через {delay:.2f} сек...")
                time.sleep(delay)
            else:
                print(f"  {worker_id_log_prefix} Все попытки исчерпаны для этого батча.")
                return empty_results_with_ids
        except Exception as e_critical: # Другие, возможно, невосстановимые ошибки
            print(f"  {worker_id_log_prefix} КРИТИЧЕСКАЯ_ОШИБКА_API (не retry): {type(e_critical).__name__} - {e_critical}. IDs: {news_batch_items_worker[0]['id']}..{news_batch_items_worker[-1]['id']}")
            return empty_results_with_ids # Не повторяем для неизвестных критических ошибок
    
    return empty_results_with_ids # Если цикл завершился без return (не должно произойти)


def _process_chunk_with_token_batches_for_worker(chunk_df, api_key, worker_id, model_name, target_tokens_total_cfg, daily_limit_cfg, delay_cfg):
    worker_log_prefix = f"[Воркер {worker_id}, Ключ ..{api_key[-4:]}]"
    print(f"{worker_log_prefix} Начинает обработку чанка из {len(chunk_df)} строк.")
    
    gemini_model_w, tokens_base_w = _init_worker_model(api_key, model_name)
    if gemini_model_w is None:
        print(f"{worker_log_prefix} НЕ СМОГ ИНИЦИАЛИЗИРОВАТЬ МОДЕЛЬ. Пропускает чанк.")
        # Возвращаем DataFrame с пустыми метками, но с правильными индексами из чанка
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

    news_with_tokens_info_w = []
    JSON_OVERHEAD_PER_ITEM_APPROX_W = 20 
    CHARS_PER_TOKEN_ESTIMATE = 3 
    
    print(f"  {worker_log_prefix} Примерный подсчет токенов для чанка...") # Можно сделать менее многословным
    start_time_counting_w = time.time()
    for i_w, (idx_w, row_w) in enumerate(chunk_df.iterrows()):
        news_text_w = str(row_w["text"])
        tokens_for_text_approx = math.ceil(len(news_text_w) / CHARS_PER_TOKEN_ESTIMATE)
        tokens_for_item_w = tokens_for_text_approx + JSON_OVERHEAD_PER_ITEM_APPROX_W
        news_with_tokens_info_w.append({"id": idx_w, "text": news_text_w, "tokens": tokens_for_item_w})
    print(f"  {worker_log_prefix} Примерный подсчет токенов завершен. Заняло: {time.time() - start_time_counting_w:.2f}s")

    for news_item_w in news_with_tokens_info_w:
        if requests_count_w >= daily_limit_cfg:
            print(f"  {worker_log_prefix} Достигнут лимит запросов ({requests_count_w}). Остановка воркера.")
            break
        item_tokens_w = news_item_w["tokens"]
        if current_batch_items_w and (current_batch_estimated_tokens_w + item_tokens_w > target_tokens_total_cfg):
            print(f"  {worker_log_prefix} Отправка батча ({len(current_batch_items_w)} новостей, ~{current_batch_estimated_tokens_w} токенов). Запрос #{requests_count_w + 1}")
            batch_results_list_w = _generate_categories_for_single_api_batch_with_retries(gemini_model_w, current_batch_items_w, worker_log_prefix)
            requests_count_w += 1
            if isinstance(batch_results_list_w, list): all_processed_results_flat_w.extend(batch_results_list_w)
            current_batch_items_w = []
            current_batch_estimated_tokens_w = tokens_base_w
            if requests_count_w < daily_limit_cfg and delay_cfg > 0: time.sleep(delay_cfg) # Задержка между успешными батчами

        if tokens_base_w + item_tokens_w > target_tokens_total_cfg:
            all_processed_results_flat_w.append({"id": news_item_w["id"], "multi_labels": [], "hier_label": []})
            continue
        current_batch_items_w.append(news_item_w)
        current_batch_estimated_tokens_w += item_tokens_w

    if current_batch_items_w and requests_count_w < daily_limit_cfg:
        print(f"  {worker_log_prefix} Отправка последнего батча ({len(current_batch_items_w)} новостей, ~{current_batch_estimated_tokens_w} токенов). Запрос #{requests_count_w + 1}")
        batch_results_list_w = _generate_categories_for_single_api_batch_with_retries(gemini_model_w, current_batch_items_w, worker_log_prefix)
        requests_count_w += 1
        if isinstance(batch_results_list_w, list): all_processed_results_flat_w.extend(batch_results_list_w)
    
    if all_processed_results_flat_w:
        results_df_w = pd.DataFrame(all_processed_results_flat_w)
        if not results_df_w.empty and 'id' in results_df_w.columns:
            original_id_type_w = chunk_df_results.index.dtype
            try:
                if pd.api.types.is_numeric_dtype(original_id_type_w):
                    results_df_w['id'] = pd.to_numeric(results_df_w['id'], errors='coerce')
                    results_df_w.dropna(subset=['id'], inplace=True)
                    if not results_df_w.empty: results_df_w['id'] = results_df_w['id'].astype(original_id_type_w)
                elif original_id_type_w == 'object': results_df_w['id'] = results_df_w['id'].astype(str)
            except Exception: pass # Ошибки приведения типов здесь менее критичны, если ID все равно совпадут

            if not results_df_w.empty and 'id' in results_df_w.columns and results_df_w['id'].notna().all():
                try:
                    results_df_w = results_df_w.set_index('id')
                    updatable_idx_w = chunk_df_results.index.intersection(results_df_w.index)
                    if not updatable_idx_w.empty:
                        for idx_to_update_w in updatable_idx_w:
                            res_row_w = results_df_w.loc[idx_to_update_w]
                            chunk_df_results.at[idx_to_update_w, "multi_labels"] = res_row_w.get("multi_labels", [])
                            chunk_df_results.at[idx_to_update_w, "hier_label"] = res_row_w.get("hier_label", [])
                except Exception: pass # Ошибки мержа (не должно быть, если ID корректны)
    
    print(f"{worker_log_prefix} Завершил обработку чанка. Обработано API-запросов: {requests_count_w}. Найдено/обработано записей в API ответах: {len(all_processed_results_flat_w)}/{len(chunk_df)}")
    return chunk_df_results