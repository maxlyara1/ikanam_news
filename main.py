import torch
import torch.nn as nn
from transformers import ElectraModel, ElectraTokenizer
import joblib
import numpy as np
import logging
from config import (BOT_TOKEN, TARGET_CHANNEL_ID, TELEGRAM_API_ID, TELEGRAM_API_HASH, 
                    RIAN_RU_CHANNEL_USERNAME, SHAP_ENABLED, SHAP_TOP_N_ML_EXPLAIN, 
                    SHAP_MAX_FEATURES_DISPLAY, SHAP_NSAMPLES)
import os
import uuid
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer

# Импорт функции для генерации SHAP из нового файла
# from shap_explainer import generate_shap_plots 

# Вместо прямого импорта функции, которую использует telegram_bot, 
# создадим здесь обертку, чтобы telegram_bot зависел от main.py для этой логики
from shap_explainer import generate_shap_plots as actual_shap_generator

logger = logging.getLogger(__name__)

# Определение класса модели (скопировано из dl-project-haha.ipynb)
class HierarchicalMultiTaskElectra(nn.Module):
    def __init__(self, model_name, num_multi_labels, num_hier_labels,
                 hier_class_weights=None, ml_pos_weight=None):
        super(HierarchicalMultiTaskElectra, self).__init__()
        self.electra = ElectraModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(self.electra.config.hidden_dropout_prob if hasattr(self.electra.config, 'hidden_dropout_prob') else 0.1)

        self.num_multi_labels = num_multi_labels
        self.num_hier_labels = num_hier_labels

        if self.num_multi_labels > 0:
            self.multi_label_classifier = nn.Linear(self.electra.config.hidden_size, num_multi_labels)
            # При загрузке модели pos_weight будет инициализирован из сохраненного состояния, если был при обучении
            self.multi_label_loss_fn = nn.BCEWithLogitsLoss(pos_weight=ml_pos_weight)
        else:
            self.multi_label_classifier = None
            self.multi_label_loss_fn = None

        if self.num_hier_labels > 0:
            self.hierarchical_classifier = nn.Linear(self.electra.config.hidden_size, num_hier_labels)
            # Аналогично для class_weights
            self.hierarchical_loss_fn = nn.CrossEntropyLoss(weight=hier_class_weights, ignore_index=-1)
        else:
            self.hierarchical_classifier = None
            self.hierarchical_loss_fn = None

    def forward(self, input_ids, attention_mask, multi_labels=None, hier_labels=None):
        outputs = self.electra(input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)

        ml_logits = None
        if self.multi_label_classifier is not None:
            ml_logits = self.multi_label_classifier(cls_output)

        hier_logits = None
        if self.hierarchical_classifier is not None:
            hier_logits = self.hierarchical_classifier(cls_output)

        # Расчет потерь не нужен для инференса, но оставим структуру, если понадобится
        current_loss = 0.0
        calculated_any_loss = False
        if self.multi_label_loss_fn is not None and ml_logits is not None and multi_labels is not None:
            if ml_logits.shape == multi_labels.shape:
                ml_loss = self.multi_label_loss_fn(ml_logits, multi_labels)
                current_loss += ml_loss
                calculated_any_loss = True
        if self.hierarchical_loss_fn is not None and hier_logits is not None and hier_labels is not None:
            if hier_logits.shape[0] == hier_labels.shape[0] and hier_labels.nelement() > 0:
                 hier_loss = self.hierarchical_loss_fn(hier_logits, hier_labels)
                 current_loss += hier_loss
                 calculated_any_loss = True
        final_loss = current_loss if calculated_any_loss else None
        return ml_logits, hier_logits, final_loss

def sync_generate_shap_plots_iterator(*args, delete_on_yield: bool = True, **kwargs):
    """Обертка, чтобы вернуть синхронный итератор из shap_explainer."""
    return actual_shap_generator(*args, delete_on_yield=delete_on_yield, **kwargs)

# Глобальные переменные для моделей и связанных объектов
MODEL_NAME = "ai-forever/ruElectra-medium"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 512

model = None
tokenizer = None
mlb = None
idx_to_hier_map = None
hier_map_to_idx = None # Для обратного преобразования, если нужно

def load_model_artifacts():
    global model, tokenizer, mlb, idx_to_hier_map, hier_map_to_idx
    logger.info(f"Using device: {DEVICE}")
    try:
        tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME)
        logger.info(f"Tokenizer for {MODEL_NAME} loaded successfully.")

        mlb_path = 'mlb_trained.joblib'
        hier_map_path = 'idx_to_hier_map_trained.joblib' # Предполагается, что это idx -> label
        model_weights_path = 'fine_tuned_electra.pt'

        mlb = joblib.load(mlb_path)
        idx_to_hier_map = joblib.load(hier_map_path)
        hier_map_to_idx = {v: k for k, v in idx_to_hier_map.items()} # Создаем обратный маппинг

        logger.info(f"MLB loaded from {mlb_path}. Number of multi-labels: {len(mlb.classes_)}")
        logger.info(f"Hierarchical map (idx_to_label) loaded from {hier_map_path}. Number of hier_labels: {len(idx_to_hier_map)}")

        num_multi_labels = len(mlb.classes_)
        num_hier_labels = len(idx_to_hier_map)

        model = HierarchicalMultiTaskElectra(MODEL_NAME, num_multi_labels, num_hier_labels)
        
        # Загрузка весов модели
        # Убедись, что файл fine_tuned_electra.pt содержит state_dict, а не всю модель
        try:
            # Используем weights_only=True для безопасности, как рекомендуется PyTorch
            state_dict = torch.load(model_weights_path, map_location=DEVICE, weights_only=True)
            # Удаляем ключи, связанные с функциями потерь, если они есть
            keys_to_remove = []
            if "multi_label_loss_fn.pos_weight" in state_dict:
                keys_to_remove.append("multi_label_loss_fn.pos_weight")
            if "hierarchical_loss_fn.weight" in state_dict:
                keys_to_remove.append("hierarchical_loss_fn.weight")
            
            if keys_to_remove:
                logger.info(f"Removing unexpected keys from state_dict: {keys_to_remove}")
                for key in keys_to_remove:
                    del state_dict[key]

            model.load_state_dict(state_dict) # Загружаем очищенный state_dict
            logger.info(f"Model weights loaded successfully from {model_weights_path}")
        except RuntimeError as e:
            logger.error(f"RuntimeError loading model weights: {e}")
            logger.warning("Attempting to load with strict=False. This might indicate a mismatch between model definition and saved weights.")
            # Также используем weights_only=True здесь
            state_dict = torch.load(model_weights_path, map_location=DEVICE, weights_only=True)
            # Повторное удаление ключей для попытки с strict=False
            keys_to_remove_strict_false = []
            if "multi_label_loss_fn.pos_weight" in state_dict:
                keys_to_remove_strict_false.append("multi_label_loss_fn.pos_weight")
            if "hierarchical_loss_fn.weight" in state_dict:
                keys_to_remove_strict_false.append("hierarchical_loss_fn.weight")
            
            if keys_to_remove_strict_false:
                logger.info(f"Removing unexpected keys from state_dict (strict=False attempt): {keys_to_remove_strict_false}")
                for key in keys_to_remove_strict_false:
                    del state_dict[key]
            
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Model weights loaded with strict=False from {model_weights_path}")


        model.to(DEVICE)
        model.eval()
        logger.info("Model initialized, weights loaded, and moved to evaluation mode on device.")
        return True

    except FileNotFoundError as e:
        logger.error(f"Error loading model artifacts: File not found - {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during model loading: {e}", exc_info=True)
        return False

def preprocess_news_with_model(news_text: str) -> dict:
    """
    Preprocesses the news text using the loaded fine-tuned model.
    SHAP generation is now handled by the caller (telegram_bot.py).
    """
    if not all([model, tokenizer, mlb, idx_to_hier_map]):
        logger.error("Model, tokenizer, or label binarizers are not loaded. Cannot preprocess.")
        return {
            "text": news_text,
            "multi_labels": ["Error: Model not loaded"],
            "hier_label": ["Error: Model not loaded"],
            # "shap_plots": [] # Больше не возвращаем ключ shap_plots отсюда
        }

    try:
        encoding = tokenizer.encode_plus(
            news_text,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)

        with torch.no_grad():
            ml_logits, hier_logits, _ = model(input_ids, attention_mask)

        multi_labels_predicted = []
        if ml_logits is not None:
            ml_probs = torch.sigmoid(ml_logits)
            binary_preds_ml = (ml_probs > 0.5).cpu().numpy().astype(int)
            multi_labels_predicted = list(mlb.inverse_transform(binary_preds_ml)[0])
            if not multi_labels_predicted:
                 multi_labels_predicted = ["Нет уверенных мульти-меток"]

        raw_hier_label_value = "Нет уверенной иерархической метки"
        if hier_logits is not None and idx_to_hier_map is not None:
            hier_probs = torch.softmax(hier_logits, dim=1)
            pred_hier_idx = torch.argmax(hier_probs, dim=1).item()
            raw_hier_label_value = idx_to_hier_map.get(pred_hier_idx, f"Неизвестный hier_idx: {pred_hier_idx}")
        
        final_hier_str = ""
        if isinstance(raw_hier_label_value, str):
            try:
                import ast
                potential_list = ast.literal_eval(raw_hier_label_value)
                if isinstance(potential_list, list):
                    final_hier_str = " / ".join(str(item) for item in potential_list)
                else:
                    final_hier_str = raw_hier_label_value 
            except (ValueError, SyntaxError):
                final_hier_str = raw_hier_label_value
        elif isinstance(raw_hier_label_value, list):
            final_hier_str = " / ".join(str(item) for item in raw_hier_label_value)
        else: 
            final_hier_str = str(raw_hier_label_value)

        if not final_hier_str: 
            final_hier_str = "Нет данных"
        
        hier_label_output = [final_hier_str]

        # SHAP plot generation is REMOVED from here
            
        return {
            "text": news_text,
            "multi_labels": multi_labels_predicted,
            "hier_label": hier_label_output
            # "shap_plots" key is no longer part of this return
        }
    except Exception as e:
        logger.error(f"Error during model prediction for text \"{news_text[:50]}...\": {e}", exc_info=True)
        return {
            "text": news_text,
            "multi_labels": [f"Error: Prediction failed - {type(e).__name__}"],
            "hier_label": [f"Error: Prediction failed - {type(e).__name__}"]
        }

if __name__ == '__main__':
    # Это для запуска бота, будет перенесено или вызвано из telegram_bot.py или отдельного run.py
    # Пока просто инициализируем логгирование и загружаем модели для теста
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
    )
    logger.info("Starting main.py for testing model loading and preprocessing.")
    
    if load_model_artifacts():
        logger.info("Models and artifacts loaded successfully.")
        # Пример использования
        test_news = "Это тестовая новость о политике и экономике в России. Обсуждаются новые законы."
        processed_news = preprocess_news_with_model(test_news)
        logger.info(f"Test news: \"{test_news}\"")
        logger.info(f"Processed: {processed_news}")

        test_news_2 = "Вчера состоялся важный футбольный матч, который повлиял на расстановку в чемпионате."
        processed_news_2 = preprocess_news_with_model(test_news_2)
        logger.info(f"Test news 2: \"{test_news_2}\"")
        logger.info(f"Processed 2: {processed_news_2}")

        test_news_empty_result = "абырвалг" # Текст, для которого модель может ничего не предсказать
        processed_news_empty = preprocess_news_with_model(test_news_empty_result)
        logger.info(f"Test news (empty result expected): \"{test_news_empty_result}\"")
        logger.info(f"Processed (empty): {processed_news_empty}")

    else:
        logger.error("Failed to load models and artifacts. Bot cannot proceed with ML features.")

    # Здесь должен быть запуск телеграм бота, который мы сделаем в telegram_bot.py
    # from telegram_bot import main as run_telegram_bot
    # run_telegram_bot() 