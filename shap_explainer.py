import matplotlib
matplotlib.use('Agg') # Добавляем эту строку для установки бэкенда
import torch
import torch.nn as nn # Нужен для аннотации типа HierarchicalMultiTaskElectra
from transformers import ElectraTokenizer
from sklearn.preprocessing import MultiLabelBinarizer # Нужен для аннотации типа
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import uuid
import logging
from typing import TYPE_CHECKING, Iterator, Tuple, Optional

# Предполагается, что класс HierarchicalMultiTaskElectra будет доступен 
# либо через импорт из main (если main.py не будет импортировать этот файл циклически)
# либо его определение нужно будет скопировать/перенести сюда или в отдельный model.py
# Для простоты пока будем ожидать, что main.py передаст инстанс модели.
# Чтобы избежать циклического импорта, если main импортирует shap_explainer,
# main не должен определять HierarchicalMultiTaskElectra здесь, а должен импортировать его, если класс вынесен.
# Пока оставляем так, но имеем в виду. Если main.py импортирует ЭТОТ файл, то HierarchicalMultiTaskElectra
# должно быть передано как объект, а не импортировано здесь из main.
# Для type hinting:
if TYPE_CHECKING:
    from main import HierarchicalMultiTaskElectra

logger = logging.getLogger(__name__)

def generate_shap_plots(
    trained_model: 'HierarchicalMultiTaskElectra',
    tokenizer_for_shap: ElectraTokenizer,
    sentence_to_explain: str,
    mlb_loaded: MultiLabelBinarizer,
    idx_to_hier_map_loaded: dict,
    device_for_shap: torch.device,
    max_len_for_shap: int = 512,
    top_n_ml_explain: int = 1,
    max_features_to_display: int = 10, # Увеличено для лучшей демонстрации, можно настроить
    nsamples_shap: int = 64,
    delete_on_yield: bool = True  # Новый параметр
) -> Iterator[Tuple[str, str]]:
    """
    Генерирует SHAP waterfall plots для иерархической и мульти-лейбл классификации.
    Является генератором, возвращающим пути к файлам графиков и их заголовки.
    """
    try:
        trained_model.eval()

        encoding = tokenizer_for_shap.encode_plus(
            sentence_to_explain, add_special_tokens=True, max_length=max_len_for_shap,
            padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(device_for_shap)
        attention_mask = encoding['attention_mask'].to(device_for_shap)

        with torch.no_grad():
            ml_logits, hier_logits, _ = trained_model(input_ids, attention_mask)

        # --- Иерархическое предсказание и SHAP plot ---
        if trained_model.num_hier_labels > 0 and hier_logits is not None and idx_to_hier_map_loaded:
            hier_probs = torch.softmax(hier_logits, dim=1)
            pred_hier_idx = torch.argmax(hier_probs, dim=1).item()
            raw_pred_hier_class_value = idx_to_hier_map_loaded.get(pred_hier_idx, f"UnknownHierIdx_{pred_hier_idx}")
            
            display_hier_class_name = raw_pred_hier_class_value
            if isinstance(raw_pred_hier_class_value, str):
                try:
                    import ast
                    potential_list = ast.literal_eval(raw_pred_hier_class_value)
                    if isinstance(potential_list, list):
                        display_hier_class_name = " / ".join(str(item).strip() for item in potential_list)
                except (ValueError, SyntaxError): pass
            elif isinstance(raw_pred_hier_class_value, list):
                 display_hier_class_name = " / ".join(str(item).strip() for item in raw_pred_hier_class_value)

            logger.info(f"SHAP: Explaining hier class: '{display_hier_class_name}' (raw: '{raw_pred_hier_class_value}')")

            def f_hier(texts_list_from_shap):
                if trained_model.num_hier_labels == 0: return np.zeros((len(texts_list_from_shap), len(idx_to_hier_map_loaded) if idx_to_hier_map_loaded else 1))
                all_probs = []
                for text_sample in texts_list_from_shap:
                    enc = tokenizer_for_shap.encode_plus(text_sample, max_length=max_len_for_shap, padding='max_length', truncation=True, return_tensors='pt')
                    with torch.no_grad():
                        _, h_logits, _ = trained_model(enc['input_ids'].to(device_for_shap), enc['attention_mask'].to(device_for_shap))
                    probs = torch.softmax(h_logits, dim=1).cpu().numpy() if h_logits is not None else np.zeros((1, trained_model.num_hier_labels))
                    all_probs.append(probs)
                return np.concatenate(all_probs, axis=0)

            masker_h = shap.maskers.Text(r"\b\w+\b")
            logger.info(f"SHAP HIER: Text masker_h created with regex: r'\b\w+\b'")
            
            explainer_h: Optional[shap.Explainer] = None
            try:
                explainer_h = shap.Explainer(f_hier, masker_h, output_names=list(idx_to_hier_map_loaded.values()))
                logger.info(f"SHAP HIER: Explainer_h created: {explainer_h}")
            except Exception as e_explainer:
                logger.error(f"SHAP HIER: Error creating Explainer_h for hier_labels: {e_explainer}", exc_info=True)

            if explainer_h:
                shap_values_h_explanation: Optional[shap.Explanation] = None
                try:
                    logger.info(f"SHAP HIER: Getting SHAP values for '{display_hier_class_name}' with nsamples_shap={nsamples_shap}")
                    shap_values_h_explanation = explainer_h([sentence_to_explain], max_evals=nsamples_shap)
                except TypeError as e_type_shap_vals:
                    if 'max_evals' in str(e_type_shap_vals).lower():
                        logger.warning(f"SHAP HIER: max_evals not accepted by explainer_h. Falling back. Error: {e_type_shap_vals}")
                        try:
                            shap_values_h_explanation = explainer_h([sentence_to_explain])
                        except Exception as e_fallback:
                            logger.error(f"SHAP HIER: Error during fallback SHAP value calculation: {e_fallback}", exc_info=True)
                    else: raise # Перевыбрасываем, если TypeError не связан с max_evals
                except Exception as e_shap_values_other:
                    logger.error(f"SHAP HIER: Error obtaining SHAP values for '{display_hier_class_name}': {e_shap_values_other}", exc_info=True)

                if shap_values_h_explanation:
                    try:
                        if not (hasattr(shap_values_h_explanation, 'shape') and len(shap_values_h_explanation.shape) >= 3):
                            logger.warning(f"SHAP HIER: Invalid SHAP values shape for '{display_hier_class_name}'. Skipping plot.")
                        else:
                            exp_slice_h = shap_values_h_explanation[0, :, pred_hier_idx]
                            plot_arg_h = exp_slice_h
                            
                            if (hasattr(exp_slice_h, 'data') and exp_slice_h.data is not None and
                                hasattr(exp_slice_h, 'values') and exp_slice_h.values is not None and
                                hasattr(exp_slice_h.data, '__len__') and len(exp_slice_h.data) == len(exp_slice_h.values) and
                                all(isinstance(t, str) for t in exp_slice_h.data)):

                                original_tokens = np.array(exp_slice_h.data)
                                indices_to_keep = [i for i, token in enumerate(original_tokens) if len(token.strip()) >= 4]
                                filtered_tokens = original_tokens[indices_to_keep]
                                filtered_values = exp_slice_h.values[indices_to_keep]

                                if len(filtered_tokens) > 0:
                                    plot_arg_h = shap.Explanation(
                                        values=filtered_values,
                                        base_values=exp_slice_h.base_values,
                                        data=filtered_tokens, 
                                        feature_names=list(filtered_tokens)
                                    )
                                    logger.info(f"SHAP HIER: Filtered tokens for '{display_hier_class_name}'. Kept {len(filtered_tokens)}/{len(original_tokens)}.")
                                else:
                                    logger.info(f"SHAP HIER: All tokens filtered out for '{display_hier_class_name}'. Plot will be skipped.")
                                
                            if hasattr(plot_arg_h, 'data') and len(plot_arg_h.data) > 0:
                                plt.figure()
                                shap.waterfall_plot(plot_arg_h, max_display=max_features_to_display, show=False)
                                title = f"SHAP Иерархия: {display_hier_class_name}"
                                plt.title(title, fontsize=10)
                                filename = f"shap_hier_{uuid.uuid4().hex[:8]}.png"
                                plt.savefig(filename, bbox_inches='tight')
                                plt.close()
                                logger.info(f"SHAP HIER: Saved plot {filename} for '{display_hier_class_name}'")
                                yield filename, title
                                if delete_on_yield: # Проверяем флаг перед удалением
                                    if os.path.exists(filename): # Доп. проверка перед удалением
                                        try: os.remove(filename); logger.info(f"SHAP HIER: Removed plot {filename} (delete_on_yield=True).")
                                        except Exception as e_remove_hier: logger.error(f"SHAP HIER: Error removing {filename}: {e_remove_hier}")
                            else:
                                logger.info(f"SHAP HIER: No data to plot for '{display_hier_class_name}' after filtering. Plot skipped.")
                                
                    except Exception as e_plot_gen_h:
                        logger.error(f"SHAP HIER: Error generating plot for '{display_hier_class_name}': {e_plot_gen_h}", exc_info=True)
                elif explainer_h:
                     logger.warning(f"SHAP HIER: SHAP values are None for '{display_hier_class_name}'. Plot cannot be generated.")
            else:
                logger.info("SHAP HIER: Explainer_h was not created, skipping SHAP for hier_labels.")

        # --- Мульти-лейбл предсказания и SHAP plots ---
        if trained_model.num_multi_labels > 0 and ml_logits is not None and mlb_loaded and hasattr(mlb_loaded, 'classes_'):
            ml_probs_for_shap = torch.sigmoid(ml_logits)
            binary_preds_ml = (ml_probs_for_shap[0].cpu().numpy() > 0.5).astype(int)
            pred_ml_indices = [i for i, val in enumerate(binary_preds_ml) if val == 1]

            if not pred_ml_indices:
                logger.info("SHAP ML: No multi-label classes predicted with >0.5 probability. Skipping ML SHAP.")
            else:
                logger.info(f"SHAP ML: Explaining multi-label classes for indices: {pred_ml_indices}")

                def f_ml(texts_list_from_shap):
                    if trained_model.num_multi_labels == 0: return np.zeros((len(texts_list_from_shap), len(mlb_loaded.classes_) if mlb_loaded.classes_ else 1))
                    all_probs = []
                    for text_sample in texts_list_from_shap:
                        enc = tokenizer_for_shap.encode_plus(text_sample, max_length=max_len_for_shap, padding='max_length', truncation=True, return_tensors='pt')
                        with torch.no_grad():
                            m_logits, _, _ = trained_model(enc['input_ids'].to(device_for_shap), enc['attention_mask'].to(device_for_shap))
                        probs = torch.sigmoid(m_logits).cpu().numpy() if m_logits is not None else np.zeros((1, trained_model.num_multi_labels))
                        all_probs.append(probs)
                    return np.concatenate(all_probs, axis=0)

                masker_ml = shap.maskers.Text(r"\b\w+\b")
                logger.info(f"SHAP ML: Text masker_ml created with regex: r'\b\w+\b'")

                explainer_m: Optional[shap.Explainer] = None
                try:
                    explainer_m = shap.Explainer(f_ml, masker_ml, output_names=list(mlb_loaded.classes_))
                    logger.info(f"SHAP ML: Explainer_m created: {explainer_m}")
                except Exception as e_explainer_ml:
                    logger.error(f"SHAP ML: Error creating Explainer_m: {e_explainer_ml}", exc_info=True)

                if explainer_m:
                    shap_values_m_explanation: Optional[shap.Explanation] = None
                    try:
                        logger.info(f"SHAP ML: Getting SHAP values for multi-label with nsamples_shap={nsamples_shap}")
                        shap_values_m_explanation = explainer_m([sentence_to_explain], max_evals=nsamples_shap)
                    except TypeError as e_type_shap_vals_ml:
                        if 'max_evals' in str(e_type_shap_vals_ml).lower():
                            logger.warning(f"SHAP ML: max_evals not accepted by explainer_m. Falling back. Error: {e_type_shap_vals_ml}")
                            try:
                                shap_values_m_explanation = explainer_m([sentence_to_explain])
                            except Exception as e_fallback_ml:
                                logger.error(f"SHAP ML: Error during fallback SHAP value calculation: {e_fallback_ml}", exc_info=True)
                        else: raise
                    except Exception as e_shap_values_other_ml:
                        logger.error(f"SHAP ML: Error obtaining SHAP values for multi-label: {e_shap_values_other_ml}", exc_info=True)

                    if shap_values_m_explanation:
                        probs_for_pred_indices = ml_probs_for_shap[0, pred_ml_indices].cpu().numpy()
                        sorted_pred_ml_indices_with_probs = sorted(zip(probs_for_pred_indices, pred_ml_indices), reverse=True)
                        
                        explained_count = 0
                        for prob, label_idx in sorted_pred_ml_indices_with_probs:
                            if explained_count >= top_n_ml_explain: break
                            label_name = mlb_loaded.classes_[label_idx]
                            
                            try:
                                if not (hasattr(shap_values_m_explanation, 'shape') and len(shap_values_m_explanation.shape) >= 3):
                                    logger.warning(f"SHAP ML: Invalid SHAP values shape for '{label_name}'. Skipping plot.")
                                    continue

                                exp_slice_m = shap_values_m_explanation[0, :, label_idx]
                                plot_arg_m = exp_slice_m

                                if (hasattr(exp_slice_m, 'data') and exp_slice_m.data is not None and
                                    hasattr(exp_slice_m, 'values') and exp_slice_m.values is not None and
                                    hasattr(exp_slice_m.data, '__len__') and len(exp_slice_m.data) == len(exp_slice_m.values) and
                                    all(isinstance(t, str) for t in exp_slice_m.data)):

                                    original_tokens_m = np.array(exp_slice_m.data)
                                    indices_to_keep_m = [i for i, token in enumerate(original_tokens_m) if len(token.strip()) >= 4]
                                    filtered_tokens_m = original_tokens_m[indices_to_keep_m]
                                    filtered_values_m = exp_slice_m.values[indices_to_keep_m]

                                    if len(filtered_tokens_m) > 0:
                                        plot_arg_m = shap.Explanation(
                                            values=filtered_values_m,
                                            base_values=exp_slice_m.base_values,
                                            data=filtered_tokens_m,
                                            feature_names=list(filtered_tokens_m)
                                        )
                                        logger.info(f"SHAP ML: Filtered tokens for '{label_name}'. Kept {len(filtered_tokens_m)}/{len(original_tokens_m)}.")
                                    else:
                                        logger.info(f"SHAP ML: All tokens filtered out for '{label_name}'. Plot will be skipped.")
                                
                                if hasattr(plot_arg_m, 'data') and len(plot_arg_m.data) > 0:
                                    plt.figure()
                                    shap.waterfall_plot(plot_arg_m, max_display=max_features_to_display, show=False)
                                    title = f"SHAP Категория: {label_name}"
                                    plt.title(title, fontsize=10)
                                    filename = f"shap_ml_{label_name.replace(' ', '_').replace('/', '_')}_{uuid.uuid4().hex[:8]}.png"
                                    plt.savefig(filename, bbox_inches='tight')
                                    plt.close()
                                    logger.info(f"SHAP ML: Saved plot {filename} for '{label_name}'")
                                    yield filename, title
                                    if delete_on_yield: # Проверяем флаг перед удалением
                                        if os.path.exists(filename): # Доп. проверка перед удалением
                                            try: os.remove(filename); logger.info(f"SHAP ML: Removed plot {filename} (delete_on_yield=True).")
                                            except Exception as e_remove_ml: logger.error(f"SHAP ML: Error removing {filename}: {e_remove_ml}")
                                    explained_count += 1
                                else:
                                    logger.info(f"SHAP ML: No data to plot for '{label_name}' after filtering. Plot skipped.")

                            except Exception as e_plot_gen_ml:
                                logger.error(f"SHAP ML: Error generating plot for '{label_name}': {e_plot_gen_ml}", exc_info=True)
                    elif explainer_m:
                        logger.warning("SHAP ML: SHAP values for multi-label are None. Plots cannot be generated.")
                else:
                    logger.info("SHAP ML: Explainer_m was not created, skipping SHAP for multi-label.")
        pass # Конец основной функции-генератора
    except Exception as e_global:
        logger.error(f"SHAP: Global error in generate_shap_plots for sentence '{sentence_to_explain[:100]}...': {e_global}", exc_info=True)
        # Генератор просто завершится, если была глобальная ошибка
        pass 