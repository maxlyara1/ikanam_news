# ikanam_news

Это проект для решения задачи классификации текстов с использованием моделей глубокого обучения.

## Структура проекта
(некоторые папки сформируются, либо их надо сздать в ходе запуска кода, в гитхабе их нет)

```
ikanam_news/
├── assets/                       # Статические файлы
│   └── shap_plots/
├── data/                         # Данные проекта
│   ├── external/                 # Внешние данные
│   │   └── data_for_llm/
│   ├── raw/                      # Необработанные ("сырые") данные
│   └── user_data/                # Пользовательские данные
├── notebooks/                    # Jupyter ноутбуки для экспериментов и анализа
├── saved_models/                 # Сохраненные обученные модели
├── scripts/                      # Скрипт для запуска авторизации в TG
├── sessions/                     # Файлы сессий
├── src/                          # Исходный код проекта
│   ├── bot/                      # Код для бота
│   ├── core/                     # Основная логика проекта
│   └── workers/                  # Код для воркеров или фоновых задач
├── .gitattributes                # Атрибуты Git
├── .gitignore                    # Файлы и папки, игнорируемые Git
└── environment.yml               # Файл для управления зависимостями Python (Conda)
```

## Установка

1.  **Клонируйте репозиторий:**
    ```bash
    git clone https://github.com/maxlyara1/ikanam_news.git
    cd ikanam_news
    ```

2.  **Создайте и активируйте окружение Conda:**
    Если у вас установлен Conda, вы можете создать окружение на основе файла `environment.yml`:
    ```bash
    conda env create -f environment.yml
    conda activate ikanam_news_env
    ```

    **Установите зависимости (альтернативный способ, если вы используете pip):**
    Если вы используете `pip` и у вас есть файл `requirements.txt`, выполните:
    ```bash
    pip install -r requirements.txt
    ```

## Использование

*   **Для запуска скрипта получения данных:**\
    Запустите `notebooks/train_dataset_getting.ipynb`

*   **Для запуска скрипта обучения моделей:**\
    Запустите `notebooks/train_model_process.ipynb`

*   **Для запуска бота:**
    ```bash
    python -m scripts.authorize_telethon 
    python -m src.bot.telegram_bot
    ```