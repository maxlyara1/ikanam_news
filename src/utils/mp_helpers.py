# src/utils/mp_helpers.py
import os
import sys
import time
import traceback

def worker_initializer(q_passed, project_root_path_for_worker=None):
    pid = os.getpid()
    base_path = project_root_path_for_worker if project_root_path_for_worker else os.getcwd()
    worker_init_error_log_path = os.path.join(base_path, f"worker_init_ERROR_PID_{pid}.txt")

    try:

        from src.workers.gemini_workers import setup_worker_logging
        setup_worker_logging(q_passed)

    except Exception as e:
        # Записываем ошибку в файл, если что-то пошло не так ВНУТРИ worker_initializer
        try:
            with open(worker_init_error_log_path, "w") as f_err: # 'w' для перезаписи
                f_err.write(f"Error in worker_initializer (PID {pid}):\n{type(e).__name__}: {e}\n\nTraceback:\n{traceback.format_exc()}")
        except:
            pass # Если и это не удалось
        raise # Перевыбрасываем ошибку, чтобы пул ее увидел