import logging
import datetime

# Генерируем отметку времени для уникальных имён лог-файлов (формат: YYYYMMDD_HHMMSS)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Имена файлов для детального и отчётного логов
detail_log_file = f"GraphPrompt_detail_{timestamp}.log"
report_log_file = f"GraphPrompt_report_{timestamp}.txt"

# Настройка корневого логгера для детального логирования
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)  # принимаем все сообщения вплоть до DEBUG
# Добавляем файл-обработчик для детального лога (если ещё не добавлен)
if not root_logger.handlers:
    detail_handler = logging.FileHandler(detail_log_file, mode='w')
    detail_handler.setLevel(logging.DEBUG)
    # Формат: время, уровень, сообщение
    detail_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    detail_handler.setFormatter(detail_format)
    root_logger.addHandler(detail_handler)

# Настройка отдельного логгера для отчётов
report_logger = logging.getLogger("report")
report_logger.setLevel(logging.INFO)
report_logger.propagate = False  # отключаем передачу сообщений корневому логгеру
# Добавляем файл-обработчик для отчётного лога (если ещё не добавлен)
if not report_logger.handlers:
    report_handler = logging.FileHandler(report_log_file, mode='w')
    report_handler.setLevel(logging.INFO)
    # Формат: только текст сообщения (отчётные сообщения лаконичные)
    report_format = logging.Formatter("%(message)s")
    report_handler.setFormatter(report_format)
    report_logger.addHandler(report_handler)

def enable_report_console_output():
    """
    Подключает вывод отчётных сообщений в консоль.
    Вызывать эту функцию, если нужно видеть отчёт в терминале.
    """
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(report_format)
    report_logger.addHandler(console_handler)
