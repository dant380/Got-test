import logging
import logger_setup  # выполняет настройку логирования

# Получаем логгеры
logger = logging.getLogger()           # корневой логгер для детальных сообщений
report_logger = logging.getLogger("report")  # логгер для отчётных сообщений

logger.debug("Отладочное сообщение")   # запишется в GraphPrompt_detail_*.log
report_logger.info("Важный итоговый результат")  # запишется в GraphPrompt_report_*.txt

# По необходимости включаем вывод отчёта в консоль:
logger_setup.enable_report_console_output()
report_logger.info("Это сообщение появится и в консоли, и в отчётном файле")

import sys
import json
from graph_of_thoughts import controller, language_models, operations

############################################
# 2. Реализация Prompter и Parser
############################################

class MyPrompter:
    def generate_prompt(self, num_branches_prompt, **thought_state) -> str:
        query = thought_state.get("query", "напиши анегдот одним предложением")
        prompt = (
            f"Пожалуйста, сгенерируй подробный ответ (в формате JSON) на вопрос:\n"
            f"{query}\n\n"
            "напиши анегдот одним предложениемn"
            "Ответ должен быть в формате JSON, например:\n"
            "Сгенерируй несколько разных вариантов, так как у нас несколько веток."
        )
        logging.getLogger().info(f"Генерация промпта. Запрос: {query}")
        return prompt

    def score_prompt(self, thought_states):
        logging.getLogger().info("Формирование промпта для оценки")
        return "Оцени полноту и полезность следующих ответов по шкале от 0 до 10:\n" + f"{thought_states}"

    def validation_prompt(self, **state):
        logging.getLogger().info(f"Формирование промпта для валидации. Состояние: {state}")
        return "Считаешь ли ты этот ответ корректным и содержательным? " + str(state)

    def improve_prompt(self, **state):
        logging.getLogger().info(f"Формирование промпта для улучшения. Состояние: {state}")
        return "Как улучшить этот ответ? Пиши только сам ответ, без обьяснений" + str(state)

    def aggregation_prompt(self, thought_states):
        logging.getLogger().info(f"Формирование промпта для агрегации. Мысли: {thought_states}")
        return (
            "Объедини полезные части из следующих ответов в один итоговый:\n"
            f"{thought_states}\n"
            "Ответ снова в формате JSON."
        )

class MyParser:
    def parse_generate_answer(self, base_state, responses):
        logging.getLogger().info("Парсинг ответа генерации")
        if isinstance(responses, list) and responses:
            text = responses[0].strip()
        else:
            text = ""
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            data = json.loads(text)
            return {**base_state, **data}
        except Exception as e:
            logging.getLogger().error(f"Ошибка парсинга генерации: {str(e)}")
            return {**base_state, "advice": text}

    def parse_score_answer(self, thought_states, responses):
        logging.getLogger().info("Парсинг ответа оценки")
        return [5.0 for _ in thought_states]

    def parse_validation_answer(self, state, responses):
        logging.getLogger().info("Парсинг ответа валидации")
        return bool(responses and responses[0].strip())

    def parse_improve_answer(self, state, responses):
        return {"improved_advice": responses[0].strip()} if responses else {}

    def parse_aggregation_answer(self, previous_thought_states, responses):
        if isinstance(responses, list) and responses:
            try:
                return json.loads(responses[0])
            except Exception:
                return {"advice": responses[0]}
        return {}

############################################
# 3. Сборка графа операций и запуск
############################################

def format_thought(thought, scores):
    state = json.dumps(thought.state, ensure_ascii=False, indent=2)
    score_info = "\n".join([f"Оценка {i+1}: {score}" for i, score in enumerate(scores)]) if scores else "Оценки отсутствуют"
    chosen = "Да" if thought.parent_ids else "Нет"
    return (
        f"\n{'='*40}\n"
        f"Мысль ID: {thought.id}\n"
        f"{state}\n"
        f"{score_info}\n"
        f"Выбрана для улучшения: {chosen}\n"
        f"{'='*40}"
    )

def log_iteration(iteration, thoughts, scores_history):
    report_logger.info(f"\n{'#'*40}")
    report_logger.info(f"Итерация #{iteration}".center(40))
    report_logger.info(f"{'#'*40}")
    report_logger.info(f"Логирование {len(thoughts)} мыслей в итерации #{iteration}")

    for thought in thoughts:
        text = thought.state.get("text", "Нет текста")  # Достаём текст мысли из state
        score = thought.score  # Получаем оценку мысли

        # Логируем в отчётный лог (только отчёт, без дублирования в технический лог)
        report_logger.info(f"Мысль: {text} (Оценка: {score})")

    report_logger.handlers[0].flush()  # Принудительная запись в файл

print(logging.getLogger("report").handlers)


def main():
    # Заголовок отчёта (первоначальный запрос)
    initial_state = {"query": "Анегдот одним предложением"}
    report_logger.info(f"\n{'='*40}")
    report_logger.info("ПЕРВОНАЧАЛЬНЫЙ ЗАПРОС".center(40))
    report_logger.info(f"{initial_state['query']}")
    report_logger.info(f"{'='*40}\n")
    
    # Инициализация структур
    scores_history = {}
    
    # Создание графа операций
    gop = operations.GraphOfOperations()
    
    # Итерация 1: Генерация и оценка
    gen_op = operations.Generate(2, 3)
    score_op = operations.Score(2)
    keep_op = operations.KeepBestN(4, True)
    gop.append_operation(gen_op)
    gop.append_operation(score_op)
    gop.append_operation(keep_op)
    report_logger.info("Итерация 1 завершена")
    report_logger.handlers[0].flush()  # Принудительная запись в файл
    
    # Итерация 2: Улучшение и оценка
    improve_op1 = operations.Improve()
    score_op2 = operations.Score(2)
    keep_op2 = operations.KeepBestN(5, True)
    gop.append_operation(improve_op1)
    gop.append_operation(score_op2)
    gop.append_operation(keep_op2)
    report_logger.info("Итерация 2 завершена")
    report_logger.handlers[0].flush()  # Принудительная запись в файл

    
    # Итерация 3: Финальное улучшение и агрегация
    improve_op2 = operations.Improve()
    score_op3 = operations.Score(2)
    aggregate_op = operations.Aggregate(1)
    gop.append_operation(improve_op2)
    gop.append_operation(score_op3)
    gop.append_operation(aggregate_op)
    report_logger.info("Итерация 3 завершена. Финальное улучшение и агрегация завершена")
    report_logger.handlers[0].flush()  # Принудительная запись в файл


    # Настройка модели и контроллера
    lm = language_models.ChatGPT("config.json", "chatgpt")
    prompter = MyPrompter()
    parser = MyParser()
    ctrl = controller.Controller(lm, gop, prompter, parser, initial_state)
    
    report_logger.info("Перед запуском ctrl.run()")
    report_logger.handlers[0].flush()  # Принудительная запись в файл


    # Запуск процесса
    report_logger.info(f"\n{'#'*40}")
    report_logger.info("НАЧАЛО ОБРАБОТКИ".center(40))
    report_logger.info(f"{'#'*40}\n")
    ctrl.run()

    report_logger.info("После выполнения ctrl.run()")
    report_logger.handlers[0].flush()  # Принудительная запись в файл

    
    # Логирование итераций: перебор всех операций, где есть «думки»
    for idx, op in enumerate(gop.operations, start=1):
        thoughts = op.get_thoughts()  # Предполагается, что каждая операция реализует этот метод
        if thoughts:
            log_iteration(idx, thoughts, scores_history)
    
    # Логирование финального результата
    report_logger.info(f"\n{'#'*40}")
    report_logger.info("ФИНАЛЬНЫЙ ОТВЕТ".center(40))
    report_logger.info(f"{'#'*40}")
    if gop.operations:
        final_op = gop.operations[-1]
        final_thoughts = final_op.get_thoughts()
        if final_thoughts:
            try:
                final_data = final_thoughts[0].state
                advice = final_data.get('improved_advice', final_data.get('advice', ''))
                clean_advice = advice.replace('\\n', '\n').replace('\\"', '"')
                report_logger.info("\n" + "\n".join([line.strip() for line in clean_advice.split('\n')]))
            except Exception as e:
                detail_logger.error(f"Ошибка форматирования: {str(e)}")
                report_logger.info(json.dumps(final_thoughts[0].state, indent=2, ensure_ascii=False))
    
    report_logger.info(f"\n{'#'*40}")
    report_logger.info("ОБРАБОТКА ЗАВЕРШЕНА".center(40))
    report_logger.info(f"{'#'*40}")

if __name__ == "__main__":
    main()

import logging
logging.shutdown()
