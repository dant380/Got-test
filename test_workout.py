import logging
import sys
import json
import random
from datetime import datetime
from graph_of_thoughts import controller, language_models, operations
from graph_of_thoughts.operations.thought import Thought

# Генерація унікальних імен файлів на основі дати і часу
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
detail_log_file = f"GraphPrompt_detail_{timestamp}.log"
report_log_file = f"GraphPrompt_report_{timestamp}.txt"

# Налаштування логерів
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
report_logger = logging.getLogger("report")
report_logger.setLevel(logging.INFO)

detail_handler = logging.FileHandler(detail_log_file, encoding="utf-8")
detail_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.addHandler(detail_handler)

report_handler = logging.FileHandler(report_log_file, encoding="utf-8")
report_handler.setFormatter(logging.Formatter('%(message)s'))
report_logger.addHandler(report_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
console_handler.setStream(open(console_handler.stream.fileno(), mode='w', encoding='utf-8', errors='replace'))
logger.addHandler(console_handler)
report_logger.addHandler(console_handler)

logger.debug("Отладочное сообщение")
report_logger.info("Це повідомлення з’явиться в консолі та звітному файлі")

############################################
# 2. Реализация Prompter и Parser
############################################

class MyPrompter:
    def generate_prompt(self, num_branches_prompt, **thought_state) -> str:
        query = thought_state.get("query", "напиши анегдот одним предложением")
        prompt = (
            f"Пожалуйста, сгенерируй анегдот одним предложением на тему SMM в формате JSON.\n"
            f"Запрос: {query}\n"
        )
        logger.info(f"Генерация промпта. Запрос: {query}")
        return prompt

    def score_prompt(self, thought_states):
        logger.info("Формирование промпта для оценки")
        return "Оцени полноту и полезность следующих ответов по шкале от 0 до 10:\n" + f"{thought_states}"

    def validation_prompt(self, **state):
        logger.info(f"Формирование промпта для валидации. Состояние: {state}")
        return "Считаешь ли ты этот ответ корректным и содержательным? " + str(state)

    def improve_prompt(self, **state):
        logger.info(f"Формирование промпта для улучшения. Состояние: {state}")
        return "Как улучшить этот ответ? Пиши только сам ответ, без обьяснений" + str(state)

    def aggregation_prompt(self, thought_states):
        logger.info(f"Формирование промпта для агрегации. Мысли: {thought_states}")
        return (
            "Объедини полезные части из следующих ответов в один итоговый:\n"
            f"{thought_states}\n"
            "Ответ снова в формате JSON."
        )

class MyParser:
    def parse_generate_answer(self, base_state, responses):
        logger.info(f"Парсинг ответа генерации. Отримано відповідей: {len(responses)}")
        thoughts = []
        for response in responses:
            text = response.strip()
            if text.startswith("```json"):
                text = text[7:-3].strip()
            elif text.startswith("```"):
                text = text[3:-3].strip()
            try:
                data = json.loads(text)
                if isinstance(data, list):
                    for item in data:
                        thoughts.append({**base_state, "advice": item.get("text", "Немає тексту")})
                else:
                    thoughts.append({**base_state, "advice": text})
            except Exception as e:
                logger.error(f"Ошибка парсинга генерации: {str(e)}")
                thoughts.append({**base_state, "advice": text})
        return thoughts

    def parse_score_answer(self, thought_states, responses):
        logger.info("Парсинг ответа оценки")
        return [random.uniform(1.0, 10.0) for _ in thought_states]

    def parse_validation_answer(self, state, responses):
        logger.info("Парсинг ответа валидации")
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
# 3. Функции для логирования и запуск
############################################

def log_iteration(step, thoughts, scores_history, selected_for_improvement):
    report_logger.info(f"\n{'#'*40}")
    report_logger.info(f"Крок {step}".center(40))
    report_logger.info(f"{'#'*40}")
    try:
        for idx, thought in enumerate(thoughts, start=1):
            text = thought.state.get("advice", thought.state.get("improved_advice", "Немає тексту"))
            scores = scores_history.get(thought.id, [5.0, 5.0])
            avg_score = sum(scores) / len(scores) if scores else 0
            selected = "так" if thought.id in selected_for_improvement else "ні"
            
            report_logger.info(f"думка {idx}:")
            if step > 1:
                parent_id = thought.parent_ids[0] if thought.parent_ids else "невідомо"
                report_logger.info(f"яка думка була вибрана для покращення: {parent_id}")
            report_logger.info(f"текст думки: {text}")
            report_logger.info(f"оцінка: {', '.join(map(str, scores))}, середня оцінка: {avg_score:.2f}")
            report_logger.info(f"чи вибраний для покращення: {selected}")
            report_logger.info("")
    except Exception as e:
        report_logger.error(f"Помилка під час логування ітерації {step}: {str(e)}")
        logger.error(f"Деталі помилки логування ітерації {step}: {str(e)}", exc_info=True)

def log_aggregation(final_thoughts):
    report_logger.info(f"\n{'#'*40}")
    report_logger.info("Агрегація".center(40))
    report_logger.info(f"{'#'*40}")
    try:
        selected_ids = [thought.id for thought in final_thoughts]
        report_logger.info(f"які думки були вибрані для агрегації фінальної відповіді: {', '.join(map(str, selected_ids))}")
        if final_thoughts:
            final_text = final_thoughts[0].state.get("advice", final_thoughts[0].state.get("improved_advice", "Немає тексту"))
            report_logger.info(f"Фінальна відповідь: {final_text}")
    except Exception as e:
        report_logger.error(f"Помилка під час логування агрегації: {str(e)}")
        logger.error(f"Деталі помилки логування агрегації: {str(e)}", exc_info=True)

def main():
    initial_state = {"query": "Анегдот одним предложением"}
    report_logger.info(f"\n{'='*40}")
    report_logger.info("Запит:".center(40))
    report_logger.info(f"{initial_state['query']}")
    report_logger.info(f"{'='*40}\n")
    
    scores_history = {}
    selected_for_improvement = set()
    
    try:
        logger.debug("Створення графа операцій")
        gop = operations.GraphOfOperations()
        
        gen_op = operations.Generate(1, 6)  # Змінено на 6 запитів по 1 відповіді
        score_op1 = operations.Score(1)
        keep_op1 = operations.KeepBestN(4, True)  # Залишаємо 4 найкращі
        gop.append_operation(gen_op)
        gop.append_operation(score_op1)
        gop.append_operation(keep_op1)
        
        improve_op1 = operations.Improve()  # 4 думки по 2 покращення = 8
        score_op2 = operations.Score(1)
        keep_op2 = operations.KeepBestN(5, True)  # Залишаємо 5 найкращих
        gop.append_operation(improve_op1)
        gop.append_operation(score_op2)
        gop.append_operation(keep_op2)
        
        improve_op2 = operations.Improve()  # 5 думок по 2 покращення = 10
        score_op3 = operations.Score(1)
        aggregate_op = operations.Aggregate(1)
        gop.append_operation(improve_op2)
        gop.append_operation(score_op3)
        gop.append_operation(aggregate_op)
        
        logger.debug("Перед початком налаштування моделі ChatGPT")
        lm = language_models.ChatGPT("config.json", "chatgpt")
        logger.debug("Модель ChatGPT успішно ініціалізована")
        
        prompter = MyPrompter()
        parser = MyParser()
        logger.debug("Створення контролера")
        ctrl = controller.Controller(lm, gop, prompter, parser, initial_state)
        
        report_logger.info(f"\n{'#'*40}")
        report_logger.info("НАЧАЛО ОБРАБОТКИ".center(40))
        report_logger.info(f"{'#'*40}\n")
        logger.debug("Запуск ctrl.run()")
        ctrl.run()
        logger.debug("ctrl.run() завершено")
        
        step = 1
        for op in gop.operations:
            if isinstance(op, (operations.Generate, operations.Improve)):
                logger.debug(f"Логування думок для операції {op.operation_type} на кроці {step}")
                thoughts = op.get_thoughts()
                if thoughts:
                    for thought in thoughts:
                        scores = [thought.score] if thought.scored else [5.0, 5.0]
                        scores_history[thought.id] = scores
                    
                    op_idx = gop.operations.index(op)
                    next_keep_op = gop.operations[op_idx + 2] if op_idx + 2 < len(gop.operations) else None
                    if next_keep_op and isinstance(next_keep_op, operations.KeepBestN):
                        kept_thoughts = next_keep_op.get_thoughts()
                        selected_for_improvement.update(thought.id for thought in kept_thoughts)
                    
                    log_iteration(step, thoughts, scores_history, selected_for_improvement)
                    step += 1
        
        logger.debug("Логування агрегації")
        final_thoughts = aggregate_op.get_thoughts()
        if final_thoughts:
            log_aggregation(final_thoughts)
        else:
            report_logger.info("Агрегація не виконана або не містить думок.")
        
        report_logger.info(f"\n{'#'*40}")
        report_logger.info("ОБРАБОТКА ЗАВЕРШЕНА".center(40))
        report_logger.info(f"{'#'*40}")
    
    except Exception as e:
        report_logger.error(f"\n{'#'*40}")
        report_logger.error("ПОМИЛКА ПІД ЧАС ВИКОНАННЯ".center(40))
        report_logger.error(f"{'#'*40}")
        report_logger.error(f"Помилка: {str(e)}")
        logger.error(f"Деталі помилки: {str(e)}", exc_info=True)
    
    finally:
        report_logger.handlers[0].flush()
        logger.handlers[0].flush()  # Додано для детального логу

if __name__ == "__main__":
    logger.debug("Скрипт запущено")
    main()
    logger.debug("Скрипт завершено")