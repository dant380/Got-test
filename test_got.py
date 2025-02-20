# test_got.py

from examples.sorting.sorting_032 import SortingPrompter, SortingParser, utils
from graph_of_thoughts import controller, language_models, operations

def main():
    # Входные данные: строковое представление списка чисел для сортировки
    to_be_sorted = "[0, 2, 6, 3, 8, 7, 1, 1, 6, 7, 7, 7, 7, 9, 3, 0, 1, 7, 9, 1, 3, 5, 1, 3, 6, 4, 5, 4, 7, 3, 5, 7]"

    # Создаем граф операций (Graph of Operations)
    # Здесь мы используем CoT-подход (цепочка рассуждений)
    gop = operations.GraphOfOperations()
    gop.append_operation(operations.Generate())  # операция генерации ответа
    gop.append_operation(operations.Score(scoring_function=utils.num_errors))  # операция оценки (считает количество ошибок)
    gop.append_operation(operations.GroundTruth(utils.test_sorting))  # проверка конечного результата

    # Настраиваем модель LLM.
    # Для работы необходим корректно настроенный config.json с API-ключом.
    lm = language_models.ChatGPT("config.json", model_name="chatgpt")

    # Задаем начальное состояние для контроллера
    initial_state = {
        "original": to_be_sorted,
        "current": "",
        "method": "cot"  # можно изменить на "got", если используешь другой метод
    }

    # Создаем контроллер, который будет управлять выполнением операций
    ctrl = controller.Controller(
        lm,
        gop,
        SortingPrompter(),
        SortingParser(),
        initial_state
    )

    # Запускаем процесс рассуждений
    ctrl.run()

    # Сохраняем граф рассуждений в файл для последующего анализа
    ctrl.output_graph("output_test.json")
    print("Тестовый скрипт завершен. Граф рассуждений сохранен в output_test.json")

if __name__ == "__main__":
    main()
