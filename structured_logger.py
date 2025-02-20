class StructuredLogger:
    def __init__(self):
        self.logs = []
        self.current_iteration = 0
        self.thought_counter = 1
        
    def set_initial_query(self, query: str):
        self.logs = [
            "\n========================================",
            "         ПЕРВОНАЧАЛЬНЫЙ ЗАПРОС          ",
            f'"{query}"',
            "========================================\n"
        ]
        
    def start_iteration(self, iteration_num: int):
        self.current_iteration = iteration_num
        self.thought_counter = 1
        self.logs.append(f"\n=== Итерация #{iteration_num} ===")
        
    def log_thought(self, thought, scores: dict, selected: bool):
        content = str(thought.state)
        if len(content) > 200:
            content = content[:197] + "..."
            
        entry = (
            f"Мысль #{self.thought_counter}\n"
            f"Содержание: {content}\n"
            f"Оценки: {', '.join([f'{k} - {v}' for k, v in scores.items()])}\n"
            f"Выбрана для улучшения: {'Да' if selected else 'Нет'}\n"
        )
        self.logs.append(entry)
        self.thought_counter += 1
        
    def log_aggregation(self, aggregated_thoughts):
        self.logs.append("\n=== Мысли для агрегации ===")
        for i, thought in enumerate(aggregated_thoughts, 1):
            content = str(thought.state)
            if len(content) > 100:
                content = content[:97] + "..."
            self.logs.append(f"{i}. {content}")
            
    def log_final_answer(self, final_answer: str):
        self.logs.append("\n========================================")
        self.logs.append("            ФИНАЛЬНЫЙ ОТВЕТ             ")
        self.logs.append("========================================\n")
        self.logs.append(final_answer)
        
    def get_log(self) -> str:
        return "\n".join(self.logs)