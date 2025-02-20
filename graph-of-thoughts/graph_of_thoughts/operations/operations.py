# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach

from __future__ import annotations
import logging
from enum import Enum
from typing import List, Iterator, Dict, Callable, Union
from abc import ABC, abstractmethod
import itertools

from graph_of_thoughts.operations.thought import Thought
from graph_of_thoughts.language_models import AbstractLanguageModel
from graph_of_thoughts.prompter import Prompter
from graph_of_thoughts.parser import Parser

class OperationType(Enum):
    """
    Enum для представления типов операций.
    """
    score = 0
    validate_and_improve = 1
    generate = 2
    improve = 3
    aggregate = 4
    keep_best_n = 5
    keep_valid = 6
    ground_truth_evaluator = 7
    selector = 8

class Operation(ABC):
    """
    Абстрактный базовый класс для всех операций.
    """

    _ids: Iterator[int] = itertools.count(0)
    operation_type: OperationType = None

    def __init__(self) -> None:
        """
        Инициализирует операцию с уникальным идентификатором и пустыми списками предшественников и наследников.
        """
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.id: int = next(Operation._ids)
        self.predecessors: List[Operation] = []
        self.successors: List[Operation] = []
        self.executed: bool = False

    def can_be_executed(self) -> bool:
        """
        Проверяет, выполнены ли все предшествующие операции.
        """
        return all(predecessor.executed for predecessor in self.predecessors)

    def get_previous_thoughts(self) -> List[Thought]:
        """
        Собирает мысли из всех предшествующих операций.
        """
        return [thought for predecessor in self.predecessors for thought in predecessor.get_thoughts()]

    def add_predecessor(self, operation: Operation) -> None:
        """
        Добавляет предшествующую операцию и обновляет связи.
        """
        self.predecessors.append(operation)
        operation.successors.append(self)

    def add_successor(self, operation: Operation) -> None:
        """
        Добавляет наследующую операцию и обновляет связи.
        """
        self.successors.append(operation)
        operation.predecessors.append(self)

    def execute(self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs) -> None:
        """
        Выполняет операцию, предварительно проверив, что все предшествующие операции выполнены.
        """
        assert self.can_be_executed(), "Not all predecessors have been executed"
        self.logger.info("Executing operation %d of type %s", self.id, self.operation_type)
        self._execute(lm, prompter, parser, **kwargs)
        self.logger.debug("Operation %d executed", self.id)
        self.executed = True

    @abstractmethod
    def _execute(self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs) -> None:
        """
        Абстрактный метод выполнения операции.
        """
        pass

    @abstractmethod
    def get_thoughts(self) -> List[Thought]:
        """
        Абстрактный метод для получения мыслей, созданных операцией.
        """
        pass

class Score(Operation):
    """
    Операция для оценки мыслей.
    """

    operation_type: OperationType = OperationType.score

    def __init__(self, num_samples: int = 1, combined_scoring: bool = False,
                 scoring_function: Callable[[Union[List[Dict], Dict]], Union[List[float], float]] = None) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.combined_scoring = combined_scoring
        self.thoughts: List[Thought] = []
        self.scoring_function = scoring_function

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts

    def _execute(self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs) -> None:
        previous_thoughts = self.get_previous_thoughts()
        assert len(self.predecessors) > 0, "Score operation needs at least one predecessor"

        if self.combined_scoring:
            states = [t.state for t in previous_thoughts]
            if self.scoring_function:
                self.logger.debug("Using custom scoring function.")
                scores = self.scoring_function(states)
            else:
                prompt = prompter.score_prompt(states)
                self.logger.debug("Score prompt: %s", prompt)
                responses = lm.get_response_texts(lm.query(prompt, num_responses=self.num_samples))
                self.logger.debug("Score responses: %s", responses)
                scores = parser.parse_score_answer(states, responses)
            for t, s in zip(previous_thoughts, scores):
                new_t = Thought.from_thought(t)
                new_t.score = s
                self.thoughts.append(new_t)
        else:
            for t in previous_thoughts:
                new_t = Thought.from_thought(t)
                if self.scoring_function:
                    self.logger.debug("Using custom scoring function for state: %s", t.state)
                    s = self.scoring_function(t.state)
                else:
                    prompt = prompter.score_prompt([t.state])
                    self.logger.debug("Score prompt: %s", prompt)
                    responses = lm.get_response_texts(lm.query(prompt, num_responses=self.num_samples))
                    self.logger.debug("Score responses: %s", responses)
                    s = parser.parse_score_answer([t.state], responses)[0]
                new_t.score = s
                self.thoughts.append(new_t)
        self.logger.info("Score operation %d scored %d thoughts", self.id, len(self.thoughts))

class ValidateAndImprove(Operation):
    """
    Операция для проверки и улучшения мыслей.
    """

    operation_type: OperationType = OperationType.validate_and_improve

    def __init__(self, num_samples: int = 1, improve: bool = True, num_tries: int = 3,
                 validate_function: Callable[[Dict], bool] = None) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.improve = improve
        self.num_tries = num_tries
        self.validate_function = validate_function
        self.thoughts: List[List[Thought]] = []

    def get_thoughts(self) -> List[Thought]:
        return [t_list[-1] for t_list in self.thoughts]

    def _execute(self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs) -> None:
        previous_thoughts = self.get_previous_thoughts()
        assert len(self.predecessors) > 0, "ValidateAndImprove needs at least one predecessor"
        for t in previous_thoughts:
            branch = []
            # Передаём t как родителя для клонирования
            current = Thought.from_thought(t)
            tries = 0
            while True:
                if self.validate_function:
                    self.logger.debug("Validating with custom function for state: %s", current.state)
                    valid = self.validate_function(current.state)
                else:
                    prompt = prompter.validation_prompt(**current.state)
                    self.logger.debug("Validation prompt: %s", prompt)
                    responses = lm.get_response_texts(lm.query(prompt, num_responses=self.num_samples))
                    self.logger.debug("Validation responses: %s", responses)
                    valid = parser.parse_validation_answer(current.state, responses)
                current.valid = valid
                branch.append(current)
                if not self.improve or current.valid or tries >= self.num_tries:
                    break
                prompt = prompter.improve_prompt(**current.state)
                self.logger.debug("Improve prompt: %s", prompt)
                responses = lm.get_response_texts(lm.query(prompt, num_responses=1))
                self.logger.debug("Improve responses: %s", responses)
                update = parser.parse_improve_answer(current.state, responses)
                # Каждая новая итерация улучшения получает предыдущую мысль в качестве родителя
                current = Thought({**current.state, **update}, parent=current)
                tries += 1
            self.thoughts.append(branch)
        self.logger.info("ValidateAndImprove operation %d produced %d valid thoughts out of %d", 
                         self.id, len([b[-1] for b in self.thoughts if b[-1].valid]), len(previous_thoughts))

class Generate(Operation):
    """
    Операция генерации мыслей (ветвление).
    """

    operation_type: OperationType = OperationType.generate

    def __init__(self, num_branches_prompt: int = 3, num_branches_response: int = 2) -> None:
        super().__init__()
        self.num_branches_prompt = num_branches_prompt
        self.num_branches_response = num_branches_response
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts

    def _execute(self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs) -> None:
        self.logger.info("Начало операции Generate. Поиск предыдущих мыслей...")
        previous_thoughts = self.get_previous_thoughts()
        if len(previous_thoughts) == 0 and len(self.predecessors) > 0:
            self.logger.warning("Нет предыдущих мыслей, хотя есть предшествующие операции. Пропуск генерации.")
            return
        if len(previous_thoughts) == 0:
            self.logger.info("Предыдущих мыслей нет. Используем kwargs как базовое состояние.")
            previous_thoughts = [Thought(state=kwargs)]
        for idx, t in enumerate(previous_thoughts):
            base_state = t.state
            self.logger.info("Обработка мысли #%d: %s", idx, base_state)
            prompt = prompter.generate_prompt(self.num_branches_prompt, **base_state)
            self.logger.debug("Сгенерированный промпт: %s", prompt)
            responses = lm.get_response_texts(lm.query(prompt, num_responses=self.num_branches_response))
            self.logger.debug("Получены ответы от LM: %s", responses)
            self.logger.info("Обработка ответов для мысли #%d", idx)
            new_states = parser.parse_generate_answer(base_state, responses)
            if isinstance(new_states, dict):
                new_states = [new_states]
            for branch_idx, state_update in enumerate(new_states):
                if not isinstance(state_update, dict):
                    state_update = {"result": state_update}
                merged = {**base_state, **state_update}
                self.logger.info("Генерация ветки #%d для мысли #%d: %s", branch_idx, idx, merged)
                # Передаем t как родителя для новой мысли
                new_thought = Thought(merged, parent=t)
                self.thoughts.append(new_thought)
                self.logger.debug("Создана мысль #%d с состоянием: %s", new_thought.id, new_thought.state)
        self.logger.info("Операция Generate завершена. Создано %d мыслей.", len(self.thoughts))

class Improve(Operation):
    """
    Операция улучшения мыслей.
    """

    operation_type: OperationType = OperationType.improve

    def __init__(self) -> None:
        super().__init__()
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts

    def _execute(self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs) -> None:
        previous_thoughts = self.get_previous_thoughts()
        assert len(self.predecessors) > 0, "Improve operation needs at least one predecessor"
        improved_thoughts = []
        # Для каждого предыдущего ответа запрашиваем два независимых варианта улучшения
        for t in previous_thoughts:
            prompt = prompter.improve_prompt(**t.state)
            self.logger.debug("Improve prompt: %s", prompt)
            # Здесь запрашиваем два ответа от LLM
            responses = lm.get_response_texts(lm.query(prompt, num_responses=2))
            self.logger.debug("Improve responses: %s", responses)
            # Для каждого полученного ответа создаем новую улучшенную мысль
            for response in responses:
                update = parser.parse_improve_answer(t.state, [response])
                new_state = {**t.state, **update}
                # Передаем t как родителя для новой улучшенной мысли
                improved_thoughts.append(Thought(new_state, parent=t))
        self.thoughts = improved_thoughts
        self.logger.info("Improve operation %d improved %d thoughts", self.id, len(self.thoughts))

class Aggregate(Operation):
    """
    Операция агрегации мыслей.
    """

    operation_type: OperationType = OperationType.aggregate

    def __init__(self, num_responses: int = 1) -> None:
        super().__init__()
        self.thoughts: List[Thought] = []
        self.num_responses = num_responses

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts

    def _execute(self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs) -> None:
        assert len(self.predecessors) >= 1, "Aggregate operation must have at least one predecessor"
        previous_thoughts = self.get_previous_thoughts()
        if not previous_thoughts:
            return
        base_state = {}
        # Объединяем состояния всех предыдущих мыслей
        for t in sorted(previous_thoughts, key=lambda t: t.score if hasattr(t, "score") else 0):
            base_state = {**base_state, **t.state}
        states = [t.state for t in previous_thoughts]
        prompt = prompter.aggregation_prompt(states)
        self.logger.debug("Aggregation prompt: %s", prompt)
        responses = lm.get_response_texts(lm.query(prompt, num_responses=self.num_responses))
        self.logger.debug("Aggregation responses: %s", responses)
        aggregated = parser.parse_aggregation_answer(states, responses)
        if isinstance(aggregated, dict):
            aggregated = [aggregated]
        for update in aggregated:
            self.thoughts.append(Thought({**base_state, **update}))
        self.logger.info("Aggregate operation %d aggregated %d thoughts", self.id, len(self.thoughts))

class KeepBestN(Operation):
    """
    Операция выбора лучших мыслей.
    """

    operation_type: OperationType = OperationType.keep_best_n

    def __init__(self, n: int, higher_is_better: bool = True) -> None:
        super().__init__()
        self.n = n
        assert self.n > 0, "KeepBestN must keep at least one thought"
        self.higher_is_better = higher_is_better
        self.thoughts: List[Thought] = []

    def get_best_n(self) -> List[Thought]:
        previous_thoughts = self.get_previous_thoughts()
        assert all(hasattr(t, "score") for t in previous_thoughts), "Not all thoughts have scores"
        try:
            return sorted(previous_thoughts, key=lambda t: t.score, reverse=self.higher_is_better)[:self.n]
        except Exception as e:
            self.logger.error("Error in KeepBestN: %s", e)
            return []

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts

    def _execute(self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs) -> None:
        assert len(self.predecessors) >= 1, "KeepBestN operation needs at least one predecessor"
        best = self.get_best_n()
        self.thoughts = [Thought.from_thought(t) for t in best]
        for t in self.thoughts:
            self.logger.debug("Kept thought %d: %s", t.id, t.state)
        self.logger.info("KeepBestN operation %d kept %d thoughts", self.id, len(self.thoughts))

class KeepValid(Operation):
    """
    Операция для сохранения валидных мыслей.
    """

    operation_type: OperationType = OperationType.keep_valid

    def __init__(self) -> None:
        super().__init__()
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts

    def _execute(self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs) -> None:
        assert len(self.predecessors) >= 1, "KeepValid operation needs at least one predecessor"
        self.thoughts = [Thought.from_thought(t) for t in self.get_previous_thoughts() if not t.validated or t.valid]
        if any(not t.validated for t in self.thoughts):
            self.logger.warning("KeepValid operation %d has unvalidated thoughts", self.id)
        for t in self.thoughts:
            self.logger.debug("Valid thought %d: %s", t.id, t.state)
        self.logger.info("KeepValid operation %d kept %d thoughts", self.id, len(self.thoughts))

class GroundTruth(Operation):
    """
    Операция оценки мыслей с использованием ground truth.
    """

    operation_type: OperationType = OperationType.ground_truth_evaluator

    def __init__(self, ground_truth_evaluator: Callable[[Dict], bool]) -> None:
        super().__init__()
        self.ground_truth_evaluator = ground_truth_evaluator
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts

    def _execute(self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs) -> None:
        assert len(self.predecessors) >= 1, "GroundTruth operation needs at least one predecessor"
        previous_thoughts = self.get_previous_thoughts()
        for t in previous_thoughts:
            new_t = Thought.from_thought(t)
            try:
                new_t.solved = self.ground_truth_evaluator(new_t.state)
            except Exception:
                new_t.solved = False
            self.thoughts.append(new_t)
        self.logger.info("GroundTruth operation %d evaluated %d thoughts, %d solved", 
                         self.id, len(self.thoughts), len([t for t in self.thoughts if t.solved]))

class Selector(Operation):
    """
    Операция выбора мыслей на основе пользовательской логики.
    """

    operation_type: OperationType = OperationType.selector

    def __init__(self, selector: Callable[[List[Thought]], List[Thought]]) -> None:
        super().__init__()
        self.selector = selector
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts

    def _execute(self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs) -> None:
        previous_thoughts = self.get_previous_thoughts()
        if not previous_thoughts:
            previous_thoughts = [Thought(state=kwargs)]
        selected = self.selector(previous_thoughts)
        self.thoughts = [Thought.from_thought(t) for t in selected]
        for t in self.thoughts:
            self.logger.debug("Selected thought %d: %s", t.id, t.state)
        self.logger.info("Selector operation %d selected %d thoughts", self.id, len(self.thoughts))