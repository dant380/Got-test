# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach

from __future__ import annotations
import logging
from typing import Iterator, Dict, Optional
import itertools

class Thought:
    """
    Represents an LLM thought with its state, constructed by the parser, and various flags.
    Кожна думка отримує унікальний id, а також, якщо вона створена з іншої думки, зберігає її id як parent_id.
    """

    _ids: Iterator[int] = itertools.count(0)

    def __init__(self, state: Optional[Dict] = None, parent: Optional[Thought] = None) -> None:
        """
        Initializes a new Thought instance with a state and various default flags.
        Якщо задано parent, зберігається його id у полі parent_ids.

        :param state: The state of the thought. Defaults to None.
        :type state: Optional[Dict]
        :param parent: (Optional) Parent Thought instance.
        :type parent: Optional[Thought]
        """
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.id: int = next(Thought._ids)
        self.state: Dict = state or {}
        self.parent_ids = [parent.id] if parent is not None else []
        self._score: float = 0.0
        self._valid: bool = False
        self._solved: bool = False
        self.scored: bool = False
        self.validated: bool = False
        self.compared_to_ground_truth: bool = False

    @staticmethod
    def from_thought(thought: Thought) -> Thought:
        """
        Creates a new Thought from an existing one.
        Новостворена думка отримує як parent id оригінальної думки.

        :param thought: An instance of a Thought to clone.
        :return: A new Thought instance with properties copied from the input thought.
        """
        new_thought = Thought(thought.state, parent=thought)
        new_thought.score = thought.score
        new_thought.valid = thought.valid
        new_thought.solved = thought.solved
        new_thought.scored = thought.scored
        new_thought.validated = thought.validated
        new_thought.compared_to_ground_truth = thought.compared_to_ground_truth
        return new_thought

    @property
    def valid(self) -> bool:
        """
        Returns the validity of the thought.
        """
        return self._valid

    @valid.setter
    def valid(self, valid: bool) -> None:
        """
        Sets the validity of the thought and the validated flag.
        """
        self.validated = True
        self._valid = valid

    @property
    def score(self) -> float:
        """
        Returns the score of the thought.
        """
        return self._score

    @score.setter
    def score(self, new_score: float) -> None:
        """
        Sets the score of the thought and the scored flag.
        """
        self.scored = True
        self._score = new_score

    @property
    def solved(self) -> bool:
        """
        Returns the solved flag of the thought.
        """
        return self._solved

    @solved.setter
    def solved(self, solved: bool) -> None:
        """
        Sets the solved flag of the thought and the compared_to_ground_truth flag.
        """
        self.compared_to_ground_truth = True
        self._solved = solved
