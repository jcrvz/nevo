import numpy as np
import pytest

from nevo.core.basal_ganglia import BasalGangliaSelector
from nevo.operators.base import Operator


class DummyOperator(Operator):
    def __init__(self, name: str):
        super().__init__(name=name)

    def generate_population(self, centre, state, population_size):
        return np.zeros((population_size, 1))


def test_tie_break_is_unbiased():
    np.random.seed(0)
    operators = [DummyOperator(f"Op{i}") for i in range(4)]
    selector = BasalGangliaSelector(
        operators=operators,
        epsilon=0.0,
        learning_rate=0.0,
    )

    counts = {op.name: 0 for op in operators}
    trials = 4000
    tie_signal = np.full(len(operators), 0.5)

    for _ in range(trials):
        chosen = selector.select_operator(tie_signal, current_best_fitness=1.0)
        counts[chosen.name] += 1

    mean = np.mean(list(counts.values()))
    for count in counts.values():
        assert abs(count - mean) <= mean * 0.15, counts


def test_argmax_wins_when_clearly_larger():
    np.random.seed(1)
    operators = [DummyOperator(f"Op{i}") for i in range(4)]
    selector = BasalGangliaSelector(
        operators=operators,
        epsilon=0.0,
        learning_rate=0.0,
    )

    dominant = np.array([0.1, 0.9, 0.2, 0.3])
    for _ in range(50):
        chosen = selector.select_operator(dominant, current_best_fitness=1.0)
        assert chosen.name == "Op1"
