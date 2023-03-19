import pytest

from pylixir.core.base import GameState
from pylixir.data.council.operation import SwapMinMax, SwapValues
from tests.randomness import DeterministicRandomness


@pytest.mark.parametrize(
    "swap_target, start, end",
    [
        ([0, 1], [1, 3, 5, 7, 9], [3, 1, 5, 7, 9]),
        ([0, 2], [1, 3, 5, 7, 9], [5, 3, 1, 7, 9]),
        ([1, 4], [1, 3, 5, 7, 9], [1, 9, 5, 7, 3]),
        ([2, 3], [1, 3, 5, 7, 9], [1, 3, 7, 5, 9]),
    ],
)
def test_swap_between_values(
    swap_target: tuple[int, int],
    start: list[int],
    end: list[int],
    abundant_state: GameState,
) -> None:
    for idx in range(5):
        abundant_state.board.set_effect_count(idx, start[idx])

    operation = SwapValues(
        ratio=0,
        value=swap_target,
        remain_turn=1,
    )

    changed_state = operation.reduce(abundant_state, [], DeterministicRandomness(42))
    assert changed_state.board.get_effect_values() == end


@pytest.mark.parametrize(
    "locked_indices, start, end",
    [
        ([], [1, 3, 5, 7, 9], [9, 3, 5, 7, 1]),
        ([1], [1, 3, 5, 7, 9], [9, 3, 5, 7, 1]),
        ([1, 2], [1, 3, 5, 7, 9], [9, 3, 5, 7, 1]),
        ([0], [1, 3, 5, 7, 9], [1, 9, 5, 7, 3]),
        ([0, 1, 2], [1, 3, 5, 7, 9], [1, 3, 5, 9, 7]),
    ],
)
def test_swap_min_max(
    locked_indices: list[int],
    start: list[int],
    end: list[int],
    abundant_state: GameState,
) -> None:
    for idx in range(5):
        abundant_state.board.set_effect_count(idx, start[idx])

    for idx in locked_indices:
        abundant_state.board.lock(idx)

    operation = SwapMinMax(
        ratio=0,
        value=(0, 0),
        remain_turn=1,
    )

    changed_state = operation.reduce(abundant_state, [], DeterministicRandomness(42))
    assert changed_state.board.get_effect_values() == end
