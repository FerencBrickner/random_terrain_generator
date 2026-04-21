import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from random_terrain import normalize_terrain_to_the_0_to_1_range_safely
from random_terrain import apply_smootherstep_polynomial_perlin_noise


def test_normalizes_basic_1d_array():
    terrain = np.array([10.0, 20.0, 30.0])

    result = normalize_terrain_to_the_0_to_1_range_safely(terrain=terrain)

    expected = np.array([0.0, 0.5, 1.0])
    assert np.allclose(result, expected)


def test_normalizes_2d_array():
    terrain = np.array(
        [
            [2.0, 4.0],
            [6.0, 8.0],
        ]
    )

    result = normalize_terrain_to_the_0_to_1_range_safely(terrain=terrain)

    expected = np.array(
        [
            [0.0, 1 / 3],
            [2 / 3, 1.0],
        ]
    )
    assert np.allclose(result, expected)


def test_constant_terrain_returns_zeros():
    terrain = np.array([7.0, 7.0, 7.0])

    result = normalize_terrain_to_the_0_to_1_range_safely(terrain=terrain)

    expected = np.array([0.0, 0.0, 0.0])
    assert np.array_equal(result, expected)


def test_handles_negative_values_correctly():
    terrain = np.array([-5.0, 0.0, 5.0])

    result = normalize_terrain_to_the_0_to_1_range_safely(terrain=terrain)

    expected = np.array([0.0, 0.5, 1.0])
    assert np.allclose(result, expected)


def test_single_value_array_returns_zero():
    terrain = np.array([42.0])

    result = normalize_terrain_to_the_0_to_1_range_safely(terrain=terrain)

    expected = np.array([0.0])
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    "float_input, expected",
    [
        (0.0, 0.0),
        (1.0, 1.0),
        (0.5, 0.5),
        (0.25, 0.103515625),
        (0.75, 0.896484375),
    ],
)
def test_apply_smootherstep_polynomial_perlin_noise_expected_values(
    float_input, expected
):
    result = apply_smootherstep_polynomial_perlin_noise(float_input)

    assert result == pytest.approx(expected, rel=0, abs=0)


def test_apply_smootherstep_polynomial_perlin_noise_monotonic_in_unit_interval():
    values = [
        apply_smootherstep_polynomial_perlin_noise(x)
        for x in [0.0, 0.1, 0.2, 0.5, 0.8, 1.0]
    ]

    assert values == sorted(values)


def test_apply_smootherstep_polynomial_perlin_noise_returns_float():
    result = apply_smootherstep_polynomial_perlin_noise(0.5)

    assert isinstance(result, float)
