import itertools
import pytest

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from PRNGs.xorshift32 import xorshift_32_float_generator
from PRNGs.logistic_map import logistic_map_pseudorandom_generator
from PRNGs.wichmann_hill import wichmann_hill_generator


@pytest.mark.parametrize(
    "seed, expected",
    [
        (
            1,
            [
                6.295018829405308e-05,
                0.015747428173199296,
                0.6164041024167091,
                0.07161863497458398,
                0.5584883580449969,
            ],
        ),
        (
            2,
            [
                0.00012590037658810616,
                0.03125834511592984,
                0.16248849080875516,
                0.3890491141937673,
                0.8894066871143878,
            ],
        ),
        (
            0xFFFFFFFF,
            [
                5.913502536714077e-05,
                0.9844969508703798,
                0.4559874690603465,
                0.9445271999575198,
                0.4771870712283999,
            ],
        ),
    ],
)
def test_xorshift_32_float_generator_produces_expected_sequence(seed, expected):
    gen = xorshift_32_float_generator(seed=seed)
    result: list[float] = [next(gen) for _ in range(len(expected))]

    assert result == pytest.approx(expected, rel=0, abs=0)


def test_xorshift_32_float_generator_seed_zero_behaves_like_seed_one():
    gen_zero = xorshift_32_float_generator(seed=0)
    gen_one = xorshift_32_float_generator(seed=1)

    zero_values: list[float] = [next(gen_zero) for _ in range(5)]
    one_values: list[float] = [next(gen_one) for _ in range(5)]

    assert zero_values == pytest.approx(one_values, rel=0, abs=0)


def test_xorshift_32_float_generator_values_stay_in_unit_interval():
    gen = xorshift_32_float_generator(seed=1)

    values = list(itertools.islice(gen, 1_000))

    assert all(0.0 <= value < 1.0 for value in values)


def test_xorshift_32_generator_values_are_floats():
    gen = xorshift_32_float_generator(seed=1)
    value = next(gen)

    assert isinstance(value, float)


def test_xorshift_32_float_generator_is_deterministic_for_same_seed():
    seed: int|float = 7

    gen_a = xorshift_32_float_generator(seed=seed)
    gen_b = xorshift_32_float_generator(seed=seed)

    values_a: list[float] = [next(gen_a) for _ in range(10)]
    values_b: list[float] = [next(gen_b) for _ in range(10)]

    assert values_a == pytest.approx(values_b, rel=0, abs=0)

@pytest.mark.parametrize(
    "seed,r,expected",
    [
        (
            0.5,
            3.99,
            [
                0.5,
                0.9975,
                0.009950062499999789,
                0.03930572443742109,
                0.15066553001084376,
            ],
        ),
        (
            0.1,
            3.99,
            [
                0.1,
                0.35910000000000003,
                0.9182872881000002,
                0.2993926210096504,
                0.8369291511835427,
            ],
        ),
        (
            0.25,
            3.99,
            [
                0.25,
                0.748125,
                0.75185159765625,
                0.7444173833043976,
                0.7591379695271385,
            ],
        ),
    ],
)
def test_logistic_map_generator_produces_expected_sequence(seed, r, expected):
    gen = logistic_map_pseudorandom_generator(seed=seed, r=r)
    result = [next(gen) for _ in range(len(expected))]

    assert result == pytest.approx(expected, rel=0, abs=0)


def test_logistic_map_generator_seed_zero_does_not_stay_zero():
    gen = logistic_map_pseudorandom_generator(seed=0.0, r=3.99)
    result = [next(gen) for _ in range(5)]

    assert result != pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0], rel=0, abs=0)


def test_logistic_map_generator_is_deterministic_for_same_seed():
    seed = 0.7
    r = 3.99

    gen_a = logistic_map_pseudorandom_generator(seed=seed, r=r)
    gen_b = logistic_map_pseudorandom_generator(seed=seed, r=r)

    values_a = list(itertools.islice(gen_a, 10))
    values_b = list(itertools.islice(gen_b, 10))

    assert values_a == pytest.approx(values_b, rel=0, abs=0)


def test_logistic_map_generator_values_are_floats():
    gen = logistic_map_pseudorandom_generator(seed=0.5, r=3.99)
    value = next(gen)

    assert isinstance(value, float)


def test_logistic_map_generator_zero_seed_defaults_to_half():
    gen = logistic_map_pseudorandom_generator(seed=0, r=3.99)

    result = [next(gen) for _ in range(5)]

    assert result == pytest.approx(
        [
            0.5,
            0.9975,
            0.009950062499999789,
            0.03930572443742109,
            0.15066553001084376,
        ],
        rel=0,
        abs=0,
    )


def test_logistic_map_float_generator_values_stay_in_unit_interval():
    gen = logistic_map_pseudorandom_generator(seed=0.5)

    values = list(itertools.islice(gen, 1_000))

    assert all(0.0 <= value < 1.0 for value in values)


@pytest.mark.parametrize(
    "seed_1, seed_2, seed_3, expected",
    [
        (
            1,
            1,
            1,
            [
                0.01693090619965683,
                0.8952539112379991,
                0.11149102121645216,
                0.9395267964111933,
                0.12822985510067042,
            ],
        ),
        (
            7,
            11,
            19,
            [
                0.20849303489971333,
                0.608216986356779,
                0.6343234569053853,
                0.9116761657982964,
                0.8280497502746622
            ],
        ),
    ],
)
def test_wichmann_hill_generator_produces_expected_sequence(
    seed_1, seed_2, seed_3, expected
):
    gen = wichmann_hill_generator(seed_1=seed_1, seed_2=seed_2, seed_3=seed_3)
    result = [next(gen) for _ in range(len(expected))]

    assert result == pytest.approx(expected, rel=0, abs=0)


def test_wichmann_hill_generator_zero_seeds_behave_like_ones():
    gen_zero = wichmann_hill_generator(seed_1=0, seed_2=0, seed_3=0)
    gen_one = wichmann_hill_generator(seed_1=1, seed_2=1, seed_3=1)

    zero_values = [next(gen_zero) for _ in range(5)]
    one_values = [next(gen_one) for _ in range(5)]

    assert zero_values == pytest.approx(one_values, rel=0, abs=0)


def test_wichmann_hill_generator_is_deterministic_for_same_seed():
    seed_1, seed_2, seed_3 = 7, 30, 50

    gen_a = wichmann_hill_generator(seed_1=seed_1, seed_2=seed_2, seed_3=seed_3)
    gen_b = wichmann_hill_generator(seed_1=seed_1, seed_2=seed_2, seed_3=seed_3)

    values_a = list(itertools.islice(gen_a, 10))
    values_b = list(itertools.islice(gen_b, 10))

    assert values_a == pytest.approx(values_b, rel=0, abs=0)


def test_wichmann_hill_generator_outputs_are_in_unit_interval():
    gen = wichmann_hill_generator(seed_1=1, seed_2=1, seed_3=1)

    values = list(itertools.islice(gen, 1000))

    assert all(0.0 <= value < 1.0 for value in values)


def test_wichmann_hill_generator_values_are_floats():
    gen = wichmann_hill_generator(seed_1=1, seed_2=1, seed_3=1)
    value = next(gen)

    assert isinstance(value, float)

