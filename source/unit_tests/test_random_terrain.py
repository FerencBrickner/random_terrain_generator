import numpy as np
import pytest
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).resolve().parents[1]))

from random_terrain import normalize_terrain_to_the_0_to_1_range_safely
from random_terrain import apply_smootherstep_polynomial_perlin_noise
from random_terrain import compute_bilinear_interpolation_on_a_2D_control_grid
from random_terrain import apply_continental_falloff
from random_terrain import convolve_rows_for_gaussian_blur
from random_terrain import convolve_columns_for_gaussian_blur
from random_terrain import compute_1d_kernel_for_gaussian_blur
from random_terrain import compute_cutoff_radius_of_gaussian_blur


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


def test_normalizes_2d_array_for_all_zero_elements():
    terrain = np.array(
        [
            [0, 0],
            [0, 0],
        ]
    )

    result = normalize_terrain_to_the_0_to_1_range_safely(terrain=terrain)

    expected = np.array(
        [
            [0, 0],
            [0, 0],
        ]
    )
    assert np.allclose(result, expected)


def test_normalization_of_constant_terrain_returns_zeros():
    terrain = np.array([7.0, 7.0, 7.0])

    result = normalize_terrain_to_the_0_to_1_range_safely(terrain=terrain)

    expected = np.array([0.0, 0.0, 0.0])
    assert np.array_equal(result, expected)


def test_normalization_handles_negative_values_correctly():
    terrain = np.array([-5.0, 0.0, 5.0])

    result = normalize_terrain_to_the_0_to_1_range_safely(terrain=terrain)

    expected = np.array([0.0, 0.5, 1.0])
    assert np.allclose(result, expected)


def test_normalization_single_value_array_returns_zero():
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


def _make_quality_grid(size: int) -> np.ndarray:
    directions = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [0.7071067811865476, 0.7071067811865476],
            [-0.7071067811865476, 0.7071067811865476],
            [-0.7071067811865476, -0.7071067811865476],
            [0.7071067811865476, -0.7071067811865476],
        ],
        dtype=float,
    )

    grid = np.empty((size, size, 2), dtype=float)
    for y in range(size):
        for x in range(size):
            grid[y, x] = directions[(x + 3 * y) % len(directions)]
    return grid


def test_compute_bilinear_interpolation_small_grid():
    grid = _make_quality_grid(2)

    result = compute_bilinear_interpolation_on_a_2D_control_grid(
        grid=grid, grid_x=0.25, grid_y=0.25
    )

    assert result == pytest.approx(0.2823557112788028, rel=1e-12, abs=1e-12)


def test_compute_bilinear_interpolation_medium_grid():
    grid = _make_quality_grid(4)

    result = compute_bilinear_interpolation_on_a_2D_control_grid(
        grid=grid, grid_x=1.25, grid_y=1.75
    )

    assert result == pytest.approx(0.2915302771455237, rel=1e-12, abs=1e-12)


def test_compute_bilinear_interpolation_larger_grid():
    grid = _make_quality_grid(8)

    result = compute_bilinear_interpolation_on_a_2D_control_grid(
        grid=grid, grid_x=5.25, grid_y=5.25
    )

    assert result == pytest.approx(0.4073478897683403, rel=1e-12, abs=1e-12)


def test_compute_bilinear_interpolation_return_type():
    grid = _make_quality_grid(4)

    result = compute_bilinear_interpolation_on_a_2D_control_grid(
        grid=grid, grid_x=1.25, grid_y=1.75
    )

    assert isinstance(result, float)


def test_compute_bilinear_interpolation_does_not_raise_exception():
    grid = _make_quality_grid(4)

    try:
        _ = compute_bilinear_interpolation_on_a_2D_control_grid(
            grid=grid, grid_x=1.5, grid_y=1.5
        )
    except Exception as exc:
        pytest.fail(f"Function raised an exception unexpectedly: {exc}")


def test_compute_bilinear_interpolation_is_fast_enough():
    grid = _make_quality_grid(64)

    start = time.perf_counter()

    for i in range(10_000):
        x = 1.25 + (i % 20) * 0.05
        y = 1.75 + (i % 20) * 0.05
        _ = compute_bilinear_interpolation_on_a_2D_control_grid(
            grid=grid, grid_x=x, grid_y=y
        )
    duration = time.perf_counter() - start

    assert duration < 1.5


def test_performance_of_continental_falloff():
    heightmap = np.ones((1000, 1000))

    start = time.time()
    apply_continental_falloff(heightmap, 2)
    duration = time.time() - start

    assert duration < 0.5


def test_returntype_of_continental_falloff():
    heightmap = np.ones((1000, 1000))

    returnvalue = apply_continental_falloff(heightmap, 2)

    assert isinstance(returnvalue, np.ndarray)


def test_returnvalue_of_continental_falloff():
    heightmap = heightmap = np.array(
        [
            [0.582, 0.143, 0.764, 0.921, 0.334, 0.615, 0.278, 0.489, 0.712, 0.051],
            [0.843, 0.227, 0.391, 0.675, 0.184, 0.952, 0.438, 0.126, 0.593, 0.807],
            [0.319, 0.744, 0.086, 0.558, 0.991, 0.402, 0.673, 0.215, 0.347, 0.768],
            [0.694, 0.531, 0.248, 0.879, 0.117, 0.463, 0.725, 0.394, 0.682, 0.159],
            [0.905, 0.276, 0.618, 0.341, 0.792, 0.054, 0.487, 0.663, 0.128, 0.570],
            [0.451, 0.822, 0.193, 0.736, 0.299, 0.641, 0.875, 0.368, 0.514, 0.247],
            [0.166, 0.589, 0.931, 0.074, 0.426, 0.758, 0.203, 0.847, 0.492, 0.635],
            [0.713, 0.358, 0.547, 0.189, 0.864, 0.421, 0.096, 0.672, 0.305, 0.944],
            [0.238, 0.697, 0.412, 0.583, 0.154, 0.826, 0.369, 0.741, 0.027, 0.658],
            [0.771, 0.104, 0.495, 0.887, 0.316, 0.552, 0.229, 0.603, 0.918, 0.140],
        ]
    )

    returnvalue = apply_continental_falloff(heightmap, 1.5)

    expected_returnvalue = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [
                0.0,
                0.0,
                0.00363166,
                0.04071467,
                0.018257,
                0.09446015,
                0.0264193,
                0.0011703,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.00691037,
                0.00853316,
                0.11659064,
                0.28279424,
                0.11471573,
                0.14061918,
                0.02133291,
                0.00322298,
                0.0,
            ],
            [
                0.0,
                0.03202887,
                0.05181806,
                0.3378112,
                0.06112055,
                0.24187021,
                0.27862698,
                0.08232386,
                0.0411369,
                0.0,
            ],
            [
                0.0,
                0.02738551,
                0.17635403,
                0.17813767,
                0.61286149,
                0.04178601,
                0.25440776,
                0.18919534,
                0.01270052,
                0.0,
            ],
            [
                0.0,
                0.08156118,
                0.05507496,
                0.38448482,
                0.23137069,
                0.49601543,
                0.45709813,
                0.1050134,
                0.05100054,
                0.0,
            ],
            [
                0.0,
                0.03552732,
                0.19452668,
                0.02843917,
                0.22254149,
                0.39597758,
                0.07801556,
                0.1769754,
                0.02967647,
                0.0,
            ],
            [
                0.0,
                0.00332515,
                0.0542749,
                0.03949038,
                0.2465532,
                0.12013761,
                0.0200586,
                0.06667775,
                0.00283288,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.00382671,
                0.03516541,
                0.01528032,
                0.08195807,
                0.02225735,
                0.0068825,
                0.0,
                0.0,
            ],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    assert returnvalue == pytest.approx(expected_returnvalue, rel=1e-5, abs=1e-5)


def test_continental_falloff_with_zero_continent_effect_strength_does_not_modify_heightmap():
    heightmap = heightmap = np.array(
        [
            [0.582, 0.143, 0.764, 0.921, 0.334, 0.615, 0.278, 0.489, 0.712, 0.051],
            [0.843, 0.227, 0.391, 0.675, 0.184, 0.952, 0.438, 0.126, 0.593, 0.807],
            [0.319, 0.744, 0.086, 0.558, 0.991, 0.402, 0.673, 0.215, 0.347, 0.768],
            [0.694, 0.531, 0.248, 0.879, 0.117, 0.463, 0.725, 0.394, 0.682, 0.159],
            [0.905, 0.276, 0.618, 0.341, 0.792, 0.054, 0.487, 0.663, 0.128, 0.570],
            [0.451, 0.822, 0.193, 0.736, 0.299, 0.641, 0.875, 0.368, 0.514, 0.247],
            [0.166, 0.589, 0.931, 0.074, 0.426, 0.758, 0.203, 0.847, 0.492, 0.635],
            [0.713, 0.358, 0.547, 0.189, 0.864, 0.421, 0.096, 0.672, 0.305, 0.944],
            [0.238, 0.697, 0.412, 0.583, 0.154, 0.826, 0.369, 0.741, 0.027, 0.658],
            [0.771, 0.104, 0.495, 0.887, 0.316, 0.552, 0.229, 0.603, 0.918, 0.140],
        ]
    )

    returnvalue = apply_continental_falloff(heightmap, 0)

    expected_returnvalue = np.array(
        [
            [0.582, 0.143, 0.764, 0.921, 0.334, 0.615, 0.278, 0.489, 0.712, 0.051],
            [0.843, 0.227, 0.391, 0.675, 0.184, 0.952, 0.438, 0.126, 0.593, 0.807],
            [0.319, 0.744, 0.086, 0.558, 0.991, 0.402, 0.673, 0.215, 0.347, 0.768],
            [0.694, 0.531, 0.248, 0.879, 0.117, 0.463, 0.725, 0.394, 0.682, 0.159],
            [0.905, 0.276, 0.618, 0.341, 0.792, 0.054, 0.487, 0.663, 0.128, 0.570],
            [0.451, 0.822, 0.193, 0.736, 0.299, 0.641, 0.875, 0.368, 0.514, 0.247],
            [0.166, 0.589, 0.931, 0.074, 0.426, 0.758, 0.203, 0.847, 0.492, 0.635],
            [0.713, 0.358, 0.547, 0.189, 0.864, 0.421, 0.096, 0.672, 0.305, 0.944],
            [0.238, 0.697, 0.412, 0.583, 0.154, 0.826, 0.369, 0.741, 0.027, 0.658],
            [0.771, 0.104, 0.495, 0.887, 0.316, 0.552, 0.229, 0.603, 0.918, 0.140],
        ]
    )

    assert returnvalue == pytest.approx(expected_returnvalue, rel=0, abs=0)


def test_continental_falloff_with_strong_continent_effect_nulls_the_heightmap():
    heightmap = heightmap = np.array(
        [
            [0.582, 0.143, 0.764, 0.921, 0.334, 0.615, 0.278, 0.489, 0.712, 0.051],
            [0.843, 0.227, 0.391, 0.675, 0.184, 0.952, 0.438, 0.126, 0.593, 0.807],
            [0.319, 0.744, 0.086, 0.558, 0.991, 0.402, 0.673, 0.215, 0.347, 0.768],
            [0.694, 0.531, 0.248, 0.879, 0.117, 0.463, 0.725, 0.394, 0.682, 0.159],
            [0.905, 0.276, 0.618, 0.341, 0.792, 0.054, 0.487, 0.663, 0.128, 0.570],
            [0.451, 0.822, 0.193, 0.736, 0.299, 0.641, 0.875, 0.368, 0.514, 0.247],
            [0.166, 0.589, 0.931, 0.074, 0.426, 0.758, 0.203, 0.847, 0.492, 0.635],
            [0.713, 0.358, 0.547, 0.189, 0.864, 0.421, 0.096, 0.672, 0.305, 0.944],
            [0.238, 0.697, 0.412, 0.583, 0.154, 0.826, 0.369, 0.741, 0.027, 0.658],
            [0.771, 0.104, 0.495, 0.887, 0.316, 0.552, 0.229, 0.603, 0.918, 0.140],
        ]
    )

    returnvalue = apply_continental_falloff(heightmap, 30)

    expected_returnvalue = np.zeros((10, 10), dtype=float)

    assert returnvalue == pytest.approx(expected_returnvalue, rel=1e-2, abs=1e-2)


def test_continental_falloff_with_very_strong_continent_effect_nulls_the_heightmap():
    heightmap = heightmap = np.array(
        [
            [0.582, 0.143, 0.764, 0.921, 0.334, 0.615, 0.278, 0.489, 0.712, 0.051],
            [0.843, 0.227, 0.391, 0.675, 0.184, 0.952, 0.438, 0.126, 0.593, 0.807],
            [0.319, 0.744, 0.086, 0.558, 0.991, 0.402, 0.673, 0.215, 0.347, 0.768],
            [0.694, 0.531, 0.248, 0.879, 0.117, 0.463, 0.725, 0.394, 0.682, 0.159],
            [0.905, 0.276, 0.618, 0.341, 0.792, 0.054, 0.487, 0.663, 0.128, 0.570],
            [0.451, 0.822, 0.193, 0.736, 0.299, 0.641, 0.875, 0.368, 0.514, 0.247],
            [0.166, 0.589, 0.931, 0.074, 0.426, 0.758, 0.203, 0.847, 0.492, 0.635],
            [0.713, 0.358, 0.547, 0.189, 0.864, 0.421, 0.096, 0.672, 0.305, 0.944],
            [0.238, 0.697, 0.412, 0.583, 0.154, 0.826, 0.369, 0.741, 0.027, 0.658],
            [0.771, 0.104, 0.495, 0.887, 0.316, 0.552, 0.229, 0.603, 0.918, 0.140],
        ]
    )

    returnvalue = apply_continental_falloff(heightmap, 100)

    expected_returnvalue = np.zeros((10, 10), dtype=float)

    assert returnvalue == pytest.approx(expected_returnvalue, rel=1e-2, abs=1e-2)


def test_continental_falloff_with_weak_continent_effect_does_not_null_the_heightmap():
    heightmap = heightmap = np.array(
        [
            [0.582, 0.143, 0.764, 0.921, 0.334, 0.615, 0.278, 0.489, 0.712, 0.051],
            [0.843, 0.227, 0.391, 0.675, 0.184, 0.952, 0.438, 0.126, 0.593, 0.807],
            [0.319, 0.744, 0.086, 0.558, 0.991, 0.402, 0.673, 0.215, 0.347, 0.768],
            [0.694, 0.531, 0.248, 0.879, 0.117, 0.463, 0.725, 0.394, 0.682, 0.159],
            [0.905, 0.276, 0.618, 0.341, 0.792, 0.054, 0.487, 0.663, 0.128, 0.570],
            [0.451, 0.822, 0.193, 0.736, 0.299, 0.641, 0.875, 0.368, 0.514, 0.247],
            [0.166, 0.589, 0.931, 0.074, 0.426, 0.758, 0.203, 0.847, 0.492, 0.635],
            [0.713, 0.358, 0.547, 0.189, 0.864, 0.421, 0.096, 0.672, 0.305, 0.944],
            [0.238, 0.697, 0.412, 0.583, 0.154, 0.826, 0.369, 0.741, 0.027, 0.658],
            [0.771, 0.104, 0.495, 0.887, 0.316, 0.552, 0.229, 0.603, 0.918, 0.140],
        ]
    )

    returnvalue = apply_continental_falloff(heightmap, 0.3)

    expected_returnvalue = np.zeros((10, 10), dtype=float)

    assert returnvalue != pytest.approx(expected_returnvalue, rel=1e-2, abs=1e-2)


def test_continental_falloff_deterministic():
    heightmap = heightmap = np.array(
        [
            [0.582, 0.143, 0.764, 0.921, 0.334, 0.615, 0.278, 0.489, 0.712, 0.051],
            [0.843, 0.227, 0.391, 0.675, 0.184, 0.952, 0.438, 0.126, 0.593, 0.807],
            [0.319, 0.744, 0.086, 0.558, 0.991, 0.402, 0.673, 0.215, 0.347, 0.768],
            [0.694, 0.531, 0.248, 0.879, 0.117, 0.463, 0.725, 0.394, 0.682, 0.159],
            [0.905, 0.276, 0.618, 0.341, 0.792, 0.054, 0.487, 0.663, 0.128, 0.570],
            [0.451, 0.822, 0.193, 0.736, 0.299, 0.641, 0.875, 0.368, 0.514, 0.247],
            [0.166, 0.589, 0.931, 0.074, 0.426, 0.758, 0.203, 0.847, 0.492, 0.635],
            [0.713, 0.358, 0.547, 0.189, 0.864, 0.421, 0.096, 0.672, 0.305, 0.944],
            [0.238, 0.697, 0.412, 0.583, 0.154, 0.826, 0.369, 0.741, 0.027, 0.658],
            [0.771, 0.104, 0.495, 0.887, 0.316, 0.552, 0.229, 0.603, 0.918, 0.140],
        ]
    )

    returnvalue_1 = apply_continental_falloff(heightmap, 30)
    returnvalue_2 = apply_continental_falloff(heightmap, 30)

    assert returnvalue_1 == pytest.approx(returnvalue_2, rel=1e-2, abs=1e-2)


@pytest.fixture
def control_grid() -> np.ndarray:
    return np.array(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [[-1.0, 0.5], [0.5, -1.0], [2.0, 0.0]],
            [[0.25, -0.75], [1.5, 0.25], [-0.5, 1.25]],
        ],
        dtype=np.float64,
    )


def test_return_type_of_compute_bilinear_interpolation_on_a_2D_control_grid(control_grid: np.ndarray):
    value = compute_bilinear_interpolation_on_a_2D_control_grid(
        control_grid, 1.25, 0.75
    )
    assert isinstance(value, (float, np.floating))


def test_zero_at_integer_grid_point_of_compute_bilinear_interpolation_on_a_2D_control_grid(control_grid: np.ndarray):
    value = compute_bilinear_interpolation_on_a_2D_control_grid(
        control_grid, 1.0, 1.0
    )
    assert value == pytest.approx(0.0)


def test_deterministic_nature_of_compute_bilinear_interpolation_on_a_2D_control_grid(control_grid: np.ndarray):
    x, y = 1.25, 0.75
    v1 = compute_bilinear_interpolation_on_a_2D_control_grid(control_grid, x, y)
    v2 = compute_bilinear_interpolation_on_a_2D_control_grid(control_grid, x, y)
    assert v1 == pytest.approx(v2)


def test_known_value_of_compute_bilinear_interpolation_on_a_2D_control_grid(control_grid: np.ndarray):
    x, y = 1.25, 0.75

    expected: float =  0.2317814826965332

    value = compute_bilinear_interpolation_on_a_2D_control_grid(control_grid, x, y)
    assert value == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_speed_of_compute_bilinear_interpolation_on_a_2D_control_grid(control_grid: np.ndarray):
    x, y = 1.25, 0.75

    start = time.perf_counter()
    for _ in range(20_000):
        compute_bilinear_interpolation_on_a_2D_control_grid(control_grid, x, y)
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0


def test_convolve_rows_for_gaussian_blur_returntype():
    padded_array = np.array(
        [
            [0, 1, 2, 3, 0],
            [4, 5, 6, 7, 0],
            [8, 9, 10, 11, 0],
        ],
        dtype=np.float64,
    )
    kernel_1d = np.array([1, 2, 1], dtype=np.float64) / 4
    map_array = np.zeros((3, 3), dtype=np.float64)

    result = convolve_rows_for_gaussian_blur(
        padded_array=padded_array,
        kernel_1d=kernel_1d,
        map_array=map_array,
    )

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64


def test_convolve_rows_for_gaussian_blur_returnvalue():
    padded_array = np.array(
        [
            [0, 1, 2, 3, 0],
            [4, 5, 6, 7, 0],
            [8, 9, 10, 11, 0],
        ],
        dtype=np.float64,
    )
    kernel_1d = np.array([1, 2, 1], dtype=np.float64) / 4
    map_array = np.zeros((3, 3), dtype=np.float64)

    result = convolve_rows_for_gaussian_blur(
        padded_array=padded_array,
        kernel_1d=kernel_1d,
        map_array=map_array,
    )

    expected = np.array(
        [
            [1.0, 2.0, 2.0],
            [5.0, 6.0, 5.0],
            [9.0, 10.0, 8.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(result, expected)


def test_convolve_rows_for_gaussian_blur_deterministic():
    padded_array = np.array(
        [
            [0, 1, 2, 3, 0],
            [4, 5, 6, 7, 0],
            [8, 9, 10, 11, 0],
        ],
        dtype=np.float64,
    )
    kernel_1d = np.array([1, 2, 1], dtype=np.float64) / 4
    map_array = np.zeros((3, 3), dtype=np.float64)

    result_1 = convolve_rows_for_gaussian_blur(
        padded_array=padded_array,
        kernel_1d=kernel_1d,
        map_array=map_array,
    )

    result_2 = convolve_rows_for_gaussian_blur(
        padded_array=padded_array,
        kernel_1d=kernel_1d,
        map_array=map_array,
    )

    np.testing.assert_allclose(result_1, result_2)


def test_convolve_rows_for_gaussian_blur_does_not_raise_exception():
    padded_array = np.array(
        [
            [0, 1, 2, 3, 0],
            [4, 5, 6, 7, 0],
            [8, 9, 10, 11, 0],
        ],
        dtype=np.float64,
    )
    kernel_1d = np.array([1, 2, 1], dtype=np.float64) / 4
    map_array = np.zeros((3, 3), dtype=np.float64)

    try:
        convolve_rows_for_gaussian_blur(
            padded_array=padded_array,
            kernel_1d=kernel_1d,
            map_array=map_array,
        )
    except Exception as exc:
        pytest.fail(f"convolve_rows_for_gaussian_blur raised {exc!r}")


def test_convolve_rows_for_gaussian_blur_speed():
    padded_array = np.tile(np.arange(512, dtype=np.float64), (256, 1))
    kernel_1d = np.array([1, 2, 1], dtype=np.float64) / 4
    map_array = np.zeros((256, 510), dtype=np.float64)

    start = time.perf_counter()
    result = convolve_rows_for_gaussian_blur(
        padded_array=padded_array,
        kernel_1d=kernel_1d,
        map_array=map_array,
    )
    elapsed = time.perf_counter() - start

    assert result.shape == (256, 510)
    assert elapsed < 1.0


def test_convolve_columns_for_gaussian_blur_returntype():
    result_rows = np.array(
        [
            [1.0, 2.0, 2.0],
            [5.0, 6.0, 5.0],
            [9.0, 10.0, 8.0],
        ],
        dtype=np.float64,
    )
    kernel_1d = np.array([1, 2, 1], dtype=np.float64) / 4
    map_array = np.zeros((1, 3), dtype=np.float64)

    result = convolve_columns_for_gaussian_blur(
        result_rows=result_rows,
        kernel_1d=kernel_1d,
        map_array=map_array,
    )

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64


def test_convolve_columns_for_gaussian_blur_returnvalue():
    result_rows = np.array(
        [
            [1.0, 2.0, 2.0],
            [5.0, 6.0, 5.0],
            [9.0, 10.0, 8.0],
        ],
        dtype=np.float64,
    )
    kernel_1d = np.array([1, 2, 1], dtype=np.float64) / 4
    map_array = np.zeros((1, 3), dtype=np.float64)

    result = convolve_columns_for_gaussian_blur(
        result_rows=result_rows,
        kernel_1d=kernel_1d,
        map_array=map_array,
    )

    expected = np.array([[5.0, 6.0, 5.0]], dtype=np.float64)
    np.testing.assert_allclose(result, expected)


def test_convolve_columns_for_gaussian_blur_deterministic():
    result_rows = np.array(
        [
            [1.0, 2.0, 2.0],
            [5.0, 6.0, 5.0],
            [9.0, 10.0, 8.0],
        ],
        dtype=np.float64,
    )
    kernel_1d = np.array([1, 2, 1], dtype=np.float64) / 4
    map_array = np.zeros((1, 3), dtype=np.float64)

    result_1 = convolve_columns_for_gaussian_blur(
        result_rows=result_rows,
        kernel_1d=kernel_1d,
        map_array=map_array,
    )

    result_2 = convolve_columns_for_gaussian_blur(
        result_rows=result_rows,
        kernel_1d=kernel_1d,
        map_array=map_array,
    )

    np.testing.assert_allclose(result_1, result_2)


def test_convolve_columns_for_gaussian_blur_does_not_raise_exception():
    result_rows = np.array(
        [
            [1.0, 2.0, 2.0],
            [5.0, 6.0, 5.0],
            [9.0, 10.0, 8.0],
        ],
        dtype=np.float64,
    )
    kernel_1d = np.array([1, 2, 1], dtype=np.float64) / 4
    map_array = np.zeros((1, 3), dtype=np.float64)

    try:
        convolve_columns_for_gaussian_blur(
            result_rows=result_rows,
            kernel_1d=kernel_1d,
            map_array=map_array,
        )
    except Exception as exc:
        pytest.fail(f"convolve_columns_for_gaussian_blur raised {exc!r}")


def test_convolve_columns_for_gaussian_blur_speed():
    result_rows = np.tile(np.arange(256, dtype=np.float64).reshape(-1, 1), (1, 256))
    kernel_1d = np.array([1, 2, 1], dtype=np.float64) / 4
    map_array = np.zeros((254, 256), dtype=np.float64)

    start = time.perf_counter()
    result = convolve_columns_for_gaussian_blur(
        result_rows=result_rows,
        kernel_1d=kernel_1d,
        map_array=map_array,
    )
    elapsed = time.perf_counter() - start

    assert result.shape == (254, 256)
    assert elapsed < 1.0


def test_compute_1d_kernel_for_gaussian_blur_returntype():
    result = compute_1d_kernel_for_gaussian_blur(sigma=1.0, radius=2)

    assert isinstance(result, np.ndarray)


def test_compute_1d_kernel_for_gaussian_blur_returntype_dtype():
    result = compute_1d_kernel_for_gaussian_blur(sigma=1.0, radius=2)

    assert result.dtype == np.float64


def test_compute_1d_kernel_for_gaussian_blur_returnshape():
    result = compute_1d_kernel_for_gaussian_blur(sigma=1.0, radius=2)

    assert result.shape == (5,)


def test_compute_1d_kernel_for_gaussian_blur_returnvalue():
    sigma = 1.0
    radius = 2

    result = compute_1d_kernel_for_gaussian_blur(sigma=sigma, radius=radius)

    x_axis = np.arange(-radius, radius + 1, dtype=np.float64)
    expected = np.exp(-0.5 * (x_axis / sigma) ** 2)
    expected /= float(expected.sum())

    np.testing.assert_allclose(result, expected)


def test_compute_1d_kernel_for_gaussian_blur_deterministic_nature():
    result_1 = compute_1d_kernel_for_gaussian_blur(sigma=1.0, radius=2)
    result_2 = compute_1d_kernel_for_gaussian_blur(sigma=1.0, radius=2)

    np.testing.assert_allclose(result_1, result_2)


def test_compute_1d_kernel_for_gaussian_blur_does_not_raise_exception():
    try:
        compute_1d_kernel_for_gaussian_blur(sigma=1.0, radius=2)
    except Exception as exc:
        pytest.fail(
            f"compute_1d_kernel_for_gaussian_blur raised an exception: {exc!r}"
        )


def test_compute_1d_kernel_for_gaussian_blur_speed():
    start = time.perf_counter()

    for _ in range(10_000):
        compute_1d_kernel_for_gaussian_blur(sigma=1.0, radius=2)

    elapsed = time.perf_counter() - start

    assert elapsed < 1.0


def test_compute_cutoff_radius_of_gaussian_blur_returntype():
    result = compute_cutoff_radius_of_gaussian_blur(sigma=1.0)

    assert isinstance(result, int)


def test_compute_cutoff_radius_of_gaussian_blur_returnvalue():
    result = compute_cutoff_radius_of_gaussian_blur(sigma=1.0)

    assert result == 3


def test_compute_cutoff_radius_of_gaussian_blur_deterministic_nature():
    result_1 = compute_cutoff_radius_of_gaussian_blur(sigma=1.0)
    result_2 = compute_cutoff_radius_of_gaussian_blur(sigma=1.0)

    assert result_1 == result_2


def test_compute_cutoff_radius_of_gaussian_blur_does_not_raise_exception():
    try:
        compute_cutoff_radius_of_gaussian_blur(sigma=1.0)
    except Exception as exc:
        pytest.fail(
            f"compute_cutoff_radius_of_gaussian_blur raised an exception: {exc!r}"
        )


def test_compute_cutoff_radius_of_gaussian_blur_speed():
    start = time.perf_counter()

    for _ in range(100_000):
        compute_cutoff_radius_of_gaussian_blur(sigma=1.0)

    elapsed = time.perf_counter() - start

    assert elapsed < 1.0