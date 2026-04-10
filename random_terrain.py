from typing import Generator, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import yaml
from PRNGs.xorshift32 import xorshift_32_float_generator
from PRNGs.wichmann_hill import wichmann_hill_generator
from PRNGs.logistic_map import logistic_map_pseudorandom_generator


class GaussianSigmaShouldBePositive(Exception):
    """Gaussian sigma should be positive."""


class GaussianPassCountShouldBePositive(Exception):
    """Gaussian pass count should be positive."""


class TerrainOctaveCountShouldBeAtLeastOne(Exception):
    """Terrain octave count should be at least one."""


class TerrainPersistenceFactorShouldBePositive(Exception):
    """Terrain persistence factor should be positive."""


class TerrainInitialScaleShouldBePositive(Exception):
    """Terrain initial scale should be positive."""


def load_configuration_yaml(*, config_path: str) -> dict:
    """
    Idea: https://pyyaml.org/wiki/PyYAMLDocumentation

    Load configuration YAML.
    """
    logging.info("Loading configuration YAML...")

    with open(config_path, "r") as text_io_wrapper:
        configuration: Any = yaml.safe_load(text_io_wrapper)

    gaussian_blur = configuration["gaussian_blur"]
    terrain_generation = configuration["terrain_generation"]
    prng = configuration["prng"]

    gaussian_sigma = float(gaussian_blur["sigma"])

    if gaussian_sigma <= 0:
        raise GaussianSigmaShouldBePositive
    
    gaussian_pass_count = int(gaussian_blur["passes"])

    if gaussian_pass_count <= 0:
        raise GaussianPassCountShouldBePositive
    
    terrain_octave_count = int(terrain_generation["octaves"])

    if terrain_octave_count < 1:
        raise TerrainOctaveCountShouldBeAtLeastOne
    
    terrain_persistence_factor = float(terrain_generation["persistence"])

    if terrain_persistence_factor <= 0:
        raise TerrainPersistenceFactorShouldBePositive
    
    terrain_initial_scale = float(terrain_generation["initial_scale"])

    prng_type = prng["prng_type"]

    if terrain_initial_scale <= 0:
        raise TerrainInitialScaleShouldBePositive
    
    return {
        "gaussian_sigma": gaussian_sigma,
        "gaussian_pass_count": gaussian_pass_count,
        "terrain_octave_count": terrain_octave_count,
        "terrain_persistence_factor": terrain_persistence_factor,
        "terrain_initial_scale": terrain_initial_scale,
        "prng_type": prng_type,
    }


def configure_logging(*, level: int = logging.INFO) -> None:
    """
    Idea: https://docs.python.org/3/library/logging.html

    Configure the root logger with a standardized format.

    This setup includes timestamps, log level, logger name,
    and the message.

    Args:
        level: The minimum logging level (default: logging.INFO).
    """

    logging.basicConfig(
        level=level,
        format=("%(asctime)s | " "%(levelname)-8s | " "%(message)s"),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Ensure timestamps use local time

    logging.Formatter.converter = time.localtime


def convolve_rows_for_gaussian_blur(
    *, padded_array: np.ndarray, kernel_1d: np.ndarray, map_array: np.ndarray
) -> np.ndarray:
    """
    Idea: https://people.csail.mit.edu/sparis/bf_course/slides/02_gaussian_blur.pdf
    Idea: https://stackoverflow.com/questions/17841098/gaussian-blur-standard-deviation-radius-and-kernel-size
    """
    logging.info("Convolving rows for Gaussian blur...")
    result_rows: np.ndarray = np.empty(
        (padded_array.shape[0], map_array.shape[1]), dtype=np.float64
    )

    for i in range(padded_array.shape[0]):
        result_rows[i, :] = np.convolve(padded_array[i, :], kernel_1d, mode="valid")

    return result_rows


def convolve_columns_for_gaussian_blur(
    *, result_rows: np.ndarray, kernel_1d: np.ndarray, map_array: np.ndarray
) -> np.ndarray:
    """
    Idea: https://people.csail.mit.edu/sparis/bf_course/slides/02_gaussian_blur.pdf
    Idea: https://stackoverflow.com/questions/17841098/gaussian-blur-standard-deviation-radius-and-kernel-size
    """
    logging.info("Convolving columns for Gaussian blur...")

    result: np.ndarray = np.empty_like(map_array, dtype=np.float64)

    for j in range(result_rows.shape[1]):
        result[:, j] = np.convolve(result_rows[:, j], kernel_1d, mode="valid")

    return result


def compute_1d_kernel_for_gaussian_blur(*, sigma: float, radius: int) -> np.ndarray:
    """
    Idea: https://people.csail.mit.edu/sparis/bf_course/slides/02_gaussian_blur.pdf
    Idea: https://stackoverflow.com/questions/17841098/gaussian-blur-standard-deviation-radius-and-kernel-size
    """
    logging.info("Computing 1d kernel for Gaussian blur...")
    x_axis: np.ndarray = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel_1d: np.ndarray = np.exp(-0.5 * (x_axis / sigma) ** 2)
    kernel_1d /= float(kernel_1d.sum())
    return kernel_1d


def compute_cutoff_radius_of_gaussian_blur(*, sigma: float) -> int:
    """
    Idea: https://people.csail.mit.edu/sparis/bf_course/slides/02_gaussian_blur.pdf
    Idea: https://stackoverflow.com/questions/17841098/gaussian-blur-standard-deviation-radius-and-kernel-size
    """
    logging.info("Computing cutoff radius of Gaussian blur...")
    return max(1, int(np.ceil(3.0 * sigma)))


def add_gaussian_blur_to_map(
    map_array: np.ndarray, *, sigma: float, passes: int
) -> np.ndarray:
    """
    Idea: https://people.csail.mit.edu/sparis/bf_course/slides/02_gaussian_blur.pdf
    Idea: https://stackoverflow.com/questions/17841098/gaussian-blur-standard-deviation-radius-and-kernel-size

    Separable Gaussian blur implemented with NumPy convolutions.
    Keep sigma small for light smoothing.
    """
    logging.info("Generating Gaussian blur...")
    logging.info(f"{sigma=}")
    logging.info(f"{passes=}")

    if sigma <= 0.0 or map_array.size == 0:
        return map_array.copy()
    
    cutoff_radius_of_gaussian_blur: int = compute_cutoff_radius_of_gaussian_blur(
        sigma=sigma
    )

    logging.info(f"{cutoff_radius_of_gaussian_blur=}")

    kernel_1d: np.ndarray = compute_1d_kernel_for_gaussian_blur(
        sigma=sigma, radius=cutoff_radius_of_gaussian_blur
    )

    output_array: np.ndarray = map_array.copy()
    for current_pass in range(1, max(1, int(passes)) + 1):
        logging.info(f"Currently at pass: {current_pass} out of {int(passes)}...")

        padded_array: np.ndarray = np.pad(
            output_array,
            (
                (cutoff_radius_of_gaussian_blur, cutoff_radius_of_gaussian_blur),
                (cutoff_radius_of_gaussian_blur, cutoff_radius_of_gaussian_blur),
            ),
            mode="reflect",
        )

        result_rows: np.ndarray = convolve_rows_for_gaussian_blur(
            padded_array=padded_array, kernel_1d=kernel_1d, map_array=map_array
        )

        result: np.ndarray = convolve_columns_for_gaussian_blur(
            result_rows=result_rows, kernel_1d=kernel_1d, map_array=map_array
        )

        output_array = result

    logging.info("Gaussian blur was generated...")

    return output_array


def apply_smootherstep_polynomial_perlin_noise(float_input: float) -> float:
    """Idea: https://iq.opengenus.org/perlin-noise/

    Applying a smooth fade function to reduce high frequency artifacts
    smootherstep: f(u)=6u^5 - 15u^4 + 10u^3"""
    return (
        float_input
        * float_input
        * float_input
        * (float_input * (float_input * 6.0 - 15.0) + 10.0)
    )


def compute_bilinear_interpolation_on_a_2D_control_grid(
    grid: np.ndarray, grid_x: float, grid_y: float
) -> float:
    """
    Idea: https://www.geeksforgeeks.org/maths/what-is-bilinear-interpolation/
    Idea: https://iq.opengenus.org/perlin-noise/

    Compute bilinear interpolation on a 2D control grid using a smooth fade
    for fractional weights to avoid sharp transitions that lead to spikes.

    Parameters:
        grid : np.ndarray
            2D array of control values indexed as grid[row, col].
        grid_x : float
            Floating point x coordinate in grid space (col coordinate).
        grid_y : float
            Floating point y coordinate in grid space (row coordinate).

    Returns:
        float
            Interpolated value at (gx, gy).
    """
    # integer grid cell indices

    x_integer_grid_cell_index: int = int(grid_x)
    y_integer_grid_cell_index: int = int(grid_y)

    # fractional offsets inside the cell

    x_fractional_offset: float = grid_x - x_integer_grid_cell_index
    y_fractional_offset: float = grid_y - y_integer_grid_cell_index

    smootherstep_x: float = apply_smootherstep_polynomial_perlin_noise(
        x_fractional_offset
    )
    smootherstep_y: float = apply_smootherstep_polynomial_perlin_noise(
        y_fractional_offset
    )

    # sample the four corner values from the control grid

    corner_value_00: float = float(
        grid[y_integer_grid_cell_index, x_integer_grid_cell_index]
    )
    corner_value_10: float = float(
        grid[y_integer_grid_cell_index, x_integer_grid_cell_index + 1]
    )
    corner_value_01: float = float(
        grid[y_integer_grid_cell_index + 1, x_integer_grid_cell_index]
    )
    corner_value_11: float = float(
        grid[y_integer_grid_cell_index + 1, x_integer_grid_cell_index + 1]
    )

    # interpolate horizontally on top and bottom rows using smooth weights

    top: float = (
        corner_value_00 * (1.0 - smootherstep_x) + corner_value_10 * smootherstep_x
    )
    bottom: float = (
        corner_value_01 * (1.0 - smootherstep_x) + corner_value_11 * smootherstep_x
    )

    # interpolate vertically between the horizontal interpolants

    interpolated_value: float = top * (1.0 - smootherstep_y) + bottom * smootherstep_y

    return interpolated_value


def generate_noise(
    *,
    scale: int,
    random_number_generator: Generator[float, None, None],
    width: int,
    height: int,
) -> np.ndarray:
    """
    Idea: https://www.geeksforgeeks.org/maths/what-is-bilinear-interpolation/
    Idea: https://iq.opengenus.org/perlin-noise/

    Create a smooth noise layer by sampling a coarse random grid and
    bilinearly interpolating it to the target resolution.
    """
    # ensure scale is at least 1 to avoid division by zero

    scale = max(1, int(scale))

    # compute control grid dimensions so that indexing x0+1, y0+1 is safe

    grid_width: int = width // scale + 2
    grid_height: int = height // scale + 2

    # fill the control grid with values from the injected RNG
    # map RNG output from [0,1) to [-1,1) to center layers around zero

    grid: np.ndarray = np.array(
        [
            [next(random_number_generator) * 2.0 - 1.0 for _ in range(grid_width)]
            for _ in range(grid_height)
        ],
        dtype=np.float64,
    )

    # prepare output noise array for this octave

    noise: np.ndarray = np.zeros((height, width), dtype=np.float64)

    # for each pixel compute its mapped position in grid space and interpolate

    for y in range(height):
        y_float: float = float(y)
        for x in range(width):
            x_float: float = float(x)

            # map pixel coordinates to control grid coordinates

            control_grid_x_coordinate: float = x_float / float(scale)
            control_grid_y_coordinate: float = y_float / float(scale)

            # bilinear interpolation with smooth fade

            value: float = compute_bilinear_interpolation_on_a_2D_control_grid(
                grid, control_grid_x_coordinate, control_grid_y_coordinate
            )

            # write interpolated value into the noise buffer

            noise[y, x] = value
    return noise


def normalize_terrain_to_the_0_to_1_range_safely(*, terrain: np.ndarray) -> np.ndarray:
    logging.info("Normalizing terrain to the 0-1 range safely...")
    minimum_value_of_terrain: float = float(np.min(terrain))
    maximum_value_of_terrain: float = float(np.max(terrain))
    logging.info(f"{minimum_value_of_terrain=}")
    logging.info(f"{maximum_value_of_terrain=}")
    terrain = terrain - minimum_value_of_terrain
    terrain_difference: float = maximum_value_of_terrain - minimum_value_of_terrain
    logging.info(f"{terrain_difference=}")
    if terrain_difference > 0.0:
        return terrain / terrain_difference
    return np.zeros_like(terrain)


def generate_terrain_heightmap(
    *,
    random_number_generator: Generator[float, None, None],
    width: int,
    height: int,
    octaves: int,
    persistence: float,
    sigma: float,
    passes: int,
    initial_scale: int | None = None,
) -> np.ndarray:
    """
    Idea: https://www.geeksforgeeks.org/maths/what-is-bilinear-interpolation/
    Idea: https://iq.opengenus.org/perlin-noise/

    Fractal heightmap generator using an injectable RNG that yields floats in [0, 1).

    Strategies to avoid spikes:
        - Use a coarse initial scale so the base octave defines large hills.
        - Centre control grid values around zero so octaves add constructive and destructive detail.
        - Use a smooth interpolation fade function to reduce sharp transitions.
    """

    logging.info("Generating terrain...")
    logging.info(f"{octaves=}")
    logging.info(f"{persistence=}")
    logging.info(f"{random_number_generator=}")

    # pick a sensible initial scale if none provided: coarse control grid for the first octave

    if initial_scale is None:
        initial_scale = max(width, height) // 10

    if initial_scale < 1:
        initial_scale = 1

    logging.info(f"{initial_scale=}")

    logging.info("Initializing the final terrain array and synthesis parameters...")

    terrain: np.ndarray = np.zeros((height, width), dtype=np.float64)
    amplitude: float = 1.0
    scale: int = int(initial_scale)

    # sum multiple octaves to produce fractal-like structure

    logging.info(
        "Adding noise for multiple octaves with Perlin noise and bilinear interpolation..."
    )

    for current_octave in range(1, octaves + 1):
        logging.info(f"Currently at octave {current_octave} out of {octaves}...")
        layer: np.ndarray = generate_noise(
            scale=scale,
            random_number_generator=random_number_generator,
            width=width,
            height=height,
        )
        terrain += amplitude * layer

        # decrease amplitude and increase frequency (halve the scale) each octave

        amplitude = amplitude * persistence
        scale = max(1, scale // 2)
    # normalize terrain to the 0-1 range safely

    logging.info("Starting to normalize terrain...")
    terrain = normalize_terrain_to_the_0_to_1_range_safely(terrain=terrain)
    logging.info("Terrain was normalized...")

    logging.info("Adding gaussian blur to map...")
    terrain = add_gaussian_blur_to_map(terrain, sigma=sigma, passes=passes)
    logging.info("Gaussian blur was added to terrain...")
    logging.info("Terrain was generated...")
    return terrain


# Default PRNG


def default_random_number_generator_from_numpy(
    seed: int,
) -> Generator[float, None, None]:
    """
    Simple infinite RNG generator that yields floats in [0, 1).
    """
    random_number_generator_object: np.random.Generator = np.random.default_rng(seed)
    while True:
        value: float = float(random_number_generator_object.random())
        yield value


def creating_2d_plot(*, heightmap: np.ndarray) -> None:
    """Idea: https://matplotlib.org/"""
    # 2D plot

    logging.info("Creating 2D plot...")
    figure_2d = plt.figure(figsize=(8, 6))
    axes_2d = figure_2d.add_subplot(1, 1, 1)
    image = axes_2d.imshow(heightmap, cmap="terrain", origin="lower")
    figure_2d.colorbar(image, ax=axes_2d)
    axes_2d.set_title("Terrain Generator - 2D")
    axes_2d.set_xlabel("X")
    axes_2d.set_ylabel("Y")
    logging.info("2D plot was created...")


def creating_contour_plot(*, heightmap: np.ndarray) -> None:
    """Idea: https://matplotlib.org/"""
    # Contour plot

    logging.info("Creating Contour plot...")
    figure_contour = plt.figure(figsize=(8, 6))
    axes_contour = figure_contour.add_subplot(1, 1, 1)
    contours = axes_contour.contour(heightmap, levels=20, colors="black")
    axes_contour.clabel(contours, inline=True, fontsize=8)
    axes_contour.set_title("Terrain Generator - Contour Map")
    axes_contour.set_xlabel("X")
    axes_contour.set_ylabel("Y")
    logging.info("Contour plot was created...")


def creating_3d_plot(*, heightmap: np.ndarray) -> None:
    """Idea: https://matplotlib.org/"""
    # 3D surface plot
    # create X, Y meshgrid matching pixel centers

    logging.info("Creating 3D plot...")
    heightmap_shape_0: int = heightmap.shape[0]
    heightmap_shape_1: int = heightmap.shape[1]
    x_coordinates: np.ndarray = np.arange(0, heightmap_shape_1, dtype=np.float64)
    y_coordinates: np.ndarray = np.arange(0, heightmap_shape_0, dtype=np.float64)

    x_coordinates_meshgrid, y_coordinates_meshgrid = np.meshgrid(
        x_coordinates, y_coordinates
    )

    # create a new figure for the 3D surface

    figure_3d = plt.figure(figsize=(10, 8))
    axes_3d = figure_3d.add_subplot(1, 1, 1, projection="3d")

    # plot the surface; linewidth set to 0 to avoid grid lines, antialiased for smoother appearance

    plot_surface = axes_3d.plot_surface(
        x_coordinates_meshgrid,
        y_coordinates_meshgrid,
        heightmap,
        cmap="terrain",
        linewidth=0,
        antialiased=True,
    )

    # add a colorbar for the surface and label axes

    figure_3d.colorbar(plot_surface, ax=axes_3d, shrink=0.6)
    axes_3d.set_title("Terrain Generator - 3D Surface")
    axes_3d.set_xlabel("X")
    axes_3d.set_ylabel("Y")
    axes_3d.set_zlabel("Elevation")

    logging.info("3D plot was created...")


def create_visualization(*, heightmap: np.ndarray) -> None:
    """Idea: https://matplotlib.org/"""

    creating_2d_plot(heightmap=heightmap)

    creating_contour_plot(heightmap=heightmap)

    creating_3d_plot(heightmap=heightmap)

    logging.info("Displaying all plots...")

    plt.show()


def log_heightmap_statistics(heightmap: np.ndarray) -> None:
    if heightmap is None or heightmap.size == 0:
        logging.info("Heightmap is empty or None")
        return None

    heightmap_shape: Tuple[int, ...] = heightmap.shape
    total_number_of_values: int = int(heightmap.size)

    minimum_height_value: float = float(np.min(heightmap))
    maximum_height_value: float = float(np.max(heightmap))
    mean_height_value: float = float(np.mean(heightmap))
    median_height_value: float = float(np.median(heightmap))
    standard_deviation_of_height_values: float = float(np.std(heightmap))
    variance_of_height_values: float = float(np.var(heightmap))

    range_of_height_values: float = maximum_height_value - minimum_height_value

    twenty_fifth_percentile_value: float = float(np.percentile(heightmap, 25))
    seventy_fifth_percentile_value: float = float(np.percentile(heightmap, 75))
    interquartile_range_value: float = seventy_fifth_percentile_value - twenty_fifth_percentile_value

    skewness_estimate: float = float(
        np.mean(((heightmap - mean_height_value) / (standard_deviation_of_height_values + 1e-12)) ** 3)
    )
    kurtosis_estimate: float = float(
        np.mean(((heightmap - mean_height_value) / (standard_deviation_of_height_values + 1e-12)) ** 4) - 3.0
    )

    number_of_unique_height_values: int = int(len(np.unique(heightmap)))

    gradient_along_x_axis: np.ndarray
    gradient_along_y_axis: np.ndarray
    gradient_along_x_axis, gradient_along_y_axis = np.gradient(heightmap)

    gradient_magnitude_array: np.ndarray = np.sqrt(
        gradient_along_x_axis ** 2 + gradient_along_y_axis ** 2
    )

    mean_gradient_magnitude: float = float(np.mean(gradient_magnitude_array))
    maximum_gradient_magnitude: float = float(np.max(gradient_magnitude_array))

    histogram_counts_array: np.ndarray
    histogram_bin_edges_array: np.ndarray
    histogram_counts_array, histogram_bin_edges_array = np.histogram(heightmap, bins=20)

    logging.info(f"Heightmap shape: {heightmap_shape}")
    logging.info(f"Total number of values: {total_number_of_values}")
    logging.info(f"Minimum height value: {minimum_height_value}")
    logging.info(f"Maximum height value: {maximum_height_value}")
    logging.info(f"Mean height value: {mean_height_value}")
    logging.info(f"Median height value: {median_height_value}")
    logging.info(f"Standard deviation: {standard_deviation_of_height_values}")
    logging.info(f"Variance: {variance_of_height_values}")
    logging.info(f"Range of values: {range_of_height_values}")
    logging.info(f"25th percentile: {twenty_fifth_percentile_value}")
    logging.info(f"75th percentile: {seventy_fifth_percentile_value}")
    logging.info(f"Interquartile range: {interquartile_range_value}")
    logging.info(f"Skewness estimate: {skewness_estimate}")
    logging.info(f"Kurtosis estimate: {kurtosis_estimate}")
    logging.info(f"Number of unique values: {number_of_unique_height_values}")
    logging.info(f"Mean gradient magnitude: {mean_gradient_magnitude}")
    logging.info(f"Maximum gradient magnitude: {maximum_gradient_magnitude}")
    logging.info(f"Histogram counts: {histogram_counts_array.tolist()}")
    logging.info(f"Histogram bin edges: {histogram_bin_edges_array.tolist()}")


def main(*args: Any, **kwargs: Any) -> None:
    """
    Run a demo producing and plotting a heightmap in 2D and 3D.
    """
    # instantiate RNG and generate a heightmap with coarser base scale and
    # moderate persistence so higher octaves do not dominate

    configure_logging()
    configuration: dict = load_configuration_yaml(config_path="configuration.yaml")
    from time import time

    prng_type: float = configuration["prng_type"]
    logging.info(f"{prng_type=}")

    random_number_generator: Optional[Generator[float, None, None]] = None

    if prng_type == "xorshift32":
        logging.info("Choosing Xorshift32 as PRNG...")

        random_seed: int = int(str(time()).replace(".", "")[12:19])
        logging.info(f"{random_seed=}")

        random_number_generator: Generator[float, None, None] = (
            xorshift_32_float_generator(seed=random_seed)
        )

    elif prng_type == "wichmann_hill":
        logging.info("Choosing Wichmann-Hill as PRNG...")

        random_seed_1: int = int(str(time()).replace(".", "")[12:19])
        logging.info(f"{random_seed_1=}")

        random_seed_2: int = int(str(time()).replace(".", "")[12:19])
        logging.info(f"{random_seed_2=}")

        random_seed_3: int = int(str(time()).replace(".", "")[12:19])
        logging.info(f"{random_seed_3=}")

        random_number_generator: Generator[float, None, None] = wichmann_hill_generator(
            seed_1=random_seed_1,
            seed_2=random_seed_2,
            seed_3=random_seed_3
        )

    elif prng_type == "logistic_map":
        logging.info("Choosing Logistic Map as PRNG...")

        random_seed: float = int(str(time()).replace(".", "")[12:19]) / 10**7
        logging.info(f"{random_seed=}")

        random_number_generator: Generator[float, None, None] = (
            logistic_map_pseudorandom_generator(seed=random_seed)
        )

    else:
        logging.info("Choosing default PRNG from Numpy...")

        random_seed: int = int(str(time()).replace(".", "")[12:19])
        logging.info(f"{random_seed=}")

        random_number_generator: Generator[float, None, None] = (
            default_random_number_generator_from_numpy(seed=random_seed)
        )

    heightmap: np.ndarray = generate_terrain_heightmap(
        random_number_generator=random_number_generator,
        width=256,
        height=256,
        octaves=configuration["terrain_octave_count"],
        persistence=configuration["terrain_persistence_factor"],
        sigma=configuration["gaussian_sigma"],
        passes=configuration["gaussian_pass_count"],
        initial_scale=configuration["terrain_initial_scale"],
    )

    log_heightmap_statistics(heightmap=heightmap)

    create_visualization(heightmap=heightmap)

    
if __name__ == "__main__":
    main()
