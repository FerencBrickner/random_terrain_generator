# Random Terrain Generation and PRNG Classification

This project explores **random terrain generation** with multiple selectable PRNGs and a separate **classification pipeline** that learns which statistical artifacts are associated with each generator.

## Overview

The project has two main parts:

1. **Terrain generation**
   - Generates random terrain using selectable PRNGs such as Xorshift32, Wichmann-Hill, and the logistic map.
   - Applies terrain-smoothing and shaping techniques such as:
     - Gaussian blur with convolution
     - smootherstep polynomial shaping with Perlin noise
     - bilinear interpolation
   - Plots the following:
     - 2D plot
     - 3D plot
     - Hillshade plot
     - Contour plot
     - heightmap histogram

2. **Terrain classification**
   - Uses a **multi-class supervised classification** approach with **gradient boosted decision trees**.
   - Learns how different PRNGs influence terrain statistics and visual artifacts.
   - Produces visual reports such as:
     - confusion matrix
     - feature importance plot
     - permutation importance plot
     - per-PRNG artifact comparisons

## Dependencies

Dependencies are located within requirements.txt

numpy
matplotlib
PyYAML
sqlalchemy
scikit-learn
pandas
pytest
memory-profiler
psutil

pip install -r requirements.txt

## Project Structure

### `random_terrain.py`
This script is responsible for terrain generation.

#### Parameters

**`gaussian_blur`**
- `sigma`
- `passes`

**`terrain_generation`**
- `octaves`
- `persistence`
- `initial_scale`
- `continent_effect_strength`

**`prng`**
- `prng_type`

### `terrain_classification.py`
This script trains a classifier using terrain statistics stored in the database.

It performs **multi-class supervised classification** using **gradient boosted decision trees** for statistical pattern recognition.

It helps identify which PRNGs tend to produce which terrain artifacts, and it generates visual summaries including:
- feature importance
- confusion matrix
- classification metrics
- artifact comparison plots

## Stored Terrain Statistics

Each generated terrain sample is written to a **SQLite** database using **SQLAlchemy** as the ORM.

The stored statistics include values such as:
- variance
- percentiles
- interquartile range
- kurtosis
- skewness
- gradients
- unique value counts
- histogram data
- Moran's I spatial correlation
- Geary's C spatial correlation

These statistics are used later for analysis and classification.

## Workflow

1. Generate terrain with a selected PRNG.
2. Compute terrain statistics.
3. Store the results in SQLite.
4. Train the classification model on the recorded samples.
5. Visualize which PRNGs produce specific statistical artifacts.

## Goal

The goal of the project is to understand whether different PRNGs leave detectable fingerprints in generated terrain and to visualize those differences in a reproducible way.

## Testing and Automation

### Unit tests

This project includes unit tests to verify the correctness of the core terrain generation and PRNG logic. These tests help ensure that changes do not accidentally break important behavior.

The unit tests focus on areas such as:

- returntype assertions
- performance checks
- returnvalue assertions
- determinisim assertions
- array shape assertions
- zero seed handling assertions
- exception safety assertions

### GitHub Actions automated unit tests

The repository includes a GitHub Actions workflow that runs the unit tests automatically on every push. This provides continuous verification that the project still works after each change is committed and pushed to the remote repository.

The automated pipeline:

- checks out the repository
- installs the required dependencies
- runs `pytest`
- shows test output in the GitHub Actions logs
- can generate an HTML test report

## Notes

- The random terrain generator Python script and the terrain classification Python script are both run locally as console applications
- The terrain database is serverless and stored locally in SQLite.
- The classification pipeline is designed for tabular statistical data.
- Heightmap statistics are the main signal used for model training and interpretation.