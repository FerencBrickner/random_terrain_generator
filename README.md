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

These statistics are used later for analysis and classification.

## Workflow

1. Generate terrain with a selected PRNG.
2. Compute terrain statistics.
3. Store the results in SQLite.
4. Train the classification model on the recorded samples.
5. Visualize which PRNGs produce specific statistical artifacts.

## Goal

The goal of the project is to understand whether different PRNGs leave detectable fingerprints in generated terrain and to visualize those differences in a reproducible way.

## Notes

- The terrain database is serverless and stored locally in SQLite.
- The classification pipeline is designed for tabular statistical data.
- Heightmap statistics are the main signal used for model training and interpretation.