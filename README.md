# PFR Deactivation Simulation

This project simulates a Plug Flow Reactor (PFR) with catalyst deactivation, based on the models for catalytic alkylation reactions.

## Description

The simulation models the evolution of propylene concentration and catalyst activity profiles along the reactor length over time. It uses the 1MARI model for reaction kinetics and a time-dependent deactivation model.

## Features

-   Time-dependent simulation of a PFR.
-   1MARI model for reaction kinetics.
-   Propylene-dependent catalyst deactivation model.
-   Configuration via JSON file.
-   Animated plot of the results over time.

## How to Run

1.  **Install dependencies:**
    ```bash
    pip install numpy pandas matplotlib
    ```

2.  **Run the simulation:**
    ```bash
    python pfr_simulation.py
    ```
    This will run the simulation with default parameters from `config/default.json`.

3.  **Customize parameters:**
    -   You can modify the `config/default.json` file to change the simulation parameters.
    -   Alternatively, you can override parameters using command-line arguments. For example:
        ```bash
        python pfr_simulation.py --k_reaction 10.0 --t_final 5.0
        ```
    -   For a full list of command-line arguments, run:
        ```bash
        python pfr_simulation.py --help
        ```

## Output

The simulation generates an animated GIF file (e.g., `results/pfr_profile_animated.gif`) showing the evolution of the concentration and activity profiles over time.

# Updated on 2026-01-09
