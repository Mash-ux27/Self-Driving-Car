# Self-Driving Car AI

This project trains a small neural-network-controlled car to navigate a track using a genetic algorithm. Each generation keeps the best-performing brains, mutates them, and evolves a new population that tries to reach more checkpoints without crashing into the walls.

## Overview

The simulation uses:

- a car with a heading angle and five distance sensors
- a feedforward neural network for steering and acceleration decisions
- checkpoint-based fitness scoring
- evolution across generations to improve the driving behavior

The program runs in a Pygame window and continuously saves the best training state so progress is not lost between runs.

## How it works

- `sensors.py` casts five rays around the car and measures how far each ray reaches before hitting a wall.
- `brain.py` defines the neural network that takes those sensor values as input and outputs steering/throttle signals.
- `car.py` converts the network output into motion for the current frame.
- `ga.py` selects the best-performing genomes, breeds new ones, and applies mutation.
- `track.py` loads the track image, detects black wall pixels, and defines the checkpoints used as the fitness target.
- `main.py` runs the training loop, updates the population, and renders the simulation.

## Requirements

Install the Python dependencies:

```bash
pip install pygame numpy
```

Python 3.10+ is recommended.

## Run the project

From the project folder:

```bash
python main.py
```

On Windows, this also works:

```bash
py main.py
```

## Project structure

- `main.py` – main training loop and simulation
- `brain.py` – neural network logic
- `car.py` – car physics and movement
- `ga.py` – genetic algorithm and fitness tracking
- `sensors.py` – sensor ray casting and normalization
- `track.py` – track image, wall mask, and checkpoint layout
- `config.py` – display/window settings
- `training_state.npz` – saved population and generation data
- `track.png` – track image used by the simulator

## Customization

You can tune the behavior by editing the following:

- `config.py` for window size and car size
- `track.py` for checkpoint positions and wall mask behavior
- `main.py` for population size and mutation strength
- `ga.py` for the fitness and breeding logic

Example tuning options in `main.py`:

```python
population = Population(50)
...
population.evolve(mutation_rate=0.1, mutation_strength=0.15)
```

## Important note about tracks

Each new track usually needs updated checkpoint rectangles in `track.py`. The fitness system rewards reaching checkpoints in order, so the checkpoint layout must match the new track design.

## Training behavior

The car is evaluated over a fixed lifespan. If it crashes or reaches the time limit, the generation is scored and evolved. The best fitness is displayed on screen while training continues.

## License

This project is provided as a simple educational example for evolutionary machine learning and self-driving simulation.
