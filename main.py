import pygame
import config
import sys
import math
import numpy as np
import os

pygame.init()
screen = pygame.display.set_mode(config.WINDOW_SIZE)

# Import resource modules after pygame has a display for alpha conversion.
from car import Car
from track import track_img, wall_mask, CHECKPOINTS
from sensors import get_sensor_readings, get_normalized_sensor_readings
from ga import Population, CheckpointFitness

clock = pygame.time.Clock()
font = pygame.font.Font(None, 28)
SAVE_PATH = "training_state.npz"
# Restore the last population when possible; otherwise begin with random brains.
if os.path.exists(SAVE_PATH):
    population, generation, best_fitness = Population.load(SAVE_PATH)
else:
    # Use more brains per generation to explore more driving behaviors.
    population = Population(50)
    generation = 1
    best_fitness = 0

# Every brain receives its own car and checkpoint tracker.
cars = [Car() for brain in population.brains]
fitness_trackers = [CheckpointFitness(CHECKPOINTS) for car in cars]

frame_count = 0
# Shorten each generation so more generations can be tested quickly.
MAX_LIFESPAN = 600

while True:
    # Clear the previous frame before drawing the track and current cars.
    screen.fill((255,255,255))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # Preserve the current population so the next run can continue training.
            population.save(SAVE_PATH, generation, best_fitness)
            pygame.quit()
            sys.exit()
        if frame_count > MAX_LIFESPAN:
            car.alive = False
    # Evaluate and replace the population when a generation has finished.
    if frame_count > MAX_LIFESPAN or not any(car.alive for car in cars):
        population.fitnesses = np.array([tracker.score for tracker in fitness_trackers])
        best_fitness = int(np.max(population.fitnesses))
        # Use moderate mutation so useful behaviors survive while new ones appear.
        population.evolve(mutation_rate=0.1, mutation_strength=0.15)
        cars = [Car() for brain in population.brains]
        fitness_trackers = [CheckpointFitness(CHECKPOINTS) for car in cars]
        frame_count = 0
        generation += 1
        population.save(SAVE_PATH, generation, best_fitness)

    # Advance each living car using its sensors and matching brain.
    frame_count += 1

    for index, car in enumerate(cars):
        if car.alive:
            sensor_inputs = get_normalized_sensor_readings(car)
            brain_outputs = population.brains[index].forward(sensor_inputs)
            new_x, new_y = car.move_with_brain(brain_outputs)

            if wall_mask.overlap(car.mask, (new_x, new_y)):
                car.alive = False
            else:
                car.rect.x, car.rect.y = new_x, new_y
                fitness_trackers[index].update(car.rect)

    screen.blit(track_img, (0, 0))

    for car in cars:
        if car.alive:
            screen.blit(car.image, car.rect)

            # Draw the five sensor rays for each living car.
            for ray_angle, distance in get_sensor_readings(car):
                radians = math.radians(ray_angle)
                end_x = car.rect.center[0] + math.cos(radians) * distance
                end_y = car.rect.center[1] + math.sin(radians) * distance
                pygame.draw.line(
                    screen,
                    (0, 0, 255),
                    car.rect.center,
                    (round(end_x), round(end_y)),
                    2,
                )

    # Draw a solid panel so the generation counter stays readable over the track.
    status = font.render(
        f"GENERATION {generation}   BEST CHECKPOINTS {best_fitness}",
        True,
        (0, 0, 0),
    )
    status_panel = pygame.Rect(10, 10, status.get_width() + 20, status.get_height() + 12)
    pygame.draw.rect(screen, (255, 255, 255), status_panel)
    pygame.draw.rect(screen, (0, 0, 0), status_panel, 2)
    screen.blit(status, (20, 16))

    pygame.display.flip()
    clock.tick(60)