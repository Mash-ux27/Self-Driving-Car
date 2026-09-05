import pygame
import math
from track import wall_mask
def cast_ray(origin, angle, max_distance=300):
    radians = math.radians(angle)

    for distance in range(max_distance):
        # Calculate the point along the ray here.
        x = round(origin[0] + math.cos(radians) * distance)
        y = round(origin[1] + math.sin(radians) * distance)

        # Stop the ray when it reaches a black wall pixel.
        if wall_mask.get_at((x, y)):
            return distance

    return max_distance
# Cast five rays around the car's heading and return each angle with its distance.
def get_sensor_readings(car):
    readings = []
    for offset in (-90,-45,0,45,90):
        ray_angle = car.angle + offset
        distance = cast_ray(car.rect.center, ray_angle)
        readings.append((ray_angle, distance))
    return readings

# Convert wall distances into values between zero and one for neural-network input.
def get_normalized_sensor_readings(car, max_distance=300):
    readings = get_sensor_readings(car)
    return [distance / max_distance for angle, distance in readings]


