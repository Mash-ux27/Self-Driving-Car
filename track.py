import pygame

# Load the track image and create the mask used to identify black walls.
track_img = pygame.image.load("track.png").convert_alpha()
wall_mask = pygame.mask.from_threshold(
    track_img,
    (0, 0, 0),
    (10, 10, 10, 255),
)

# Define ordered checkpoint areas that cars must reach for fitness.
CHECKPOINTS = [
    pygame.Rect(80, 250, 40, 20),
    pygame.Rect(80, 200, 40, 20),
    pygame.Rect(80, 150, 40, 20),
    pygame.Rect(80, 100, 40, 20),
    pygame.Rect(80, 50, 40, 20),
]
