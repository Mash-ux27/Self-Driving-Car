import pygame
import config
import math

origin_x = 100
origin_y = 300

# Represents the car's image, physical state, heading, and keyboard movement.
class Car:
	def __init__(self):
		# Create a vertical car shape so its heading is visible while steering.
		self.base_image = pygame.Surface((20,20))
		self.base_image.fill((255,0,0))
		self.image = self.base_image.copy()
		self.mask = pygame.mask.from_surface(self.image)
		self.rect = self.image.get_rect(center=(origin_x,origin_y))
		self.speed = 4
		self.angle = -90
		self.alive = True

	# Calculate the next position from steering and throttle without committing it.
	def move(self, keys):
		if keys[pygame.K_LEFT]: self.angle -= 3
		if keys[pygame.K_RIGHT]: self.angle += 3

		new_x, new_y = self.rect.x, self.rect.y

		if keys[pygame.K_UP]:
			radians = math.radians(self.angle)
			new_x += self.speed * math.cos(radians)
			new_y += self.speed * math.sin(radians)
		if keys[pygame.K_DOWN]:
			radians = math.radians(self.angle)
			new_x -= self.speed * math.cos(radians)
			new_y -= self.speed * math.sin(radians)

		return new_x, new_y

	# Convert neural-network steering and throttle outputs into a candidate position.
	def move_with_brain(self, outputs):
		# Limit early random behavior to gentle steering and forward-only throttle.
		steering = float(outputs[0]) * 0.5
		throttle = (float(outputs[1]) + 1) / 2
		self.angle += steering * 3

		radians = math.radians(self.angle)
		new_x = self.rect.x + self.speed * throttle * math.cos(radians)
		new_y = self.rect.y + self.speed * throttle * math.sin(radians)
		return new_x, new_y

	# Restore the car to its starting state after it dies or reaches its lifespan cap.
	def reset(self):
		self.angle = -90
		self.alive = True
		self.image = self.base_image.copy()
		self.mask = pygame.mask.from_surface(self.image)
		self.rect = self.image.get_rect(center=(100,300))
