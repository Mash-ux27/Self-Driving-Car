import numpy as np

from brain import Brain


# Holds the brains and fitness values for one genetic-algorithm generation.
class Population:
	def __init__(self, size=20):
		# Start with independently randomized brains and one fitness slot per brain.
		self.brains = [Brain() for _ in range(size)]
		self.fitnesses = np.zeros(size)

	# Return all brain genomes so selection and mutation can process them later.
	def genomes(self):
		return [brain.genome() for brain in self.brains]

	# Save every brain genome and the visible training progress to disk.
	def save(self, path, generation, best_fitness):
		np.savez(
			path,
			genomes=np.array(self.genomes()),
			generation=generation,
			best_fitness=best_fitness,
		)

	# Restore a previously saved population instead of creating random brains.
	@classmethod
	def load(cls, path):
		data = np.load(path)
		genomes = data["genomes"]
		population = cls(len(genomes))
		# Discard older saved brains when their genome shape differs from this network.
		if genomes.shape[1] != population.brains[0].genome().size:
			return population, 1, 0
		population.brains = [Brain.from_genome(genome) for genome in genomes]
		return population, int(data["generation"]), int(data["best_fitness"])

	# Create the next generation by preserving the best brain and breeding the rest.
	def evolve(self, mutation_rate=0.05, mutation_strength=0.2):
		# Rank genomes so the highest-scoring brain becomes the elite parent.
		ranked_indices = np.argsort(self.fitnesses)[::-1]
		parent_count = max(2, len(self.brains) // 2)
		parent_genomes = [self.brains[index].genome() for index in ranked_indices[:parent_count]]
		new_genomes = [parent_genomes[0].copy()]

		while len(new_genomes) < len(self.brains):
			# Combine two selected parents, then mutate some child genes.
			first_parent = parent_genomes[np.random.randint(parent_count)]
			second_parent = parent_genomes[np.random.randint(parent_count)]
			child = np.where(np.random.random(first_parent.size) < 0.5, first_parent, second_parent)
			mutation_mask = np.random.random(child.size) < mutation_rate
			child[mutation_mask] += np.random.normal(0, mutation_strength, mutation_mask.sum())
			new_genomes.append(child)

		self.brains = [Brain.from_genome(genome) for genome in new_genomes]
		self.fitnesses = np.zeros(len(self.brains))


# Tracks ordered checkpoint progress for one car without rewarding raw distance.
class CheckpointFitness:
	def __init__(self, checkpoints):
		self.checkpoints = checkpoints
		self.next_checkpoint = 0
		self.score = 0

	# Award one point when the car reaches its next checkpoint in order.
	def update(self, car_rect):
		if self.next_checkpoint >= len(self.checkpoints):
			return self.score

		if car_rect.colliderect(self.checkpoints[self.next_checkpoint]):
			self.next_checkpoint += 1
			self.score += 1

		return self.score

	# Clear checkpoint progress when a new generation begins.
	def reset(self):
		self.next_checkpoint = 0
		self.score = 0
