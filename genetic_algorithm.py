import random

target = "hello"

class Individual:
    def __init__(self, genes=None):
        if genes is None:
            genes = [chr(random.randint(97,122)) for _ in target]
        self.genes = genes
        self.fitness = self.evaluate()

    def evaluate(self):
        return sum(g == t for g,t in zip(self.genes, target))

    def mutate(self, rate=0.1):
        for i in range(len(self.genes)):
            if random.random() < rate:
                self.genes[i] = chr(random.randint(97,122))
        self.fitness = self.evaluate()

    def crossover(self, other):
        point = random.randint(1, len(self.genes)-1)
        child_genes = self.genes[:point] + other.genes[point:]
        return Individual(child_genes)

class GA:
    def __init__(self, pop_size=20):
        self.population = [Individual() for _ in range(pop_size)]

    def evolve(self, generations=100):
        for _ in range(generations):
            self.population.sort(key=lambda ind: ind.fitness, reverse=True)
            if self.population[0].fitness == len(target):
                break
            next_gen = self.population[:2]
            while len(next_gen) < len(self.population):
                parent1 = random.choice(self.population[:10])
                parent2 = random.choice(self.population[:10])
                child = parent1.crossover(parent2)
                child.mutate()
                next_gen.append(child)
            self.population = next_gen
        return self.population[0]

if __name__ == "__main__":
    ga = GA(pop_size=20)
    best = ga.evolve(generations=200)
    print("Evolved:", ''.join(best.genes))
