import random
import numpy as np
from typing import List, Tuple
import statistics
import time

class GeneticAlgorithm:
    def __init__(self, population_size: int, chromosome_length: int, pc: float, pm: float):
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.pc = pc  # crossover probability
        self.pm = pm  # mutation probability
        
    def create_population(self) -> List[List[int]]:
        """Create initial random population"""
        return [[random.randint(0, 1) for _ in range(self.chromosome_length)] 
                for _ in range(self.population_size)]
    
    def fitness(self, chromosome: List[int]) -> int:
        """Count number of ones in chromosome"""
        return sum(chromosome)
    
    def select_parent(self, population: List[List[int]], fitness_values: List[int]) -> List[int]:
        """Select parent using roulette wheel selection"""
        total_fitness = sum(fitness_values)
        if total_fitness == 0:
            return random.choice(population)
        
        r = random.uniform(0, total_fitness)
        current_sum = 0
        
        for chromosome, fit in zip(population, fitness_values):
            current_sum += fit
            if current_sum > r:
                return chromosome
        
        return population[-1]  # fallback
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Perform single-point crossover"""
        if random.random() < self.pc:
            point = random.randint(1, self.chromosome_length - 1)
            offspring1 = parent1[:point] + parent2[point:]
            offspring2 = parent2[:point] + parent1[point:]
            return offspring1, offspring2
        return parent1.copy(), parent2.copy()
    
    def mutate(self, chromosome: List[int]) -> List[int]:
        """Perform bitwise mutation"""
        return [1 - bit if random.random() < self.pm else bit 
                for bit in chromosome]
    
    def run_generation(self, population: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
        """Run one generation of the GA"""
        fitness_values = [self.fitness(chrom) for chrom in population]
        new_population = []
        
        while len(new_population) < self.population_size:
            parent1 = self.select_parent(population, fitness_values)
            parent2 = self.select_parent(population, fitness_values)
            
            offspring1, offspring2 = self.crossover(parent1, parent2)
            offspring1 = self.mutate(offspring1)
            offspring2 = self.mutate(offspring2)
            
            new_population.extend([offspring1, offspring2])
        
        if len(new_population) > self.population_size:
            new_population = new_population[:self.population_size]
            
        return new_population, fitness_values

def run_experiment(pc: float, pm: float, num_runs: int = 20) -> List[int]:
    """Run multiple experiments and return generations needed to find optimal solution"""
    optimal_fitness = 20  # all ones
    max_generations = 1000  # prevent infinite loops
    generations_to_optimal = []
    
    ga = GeneticAlgorithm(100, 20, pc, pm)
    
    for run in range(num_runs):
        population = ga.create_population()
        generation = 0
        found_optimal = False
        
        while generation < max_generations and not found_optimal:
            population, fitness_values = ga.run_generation(population)
            max_fitness = max(fitness_values)
            
            if max_fitness == optimal_fitness:
                generations_to_optimal.append(generation)
                found_optimal = True
            
            generation += 1
            
        if not found_optimal:
            generations_to_optimal.append(max_generations)
    
    return generations_to_optimal

# Run experiments with different parameters
def run_comparison_experiments():
    experiments = [
        ("With crossover", 0.7, 0.001),
        ("Without crossover", 0.0, 0.001),
        ("High mutation", 0.7, 0.01),
        ("Low mutation", 0.7, 0.0001),
        ("High crossover", 0.9, 0.001),
        ("Low crossover", 0.3, 0.001),
    ]
    
    results = {}
    
    for name, pc, pm in experiments:
        
        start_time = time.time()

        generations = run_experiment(pc, pm)
        avg_generations = statistics.mean(generations)
        std_dev = statistics.stdev(generations)
        end_time = time.time()
        execution_time = end_time - start_time
        results[name] = {
            "avg_generations": avg_generations,
            "std_dev": std_dev,
            "min": min(generations),
            "max": max(generations),
            "execution_time": execution_time
        }
    return results

# Run the experiments and print results
results = run_comparison_experiments()
for name, stats in results.items():
    print(f"\n{name}:")
    print(f"Average generations: {stats['avg_generations']:.2f} ± {stats['std_dev']:.2f}")
    print(f"Range: {stats['min']} to {stats['max']} generations")
    print(f"Execution time: {stats['execution_time']:.2f} seconds")
