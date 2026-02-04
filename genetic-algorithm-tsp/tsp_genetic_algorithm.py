"""
CSCI 561 - Foundations of Artificial Intelligence
Problem 1: 3D Traveling Salesman Problem using Genetic Algorithm
Name: Rohan Kumar Lala
USC ID: 1220932027
"""

import random
import math
import time

class TSPGeneticAlgorithm:
    def __init__(self, cities, population_size=100, elite_size=20, mutation_rate=0.01, generations=500):
        self.cities = cities
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        
    def euclidean_distance(self, city1, city2):
        """Calculate 3D Euclidean distance between two cities"""
        return math.sqrt((city1[0] - city2[0])**2 + 
                        (city1[1] - city2[1])**2 + 
                        (city1[2] - city2[2])**2)
    
    def calculate_route_distance(self, route):
        """Calculate total distance for a given route"""
        total_distance = 0
        for i in range(len(route)):
            from_city = self.cities[route[i]]
            to_city = self.cities[route[(i + 1) % len(route)]]
            total_distance += self.euclidean_distance(from_city, to_city)
        return total_distance
    
    def create_initial_population(self):
        """Create initial population with some optimization"""
        population = []
        
        # Add completely random routes
        for _ in range(self.population_size // 2):
            route = list(range(len(self.cities)))
            random.shuffle(route)
            population.append(route)
        
        # Add nearest neighbor heuristic variants
        for start_city in range(min(len(self.cities), self.population_size // 2)):
            route = self.nearest_neighbor_route(start_city % len(self.cities))
            population.append(route)
        
        # Fill remaining spots with random routes if needed
        while len(population) < self.population_size:
            route = list(range(len(self.cities)))
            random.shuffle(route)
            population.append(route)
            
        return population
    
    def nearest_neighbor_route(self, start_city):
        """Generate route using nearest neighbor heuristic"""
        unvisited = set(range(len(self.cities)))
        route = [start_city]
        unvisited.remove(start_city)
        
        current_city = start_city
        while unvisited:
            nearest_city = min(unvisited, 
                             key=lambda city: self.euclidean_distance(
                                 self.cities[current_city], self.cities[city]))
            route.append(nearest_city)
            unvisited.remove(nearest_city)
            current_city = nearest_city
            
        return route
    
    def fitness_function(self, route):
        """Calculate fitness (inverse of distance)"""
        distance = self.calculate_route_distance(route)
        return 1 / distance if distance > 0 else float('inf')
    
    def rank_routes(self, population):
        """Rank routes by fitness"""
        fitness_results = []
        for i, route in enumerate(population):
            fitness_results.append((i, self.fitness_function(route)))
        return sorted(fitness_results, key=lambda x: x[1], reverse=True)
    
    def selection(self, ranked_population):
        """Roulette wheel selection with elitism"""
        selection_results = []
        
        # Keep elite individuals
        for i in range(self.elite_size):
            selection_results.append(ranked_population[i][0])
        
        # Roulette wheel selection for remaining
        total_fitness = sum([x[1] for x in ranked_population])
        
        for _ in range(len(ranked_population) - self.elite_size):
            pick = random.uniform(0, total_fitness)
            current = 0
            for individual in ranked_population:
                current += individual[1]
                if current > pick:
                    selection_results.append(individual[0])
                    break
            
        return selection_results
    
    def create_mating_pool(self, population, selection_results):
        """Create mating pool from selected individuals"""
        return [population[i] for i in selection_results]
    
    def crossover(self, parent1, parent2):
        """Two-point crossover exactly as described in PDF"""
        size = len(parent1)
        start = random.randint(0, size - 2)
        end = random.randint(start + 1, size - 1)
        
        # Step 1: Take substring from parent1
        child = []
        substring_from_parent1 = parent1[start:end + 1]
        taken = set(substring_from_parent1)
        
        # Step 2: Fill positions before substring with cities from parent2
        for i in range(start):
            for city in parent2:
                if city not in taken:
                    child.append(city)
                    taken.add(city)
                    break
        
        # Step 3: Add the substring from parent1
        child.extend(substring_from_parent1)
        
        # Step 4: Fill remaining positions with remaining cities from parent2
        for city in parent2:
            if city not in taken:
                child.append(city)
                taken.add(city)
                
        return child
    
    def mutate(self, individual):
        """Improved mutation with multiple strategies"""
        if random.random() < self.mutation_rate:
            # 70% chance for swap mutation
            if random.random() < 0.7:
                i = random.randint(0, len(individual) - 1)
                j = random.randint(0, len(individual) - 1)
                individual[i], individual[j] = individual[j], individual[i]
            # 30% chance for reverse mutation (helps with local optimization)
            else:
                i = random.randint(0, len(individual) - 2)
                j = random.randint(i + 1, len(individual) - 1)
                individual[i:j+1] = individual[i:j+1][::-1]
                
        return individual
    
    def breed_population(self, mating_pool):
        """Create new generation through crossover and mutation"""
        children = []
        
        # Keep elite individuals
        for i in range(self.elite_size):
            children.append(mating_pool[i])
        
        # Generate children through crossover
        for _ in range(len(mating_pool) - self.elite_size):
            parent1, parent2 = random.sample(mating_pool[:self.elite_size * 2], 2)
            child = self.crossover(parent1, parent2)
            children.append(child)
        
        # Apply mutation
        mutated_population = [self.mutate(child[:]) for child in children]
        return mutated_population
    
    def solve(self):
        """Main genetic algorithm loop"""
        population = self.create_initial_population()
        best_distance = float('inf')
        best_route = None
        
        for generation in range(self.generations):
            ranked_pop = self.rank_routes(population)
            current_best_distance = 1 / ranked_pop[0][1]
            
            if current_best_distance < best_distance:
                best_distance = current_best_distance
                best_route = population[ranked_pop[0][0]][:]
            
            selection_results = self.selection(ranked_pop)
            mating_pool = self.create_mating_pool(population, selection_results)
            population = self.breed_population(mating_pool)
        
        return best_route, best_distance

def read_input():
    """Read input from file"""
    with open('input.txt', 'r') as f:
        lines = f.readlines()
    
    n = int(lines[0].strip())
    cities = []
    for i in range(1, n + 1):
        x, y, z = map(int, lines[i].strip().split())
        cities.append((x, y, z))
    
    return cities

def write_output(route, distance, cities):
    """Write output to file"""
    # For TSP, any starting city is valid since it's a cycle
    # Just use the route as found by the algorithm
    with open('output.txt', 'w') as f:
        f.write(f"{distance:.3f}\n")
        for city_idx in route:
            city = cities[city_idx]
            f.write(f"{city[0]} {city[1]} {city[2]}\n")
        # Return to start city
        start_city = cities[route[0]]
        f.write(f"{start_city[0]} {start_city[1]} {start_city[2]}\n")

def main():
    """Main function"""
    cities = read_input()
    
    # Balanced parameters - better solutions without timeouts
    n_cities = len(cities)
    if n_cities <= 10:
        pop_size, elite_size, generations = 80, 16, 400
    elif n_cities <= 20:
        pop_size, elite_size, generations = 100, 20, 300
    elif n_cities <= 50:
        pop_size, elite_size, generations = 120, 24, 200
    else:
        pop_size, elite_size, generations = 150, 30, 150
    
    # Single run to stay within time limits
    ga = TSPGeneticAlgorithm(cities, pop_size, elite_size, 
                           mutation_rate=0.02, generations=generations)
    route, distance = ga.solve()
    
    write_output(route, distance, cities)

if __name__ == "__main__":
    main()