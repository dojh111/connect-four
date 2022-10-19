import random
from hundred_tournament import HundredTournament

class TrainGeneticAgents():
    def __init__(self, num_features=11):
        self.num_features = num_features
        self.num_decimals = 3
        self.num_agents = 100
        self.max_generations = 500
        self.max_consecutive_wins = 30
        self.current_generation = 0
        self.mutation_rate = 0.05
        print(f'[NUMBER OF FEATURES] {str(self.num_features)}')
        print(f'[DECIMAL POINTS] Generating weights to: {str(self.num_decimals)} decimal numbers')
        print(f'[MUTATION RATE] Percentage mutation: {str(self.mutation_rate)}')
        print(f'[NUMBER OF TRAINING GENERATIONS] {str(self.max_generations)}')
    
    def train(self):
        best_weights = []
        previous_winner_weights = []    # If same weights wins 50 tournaments in a row, end
        consecutive_win_counter = 0
        # ----------- Start Training ----------- #
        print(f'[TRAINING] Current Generation: {str(self.current_generation)}/{str(self.max_generations)}')
        self.initial_weights = self.generate_initial_weights()
        tournament = HundredTournament(self.initial_weights, 9, 6, 7)
        results = tournament.play_tournament()
        best_weights = results[0]
        previous_winner_weights = results[1]
        consecutive_win_counter += 1
        self.current_generation += 1
        # Continue training remaining generations
        for i in range(1, self.max_generations):
            print(f'[TRAINING] Current Generation: {str(self.current_generation)}/{str(self.max_generations)}')
            new_generation = self.generate_new_generation(best_weights)
            print('[NEW GENERATION] Done generating new generation')
            contestant_weights = new_generation + best_weights
            tournament = HundredTournament(contestant_weights, 9, 6, 7)
            results = tournament.play_tournament()
            best_weights = results[0]
            winner_weights = results[1]
            print('GENERATION WINNER WEIGHTS:')
            print(winner_weights)
            if self.check_if_same_winner(previous_winner_weights, winner_weights):
                consecutive_win_counter += 1
                if consecutive_win_counter == self.max_consecutive_wins:
                    print(f'[TRAINING TERMINATION] Stopping at {str(self.current_generation)}/{str(self.max_generations)} generations')
                    print(f'[TRAINING TERMINATION] Agent has won {str(consecutive_win_counter)} times in a row')
                    print('Winner Weights: ')
                    print(previous_winner_weights)
                    return previous_winner_weights
            else:
                previous_winner_weights = winner_weights
                consecutive_win_counter = 1
            self.current_generation += 1
        print(f'[TRAINING TERMINATION] Max number of generations reached {str(self.current_generation)}/{str(self.max_generations)}')
        print('BEST WEIGHTS OBTAINED: ')
        print(previous_winner_weights)
        return previous_winner_weights

    def check_if_same_winner(self, previous_winner_weights, new_winner_weights):
        for i in range(0, self.num_features):
            if previous_winner_weights[i] != new_winner_weights[i]:
                return False
        return True

    '''
    Generates the initial 100 random weights for use in the genetic algorithm
    '''
    def generate_initial_weights(self):
        contestant_weights = []
        for i in range(0, self.num_agents):
            feature_weights = []
            for j in range(0, self.num_features):
                weight = round(random.random(), self.num_decimals)
                feature_weights.append(weight)
            contestant_weights.append(feature_weights)
        return contestant_weights

    '''
    Generate new generation by taking the original 10 winners to generate 90 more new generation
    Randomly mutate the values for a feature weight at a 5% rate. Regenerate feature weight
    '''
    def generate_new_generation(self, best_weights):
        new_generation = []
        while len(new_generation) < 90:
            weight_sample = random.sample(best_weights, 2)
            cut_index = random.randrange(1, self.num_features)
            weight_set_one = weight_sample[0]
            weight_set_two = weight_sample[1]
            cut_one = weight_set_one[0:cut_index]
            cut_two = weight_set_two[cut_index:len(weight_set_two)]
            new_child = cut_one + cut_two
            # Roll mutation chance
            mutation_roll = random.random()
            if mutation_roll <= self.mutation_rate:
                # Mutate random feature weight by regenerating column
                mutation_index = random.randrange(0, self.num_features)
                new_child[mutation_index] = round(random.random(), self.num_decimals)
            new_generation.append(new_child)
        return new_generation

if __name__ == '__main__':
    genetic_trainer = TrainGeneticAgents(13)  # Remember to update number of features
    best_weights = genetic_trainer.train()