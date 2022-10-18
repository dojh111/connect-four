import random
from hundred_tournament import HundredTournament

class TrainGeneticAgents():
    def __init__(self, num_features=11):
        self.num_features = num_features
        self.num_decimals = 3
        self.num_agents = 100

        # ----------- Start Training ----------- #
        self.initial_weights = self.generate_initial_weights()
        tournament1 = HundredTournament(self.initial_weights, 9, 6, 7)
        tournament1.play_tournament()
        return

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
    

if __name__ == '__main__':
    TrainGeneticAgents(11)