import random
from geneticAgent import GeneticAgent
from ConnectFourEnv.geneticConnectFour import GeneticConnectFour

'''
The HundredTournament class generates a tournament for 100 players
Knockout style tournament is used. In total 99 games will be played in 1 tournament

contestant_weights: 2D array of 100 rows, each row being the weights of each participant
best_of: Number of rounds to play for each matchup between 2 players

-------- TOURNAMENT STYLE --------
Round 1 - 28 players get a bye, i.e. straight entry to round 2 and 72 players play 36 matches in which 36 win and progress to the next round.

Round 2 - no of participants 64 (28 with direct entry to round 2 and 36 that won in round 1), no of matches 32

Round 3 - 32 participants play in 16 matches, 16 progress to next round

Round 4 - 16 participants play in 8 matches, 8 progress to next round

Round 5 - 8 participants play in 4 matches, 4 progress to next round

Round 6 - 4 participants play 2 matches, 2 progress to next round/ finals.

Finals - 2 participants play 1 match and the winner is determined.
'''
class HundredTournament():
    def __init__(self, contestant_weights=None, best_of=9, board_height=6, board_width=7):
        self.contestant_weights = contestant_weights
        self.best_of = best_of
        self.board_height = board_height
        self.board_width = board_width
        self.round_number = 2

        # Participants for each round
        self.round_participants = {
            1: [], # Participate in round 1: 72 Players
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [], # Finals round
            "bye_participants": [],  # Go straight into round 2: 28 Player
            "winner": []
        }
        self.round_four_losers = []
        # Initialise round 1 participants
        self.select_round_one_participants()

    def get_best_of_losers(self):
        index = 0
        max_num_wins = 0
        best_loser_index = 0
        for player in self.round_four_losers:
            if player[1] > max_num_wins:
                max_num_wins = player[1]
                best_loser_index = index
            index += 1
        return self.round_four_losers.pop(best_loser_index)

    # Plays the entire tournament out and returns weights of top 10 performers
    def play_tournament(self):
        best_weights = []
        self.play_round_one()
        for i in range(2, 8):
            self.play_remaining_rounds(i)
        # print('TOURNAMENT OVER')
        print(f'Winner weights: {str(self.contestant_weights[self.round_participants["winner"][0]])}')
        print(self.round_participants["winner"])
        # print(self.round_participants[7])
        # print(self.round_participants[6])
        print(self.round_participants[5])

        # print('Getting top 8 player weights...')
        for participant in self.round_participants[5]:
            best_weights.append(self.contestant_weights[participant])
        # print('Getting best of loser weights...')
        # print(self.round_four_losers)
        for i in range(0, 2):
            best_loser = self.get_best_of_losers()
            best_weights.append(self.contestant_weights[best_loser[0]])
        return best_weights

    # Plays a match between 2 players - Best of x
    def play_match(self, player_one_index, player_one_weights, player_two_index, player_two_weights):
        player_one = GeneticAgent(1, player_one_weights, self.board_height, self.board_width)
        player_two = GeneticAgent(2, player_two_weights, self.board_height, self.board_width)
        player_one_wins = 0
        player_one_losses = 0
        player_two_wins = 0
        player_two_losses = 0
        draws = 0
        for i in range(0, self.best_of):
            # Start a new game
            connect_four = GeneticConnectFour(self.board_height, self.board_width)
            game_result = -1
            while not connect_four.is_done:
                available_actions = connect_four.get_available_actions()
                turn_number = connect_four.get_turn_number()
                game_state = connect_four.get_state()
                if turn_number % 2 == 1:
                    best_action = player_one.selectAction(game_state, available_actions)
                    game_result = connect_four.play_turn(best_action)
                # Is Random AI turn
                elif turn_number % 2 == 0:
                    best_action = player_two.selectAction(game_state, available_actions)
                    game_result = connect_four.play_turn(best_action)
            # Game is over, tally results
            if game_result == 1:    # Player 1 won
                player_one_wins += 1
                player_two_losses += 1
            elif game_result == 2:  # Player 2 won
                player_two_wins += 1
                player_one_losses += 1
            elif game_result == 0:  # Draw
                draws += 1
            else:                   # Game result = -1, error has occured
                print('[AN ERROR HAS OCCURRED] No game result available')
        # All 9 matches over
        player_one_stats = {
            "index": player_one_index,
            "wins": player_one_wins,
            "losses": player_one_losses,
            "draws": draws
        }
        player_two_stats = {
            "index": player_two_index,
            "wins": player_two_wins,
            "losses": player_two_losses,
            "draws": draws
        }
        return [player_one_stats, player_two_stats]

    def play_remaining_rounds(self, round_number):
        print(f' ------------ ROUND {str(round_number)} ------------')
        # print(self.round_participants[round_number])
        # print(f'NUMBER OF PARTICIPANTS: {str(len(self.round_participants[round_number]))}')
        # Main tournament logic
        match_number = 1
        for i in range(0, len(self.round_participants[round_number]), 2):  # Stop 1 prior
            # print(f'[MATCH NUMBER] {str(match_number)}')
            contestant_one = self.round_participants[round_number][i]
            contestant_two = self.round_participants[round_number][i + 1]
            contestant_one_weights = self.contestant_weights[contestant_one]
            contestant_two_weights = self.contestant_weights[contestant_two]
            # print(f'[CONTESTANT] {str(contestant_one)}, {str(contestant_two)}')
            # print(contestant_one_weights)
            # print(contestant_two_weights)
            results = self.play_match(contestant_one, contestant_one_weights, contestant_two, contestant_two_weights)
            # Take winner of match, and add to round 2 contestants
            player_one_wins = results[0]["wins"]
            player_two_wins = results[1]["wins"]
            if round_number < 7:
                if player_one_wins >= player_two_wins: # If draw, just take player 1
                    print(f'PLAYER {str(contestant_one)} won {str(player_one_wins)}:{str(player_two_wins)}')
                    self.round_participants[round_number + 1].append(contestant_one)
                    if round_number == 4:   # Append loser wins
                        self.round_four_losers.append((contestant_two, player_two_wins))
                elif player_two_wins > player_one_wins:
                    print(f'PLAYER {str(contestant_two)} won {str(player_two_wins)}:{str(player_one_wins)}')
                    self.round_participants[round_number + 1].append(contestant_two)
                    if round_number == 4:   # Append loser wins
                        self.round_four_losers.append((contestant_one, player_one_wins))
            else:   # Is final round
                if player_one_wins >= player_two_wins: # If draw, just take player 1
                    print(f'PLAYER {str(contestant_one)} won {str(player_one_wins)}:{str(player_two_wins)}')
                    self.round_participants["winner"].append(contestant_one)
                elif player_two_wins > player_one_wins:
                    print(f'PLAYER {str(contestant_two)} won {str(player_two_wins)}:{str(player_one_wins)}')
                    self.round_participants["winner"].append(contestant_two)
        match_number += 1
        print(f' ------------ ROUND OVER ------------ ')
        return

    # Randomly selects 72 players to play round 1
    def select_round_one_participants(self):
        participant_list = range(0, 100)
        self.round_participants[1] = random.sample(participant_list, 72)
        for i in range(0, 100):
            if i not in self.round_participants[1]:
                self.round_participants["bye_participants"].append(i)
        return

    '''
    Go through list of 72 participants and make them compete
    '''
    def play_round_one(self):
        print(' ------------ ROUND 1 ------------')
        # print(self.round_participants[1])
        match_number = 1
        for i in range(0, 72, 2):  # Stop 1 prior
            # print(f'[MATCH NUMBER] {str(match_number)}')
            contestant_one = self.round_participants[1][i]
            contestant_two = self.round_participants[1][i + 1]
            contestant_one_weights = self.contestant_weights[contestant_one]
            contestant_two_weights = self.contestant_weights[contestant_two]
            print(f'[CONTESTANTS] {str(contestant_one)}, {str(contestant_two)}')
            # print(contestant_one_weights)
            # print(contestant_two_weights)
            results = self.play_match(contestant_one, contestant_one_weights, contestant_two, contestant_two_weights)
            # Take winner of match, and add to round 2 contestants
            player_one_wins = results[0]["wins"]
            player_two_wins = results[1]["wins"]
            if player_one_wins >= player_two_wins: # If draw, just take player 1
                print(f'PLAYER {str(contestant_one)} won {str(player_one_wins)}:{str(player_two_wins)}')
                self.round_participants[2].append(contestant_one)
                if len(self.round_participants["bye_participants"]) > 0:
                    bye_participant = self.round_participants["bye_participants"].pop(0)
                    self.round_participants[2].append(bye_participant)
                    # print(f'[BYE PARTICIPANT INSERTION] Player number: {str(bye_participant)}')
            elif player_two_wins > player_one_wins:
                print(f'PLAYER {str(contestant_two)} won {str(player_two_wins)}:{str(player_one_wins)}')
                self.round_participants[2].append(contestant_two)
                if len(self.round_participants["bye_participants"]) > 0:
                    bye_participant = self.round_participants["bye_participants"].pop(0)
                    self.round_participants[2].append(bye_participant)
                    # print(f'[BYE PARTICIPANT INSERTION] Player number: {str(bye_participant)}')
            match_number += 1
        print(' ------------ ROUND 1 OVER ------------')
        return