import numpy as np

'''
ConnectFour is a data structure to represent the state of the playing environment
Pieces for each player is defined by the integers: 1 for player 1 and 2 for player 2
'''
class ConnectFour:
    # Constructor
    def __init__(self):
        print('[NEW GAME] Starting new connect 4 game')
        self.game_state = np.zeros((6,7))
        # Initialise all available positions for pieces to land on at start of game
        self.available_actions = [(5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6)]
        # Tokens we will use to differentiate players
        self.player_1_token = 1
        self.player_2_token = 2
        # Start with turn for player 1
        self.player_turn = 1
        self.is_done = False
        self.turn_count = 1
        self.start_game()

    def start_game(self):
        while not self.is_done:
            print(f'[TURN {str(self.turn_count)}] Player {str(self.player_turn)}\'s turn:')
            print('======= GAME STATE =======')
            print(self.game_state)
            print('==========================')

            player_selection = self.get_player_selection()
            self.update_game_state(player_selection)

            self.turn_count += 1
            if self.player_turn == 1:
                self.player_turn = 2
            else:
                self.player_turn = 1
        return

    # Get from player which column to add token to
    def get_player_selection(self):
        while True:
            try:
                available_columns = []
                for action in self.available_actions:
                    available_columns.append(action[1])
                print(f'Available columns: {str(available_columns)}')
                selection = input('Select column --> ')
                selection = int(selection)
                if selection not in available_columns:
                    raise Exception('Invalid column selected')
                # Select the correct action from the remaining list
                index = 0
                for action in self.available_actions:
                    if action[1] == selection:
                        return index
                    index += 1
            except Exception as e:
                print('[ERROR] An error has occurred: ' + str(e))
                print('[ERROR] Please input a valid column')

    # Update the game with the newly added token for the player, and update new available actions
    def update_game_state(self, player_selection):
        player_token = self.player_1_token
        if self.player_turn == 2:
            player_token = self.player_2_token
        token_coordinate = self.available_actions[player_selection]
        print(token_coordinate)
        y = token_coordinate[0]
        x = token_coordinate[1]
        # Set the board as the player's token
        self.game_state[y][x] = player_token
        self.update_available_actions(player_selection)

    # Update the new coordinate according to where the piece was placed
    def update_available_actions(self, player_selection):
        token_coordinate = self.available_actions[player_selection]
        if token_coordinate[0] == 0:
            # Remove coordinate as at top of board - Stop here as no more actions
            del self.available_actions[player_selection]
            return
        new_action = (token_coordinate[0] - 1, token_coordinate[1])
        self.available_actions[player_selection] = new_action
        return

    # Returns all available places that a piece can go in current state
    def get_available_actions(self):
        return self.available_actions
    
    def get_state(self):
        return self.game_state

if __name__ == '__main__':
    ConnectFour()