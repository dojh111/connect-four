from re import L
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
        self.available_actions = [5, 5, 5, 5, 5, 5, 5]
        # Start with turn for player 1
        self.player_turn = 1
        self.is_done = False
        self.is_draw = False
        self.turn_count = 1
        self.start_game()

    def start_game(self):
        while True:
            print()
            print(f'[TURN {str(self.turn_count)}] Player {str(self.player_turn)}\'s turn:')
            self.print_board()
            # Get player action
            player_selection = self.get_player_selection()
            self.update_game_state(player_selection)
            # Game has terminated
            if self.is_done:
                print(f'[GAME OVER] Player {str(self.player_turn)} won!')
                return
            elif self.is_draw:
                print(f'[GAME OVER] Draw! - No more turns remaining')
                return
            self.turn_count += 1
            if self.player_turn == 1:
                self.player_turn = 2
            else:
                self.player_turn = 1
            print('[Available Actions]')
            print(self.get_available_actions())

    def print_board(self):
        if self.is_done:
            print('------- GAME OVER ------')
        elif self.is_draw:
            print('--------- DRAW ---------')
        else:
            print('------ GAME STATE ------')
        print(self.game_state)
        print('========= ROW ==========')
        print('  0  1  2  3  4  5  6   ')
        print('------------------------')

    # Checks to the left and right of last placed piece to determine if 4 in a row has been achieved
    def calculate_horizontal_length(self, selected_column):
        left_count = 0
        right_count = 0
        row = self.available_actions[selected_column]
        player_token = self.player_turn
        # Calculate number of pieces to left
        if selected_column != 0:
            curr_column = selected_column - 1
            while curr_column >= 0:
                if self.game_state[row][curr_column] == player_token:
                    left_count += 1
                else:
                    break # Not continuous matching token
                curr_column -= 1
        # Calculate number of pieces to right
        if selected_column != 6:
            curr_column = selected_column + 1
            while curr_column <= 6:
                if self.game_state[row][curr_column] == player_token:
                    right_count += 1
                else:
                    break # Not continuous matching token
                curr_column += 1
        return 1 + left_count + right_count
    
    # Checks to the top and bottom of last placed piece to determine if 4 in a row has been achieved
    def calculate_vertical_length(self, selected_column):
        up_count = 0
        down_count = 0
        row = self.available_actions[selected_column]
        player_token = self.player_turn
        # Calculate number of pieces to bottom
        if row != 5:
            curr_row = row + 1
            while curr_row <= 5:
                if self.game_state[curr_row][selected_column] == player_token:
                    down_count += 1
                else:
                    break # Not continuous matching token
                curr_row += 1
        # Calculate number of pieces to top
        if row != 0:
            curr_row = row - 1
            while curr_row >= 0:
                if self.game_state[curr_row][selected_column] == player_token:
                    up_count += 1
                else:
                    break # Not continuous matching token
                curr_row -= 1
        return 1 + down_count + up_count

    '''
    Calculates diagonal from top left down to bottom right
    Example:
    1 0 0 0 
    0 1 0 0 
    0 0 1 0 
    0 0 0 1
    '''
    def calculate_diagonal_one(self, selected_column):
        diagonal_down_right_count = 0
        diagonal_up_left_count = 0
        selected_row = self.available_actions[selected_column]
        player_token = self.player_turn
        # Calculate pieces to diagonal bottom right of current piece
        if selected_column != 6 and selected_row != 5:   # Ensure there are still spaces to right, and bottom
            curr_row = selected_row + 1
            curr_column = selected_column + 1
            while curr_row <= 5 and curr_column <= 6:
                if self.game_state[curr_row][curr_column] == player_token:
                    diagonal_down_right_count += 1
                else:
                    break # Not continuous matching token
                curr_row += 1
                curr_column += 1
        # Calculate pieces to diagonal top lef of current piece
        if selected_column != 0 and selected_row != 0: # Ensure there are still spaces to left, and top
            curr_row = selected_row - 1
            curr_column = selected_column - 1
            while curr_row >= 0 and curr_column >= 0:
                if self.game_state[curr_row][curr_column] == player_token:
                    diagonal_up_left_count += 1
                else:
                    break # Not continuous matching token
                curr_row -= 1
                curr_column -= 1
        return 1 + diagonal_down_right_count + diagonal_up_left_count

    '''
    Check if the game has ended

    We will only need to check 8 directions from when the last piece was placed for the player
    '''
    def check_if_game_done(self, selected_column):
        # Check horizontals
        if self.calculate_horizontal_length(selected_column) == 4:
            self.is_done = True
            self.print_board()
            print(f'[GAME TERMINATION ENGINE] Player {str(self.player_turn)} won by HORIZONTAL')
            return True
        # Check verticals
        elif self.calculate_vertical_length(selected_column) == 4:
            self.is_done = True
            self.print_board()
            print(f'[GAME TERMINATION ENGINE] Player {str(self.player_turn)} won by VERTICAL')
            return True
        # Check diagonal 1 - Top left to bottom right
        elif self.calculate_diagonal_one(selected_column) == 4:
            self.is_done = True
            self.print_board()
            print(f'[GAME TERMINATION ENGINE] Player {str(self.player_turn)} won by DIAGONAL 1: Top Left to Bottom Right')
            return True
        return False

    # Get from player which column to add token to
    def get_player_selection(self):
        while True:
            try:
                print(f'Available columns: {str(self.available_actions)}')
                selection = input('Select column [0-6] which is not -1: ')
                selection = int(selection)
                if self.available_actions[selection] == -1:
                    raise Exception('Invalid column selected')
                return selection
            except Exception as e:
                print('[ERROR] An error has occurred: ' + str(e))
                print('[ERROR] Please input a valid column')

    def check_for_draw(self):
        # Check if all actions are -1
        for action in self.available_actions:
            if action >= 0:
                return
        # Draw has happened
        self.is_draw = True
        self.print_board()
        return


    # Update the game with the newly added token for the player, and update new available actions
    def update_game_state(self, selected_column):
        y = self.available_actions[selected_column]
        x = selected_column
        # Set the board as the player's token
        self.game_state[y][x] = self.player_turn
        if self.check_if_game_done(selected_column):
            return
        self.update_available_actions(selected_column)
        self.check_for_draw()
        return

    # Update the new coordinate according to where the piece was placed
    def update_available_actions(self, selected_column):
        column_height = self.available_actions[selected_column]
        self.available_actions[selected_column] = column_height - 1
        return

    # Returns all available places that a piece can go in current state
    def get_available_actions(self):
        return self.available_actions
    
    def get_state(self):
        return self.game_state

if __name__ == '__main__':
    ConnectFour()