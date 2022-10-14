from os import system, name
import numpy as np

'''
ConnectFour is a data structure to represent the state of the playing environment
Pieces for each player is defined by the integers: 1 for player 1 and 2 for player 2
'''
class ConnectFour:
    # Constructor
    def __init__(self, agent_player_number):
        print('[NEW GAME] Starting new connect 4 game')
        self.game_state = np.zeros((6,7))
        # Initialise all available positions for pieces to land on at start of game
        self.available_actions = [5, 5, 5, 5, 5, 5, 5]
        self.player_turn = 1
        self.is_done = False
        self.is_draw = False
        self.turn_count = 1
        self.agent_player_number = agent_player_number  # Determines if agent will be player 1 or 2
        self.agent_game_outcome = 0    # Outcome for agent - 0 = Draw, -1 = Lose, 1 = Win 
        # self.start_game()     # Uncomment to start game for human player - For testing and debugging purposes

    # Clears the terminal screen to reduce clutter
    def clear(self):
        # for windows
        if name == 'nt':
            _ = system('cls')
        # for mac and linux(here, os.name is 'posix')
        else:
            _ = system('clear')

    '''
    Main function for AI agents to play game

    Returns True if action is valid, False if chosen action is not valid
    '''
    def play_turn(self, selected_column):
        # Check if selected column is invalid
        if self.available_actions[selected_column] == -1:
            print('[INVALID ACTION] Invalid action selected, please try again')
            return False
        is_terminated = self.update_game_state(selected_column)
        # Game has terminated
        if is_terminated:
            if self.agent_player_number == self.player_turn:
                print(f'[GAME OVER] Player {str(self.player_turn)}, AGENT has won!')
                self.agent_game_outcome = 1     # Agent has won the game
                return self.agent_game_outcome
            else:
                print(f'[GAME OVER] Player {str(self.player_turn)}, AI has won!')
                self.agent_game_outcome = -1    # Agent has lost the game
                return self.agent_game_outcome
        # Game continues
        self.turn_count += 1
        if self.turn_count == 42:   # Board has been filled
            self.is_draw = True
            self.is_done = True
            print(f'[GAME OVER] Draw! - No more turns remaining')
            return self.agent_game_outcome
        # Set player turn
        if self.player_turn == 1:
            self.player_turn = 2
        else:
            self.player_turn = 1
        return True

    '''
    Starts the main game loop - For testing purposes
    '''
    def start_game(self):
        while True:
            self.clear()
            # Main game loop
            player_selection = -1
            self.print_board()
            if self.player_turn == self.agent_player_number:    # Action selection for our AI agent
                print(f'[TURN {str(self.turn_count)} - AGENT] Player {str(self.player_turn)}\'s turn:')
                player_selection = self.get_player_selection()
            else:                                               # Actions selection for the opponent player
                print(f'[TURN {str(self.turn_count)} - AI] Player {str(self.player_turn)}\'s turn:')
                player_selection = self.get_player_selection()
            is_terminated = self.update_game_state(player_selection)
            # Game has terminated
            if is_terminated:
                if self.agent_player_number == self.player_turn:
                    print(f'[GAME OVER] Player {str(self.player_turn)}, AGENT has won!')
                    self.agent_game_outcome = 1     # Agent has won the game
                    return self.agent_game_outcome
                else:
                    print(f'[GAME OVER] Player {str(self.player_turn)}, AI has won!')
                    self.agent_game_outcome = -1    # Agent has lost the game
                    return self.agent_game_outcome
            # Game continues
            self.turn_count += 1
            if self.turn_count == 42:   # Board has been filled
                self.is_draw = True
                self.is_done = True
                print(f'[GAME OVER] Draw! - No more turns remaining')
                return self.agent_game_outcome
            # Set player turn
            if self.player_turn == 1:
                self.player_turn = 2
            else:
                self.player_turn = 1
    
    '''
    Prompts a human player for a input to put valid column to put their tile
    '''
    def get_player_selection(self):
        while True:
            try:
                print(f'Available Actions: {str(self.available_actions)}')
                selection = input('Select column [0-6]: ')
                selection = int(selection)
                if self.available_actions[selection] == -1:
                    raise Exception('Invalid column selected')
                return selection
            except Exception as e:
                print('[ERROR] An error has occurred: ' + str(e))
                print('[ERROR] Please input a valid column')

    # Prints out the current board
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
        # Calculate pieces to diagonal top left of current piece
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
    Calculates diagonal from top right down to bottom left
    Example:
    0 0 0 1
    0 0 1 0
    0 1 0 0
    1 0 0 0
    '''
    def calculate_diagonal_two(self, selected_column):
        diagonal_down_left_count = 0
        diagonal_up_right_count = 0
        selected_row = self.available_actions[selected_column]
        player_token = self.player_turn
        # Calculate pieces to diagonal bottom left of current piece
        if selected_column != 0 and selected_row != 5:   # Ensure there are still spaces to left, and bottom
            curr_row = selected_row + 1
            curr_column = selected_column - 1
            while curr_row <= 5 and curr_column >= 0:
                if self.game_state[curr_row][curr_column] == player_token:
                    diagonal_down_left_count += 1
                else:
                    break # Not continuous matching token
                curr_row += 1
                curr_column -= 1
        # Calculate pieces to diagonal top right of current piece
        if selected_column != 6 and selected_row != 0: # Ensure there are still spaces to right, and top
            curr_row = selected_row - 1
            curr_column = selected_column - 1
            while curr_row >= 0 and curr_column <= 6:
                if self.game_state[curr_row][curr_column] == player_token:
                    diagonal_up_right_count += 1
                else:
                    break # Not continuous matching token
                curr_row -= 1
                curr_column += 1
        return 1 + diagonal_down_left_count + diagonal_up_right_count

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
        elif self.calculate_diagonal_two(selected_column) == 4:
            self.is_done = True
            self.print_board()
            print(f'[GAME TERMINATION ENGINE] Player {str(self.player_turn)} won by DIAGONAL 2: Top Right to Bottom Left')
            return True
        return False

    # Update the game with the newly added token for the player, and update new available actions
    def update_game_state(self, selected_column):
        y = self.available_actions[selected_column]
        x = selected_column
        # Set the board as the player's token
        self.game_state[y][x] = self.player_turn
        if self.check_if_game_done(selected_column):
            return True
        self.update_available_actions(selected_column)
        return False

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

    def get_turn_number(self):
        return self.turn_count
