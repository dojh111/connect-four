'''
The GeneticAgent class is an implementation for an agent to play connect4
The agent will choose moves by calculating the values of each column, and selecting the column with the highest value

Genetic algorithm will be used to determine how we weigh the importance of each feature, to generate the values of each action

player_number: integer, either 1 or 2
feature_weights: list/array of values of weights for each
'''
class GeneticAgent:
    def __init__(self, player_number, feature_weights, board_height=6, board_width=7):
        self.player_number = player_number          # Player number corresponds to board token
        self.opponent_number = 0
        # ------ Set Opponent Number ------ #
        if self.player_number == 1:
            self.opponent_number = 2
        elif self.player_number == 2:
            self.opponent_number = 1
        self.feature_weights = feature_weights
        # ------ Board indexes ------ #
        self.bottom_index = board_height - 1        # Bottom index = largest number
        self.top_index = 0
        self.rightmost_index = board_width - 1
        self.leftmost_index = 0
    
    # --------- Calculate score of each column, and select best --------- #
    def selectAction(self, board, actions):
        # Available actions are non -1
        features = []
        column_number = 0
        for row_index in actions:
            if row_index == -1: # Invalid action, skip
                column_number += 1
                continue
            # Add feature to see if winning move
            features.append(self.check_if_winning_move(board=board, selected_column=column_number, selected_row=row_index))
            column_number += 1

        print('[FEATURE EXTRACTION] Extracted features: ')
        print(features)
        return 0

    # ------------ Check for winning move ------------ #
    def check_if_winning_move(self, board, selected_column, selected_row):
        if self.check_horizontal_win(board=board, selected_column=selected_column, selected_row=selected_row) == 4:
            return 1
        elif self.check_vertical_win(board=board, selected_column=selected_column, selected_row=selected_row) == 4:
            return 1
        elif self.check_diagonalone_win(board=board, selected_column=selected_column, selected_row=selected_row) == 4:
            return 1
        elif self.check_diagonaltwo_win(board=board, selected_column=selected_column, selected_row=selected_row) == 4:
            return 1
        return 0

    def check_if_blocking_move(self, board, selected_column, selected_row):
        return

    def check_if_row_of_two(self):
        return

    def check_if_row_of_three(self):
        return

    def check_if_middle_column(self):
        return

    def check_if_edge_column(self):
        return

    def check_if_other_column(self):
        return

    def check_if_inevitable_horizontal_win(self):
        return

    # ------------ Tokens in direction count functions: Continues when encountering blank spaces ------------ #

    # Calculates the number of unblocked tokens to left - Can include enoty spaces
    def count_tokens_to_left(self, player_number, board, selected_column, selected_row):
        count = 0
        if selected_column == self.leftmost_index:
            return count
        curr_column = selected_column - 1
        while curr_column >= self.leftmost_index:
            if board[selected_row][curr_column] == player_number:
                count += 1
                curr_column -= 1
            elif board[selected_row][curr_column] == 0: # Empty space
                curr_column -= 1
            else: # Is opponent tile
                return count
        return count

    def count_tokens_to_right(self, player_number, board, selected_column, selected_row):
        count = 0
        if selected_column == self.rightmost_index:
            return count
        curr_column = selected_column + 1
        while curr_column <= self.rightmost_index:
            if board[selected_row][curr_column] == player_number:
                count += 1
                curr_column += 1
            elif board[selected_row][curr_column] == 0: # Empty space
                curr_column += 1
            else: # Is opponent tile
                return count
        return count

    '''
    Due to logic of gravity, can never have empty spaces below - Only own, or opponent tokens
    Extended, we don't have to check upwards to count number of tokens
    '''
    def count_tokens_to_bottom(self, player_number, board, selected_column, selected_row):
        count = 0
        if selected_row == self.bottom_index:
            return count
        curr_row = selected_row + 1
        while curr_row <= self.bottom_index:
            if board[curr_row][selected_column] == player_number:
                count += 1
                curr_row += 1
            else: # Is opponent tile
                return count
        return count

    def count_tokens_to_diagonal_topleft(self, player_number, board, selected_column, selected_row):
        count = 0
        if selected_row == self.top_index or selected_column == self.leftmost_index: # Either at leftmost, or top of board
            return count
        curr_row = selected_row - 1
        curr_column = selected_column - 1
        while curr_row >= self.top_index and curr_column >= self.leftmost_index:
            if board[curr_row][curr_column] == player_number:
                count += 1      # Increment count
                curr_row -= 1
                curr_column -= 1
            elif board[curr_row][curr_column] == 0: # Empty space
                curr_row -= 1
                curr_column -= 1
            else: # Is opponent tile
                return count
        return count

    def count_tokens_to_diagonal_topright(self, player_number, board, selected_column, selected_row):
        count = 0
        if selected_row == self.top_index or selected_column == self.rightmost_index: # Either at leftmost, or top of board
            return count
        curr_row = selected_row - 1
        curr_column = selected_column + 1
        while curr_row >= self.top_index and curr_column <= self.rightmost_index:
            if board[curr_row][curr_column] == player_number:
                count += 1      # Increment count
                curr_row -= 1
                curr_column += 1
            elif board[curr_row][curr_column] == 0: # Empty space
                curr_row -= 1
                curr_column += 1
            else: # Is opponent tile
                return count
        return count

    def count_tokens_to_diagonal_botleft(self, player_number, board, selected_column, selected_row):
        count = 0
        return count

    def count_tokens_to_diagonal_botright(self, player_number, board, selected_column, selected_row):
        count = 0
        return count

    # --------------- Winning move calculation functions: Must be CONSECUTIVE ---------------- #

    # Checks to the left and right of last placed piece to determine if 4 in a row has been achieved
    def check_horizontal_win(self, board, selected_column, selected_row):
        left_count = 0
        right_count = 0
        player_token = self.player_number
        # Calculate number of pieces to left
        if selected_column != self.leftmost_index:
            curr_column = selected_column - 1
            while curr_column >= self.leftmost_index:
                if board[selected_row][curr_column] == player_token:
                    left_count += 1
                else:
                    break # Not continuous matching token
                curr_column -= 1
        # Calculate number of pieces to right
        if selected_column != self.rightmost_index:
            curr_column = selected_column + 1
            while curr_column <= self.rightmost_index:
                if board[selected_row][curr_column] == player_token:
                    right_count += 1
                else:
                    break # Not continuous matching token
                curr_column += 1
        return 1 + left_count + right_count
    
    # Checks to the top and bottom of last placed piece to determine if 4 in a row has been achieved
    def check_vertical_win(self, board, selected_column, selected_row):
        up_count = 0
        down_count = 0
        player_token = self.player_number
        # Calculate number of pieces to bottom
        if selected_row != 5:
            curr_row = selected_row + 1
            while curr_row <= self.bottom_index:
                if board[curr_row][selected_column] == player_token:
                    down_count += 1
                else:
                    break # Not continuous matching token
                curr_row += 1
        # Calculate number of pieces to top
        if selected_row != 0:
            curr_row = selected_row - 1
            while curr_row >= self.top_index:
                if board[curr_row][selected_column] == player_token:
                    up_count += 1
                else:
                    break # Not continuous matching token
                curr_row -= 1
        return 1 + down_count + up_count

    def check_diagonalone_win(self, board, selected_column, selected_row):
        diagonal_down_right_count = 0
        diagonal_up_left_count = 0
        player_token = self.player_number
        # Calculate pieces to diagonal bottom right of current piece
        if selected_column != self.rightmost_index and selected_row != self.bottom_index:   # Ensure there are still spaces to right, and bottom
            curr_row = selected_row + 1
            curr_column = selected_column + 1
            while curr_row <= self.bottom_index and curr_column <= self.rightmost_index:
                if board[curr_row][curr_column] == player_token:
                    diagonal_down_right_count += 1
                else:
                    break # Not continuous matching token
                curr_row += 1
                curr_column += 1
        # Calculate pieces to diagonal top left of current piece
        if selected_column != 0 and selected_row != 0: # Ensure there are still spaces to left, and top
            curr_row = selected_row - 1
            curr_column = selected_column - 1
            while curr_row >= self.top_index and curr_column >= self.leftmost_index:
                if board[curr_row][curr_column] == player_token:
                    diagonal_up_left_count += 1
                else:
                    break # Not continuous matching token
                curr_row -= 1
                curr_column -= 1
        return 1 + diagonal_down_right_count + diagonal_up_left_count

    def check_diagonaltwo_win(self, board, selected_column, selected_row):
        diagonal_down_left_count = 0
        diagonal_up_right_count = 0
        player_token = self.player_number
        # Calculate pieces to diagonal bottom left of current piece
        if selected_column != self.leftmost_index and selected_row != self.bottom_index:   # Ensure there are still spaces to left, and bottom
            curr_row = selected_row + 1
            curr_column = selected_column - 1
            while curr_row <= self.bottom_index and curr_column >= self.leftmost_index:
                if board[curr_row][curr_column] == player_token:
                    diagonal_down_left_count += 1
                else:
                    break # Not continuous matching token
                curr_row += 1
                curr_column -= 1
        # Calculate pieces to diagonal top right of current piece
        if selected_column != self.rightmost_index and selected_row != self.top_index: # Ensure there are still spaces to right, and top
            curr_row = selected_row - 1
            curr_column = selected_column - 1
            while curr_row >= self.top_index and curr_column <= self.rightmost_index:
                if board[curr_row][curr_column] == player_token:
                    diagonal_up_right_count += 1
                else:
                    break # Not continuous matching token
                curr_row -= 1
                curr_column += 1
        return 1 + diagonal_down_left_count + diagonal_up_right_count