# col, minimax_score = minimax(board, 5, -math.inf, math.inf, True)

# 		if is_valid_location(board, col):
# 			#pygame.time.wait(500)
# 			row = get_next_open_row(board, col)
# 			drop_piece(board, row, col, self.ai_number)
import math
import random

WINNING_TOKEN_COUNT = 4

class MinimaxAgent:
    def __init__(self, ai_number,board_height=6, board_width=7):
        self.ai_number = ai_number          # Player number corresponds to board token
        self.opponent_number = 0
        # ------ Set Opponent Number ------ #
        if self.ai_number == 1:
            self.opponent_number = 2
        elif self.ai_number == 2:
            self.opponent_number = 1
        # ------ Board indexes ------ #
        self.board_height = board_height
        self.board_width = board_width
        self.bottom_index = self.board_height - 1        # Bottom index = largest number
        self.top_index = 0
        self.rightmost_index = self.board_width - 1
        self.leftmost_index = 0
        self.middle_column_index = math.floor(self.board_width / 2)      # Board should always be an odd number width
    
    def winning_move(self,board, piece):
        # Check horizontal locations for win
        for c in range(self.board_width-3):
            for r in range(self.board_height):
                if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                    return True

        # Check vertical locations for win
        for c in range(self.board_width):
            for r in range(self.board_height-3):
                if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                    return True

        # Check positively sloped diaganols
        for c in range(self.board_width-3):
            for r in range(self.board_height-3):
                if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                    return True

        # Check negatively sloped diaganols
        for c in range(self.board_width-3):
            for r in range(3, self.board_height):
                if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                    return True

    def evaluate_window(self,window, piece):
        score = 0
        opp_piece = self.opponent_number
        if piece == self.opponent_number:
            opp_piece = self.ai_number

        if window.count(piece) == 4:
            score += 100
        elif window.count(piece) == 3 and window.count(0) == 1:
            score += 5
        elif window.count(piece) == 2 and window.count(0) == 2:
            score += 2

        if window.count(opp_piece) == 3 and window.count(0) == 1:
            score -= 4

        return score

    def score_position(self,board, piece):
        score = 0

        ## Score center column
        center_array = [int(i) for i in list(board[:, self.board_width//2])]
        center_count = center_array.count(piece)
        score += center_count * 3

        ## Score Horizontal
        for r in range(self.board_height):
            row_array = [int(i) for i in list(board[r,:])]
            for c in range(self.board_width-3):
                window = row_array[c:c+WINNING_TOKEN_COUNT]
                score += self.evaluate_window(window, piece)

        ## Score Vertical
        for c in range(self.board_width):
            col_array = [int(i) for i in list(board[:,c])]
            for r in range(self.board_height-3):
                window = col_array[r:r+WINNING_TOKEN_COUNT]
                score += self.evaluate_window(window, piece)

        ## Score posiive sloped diagonal
        for r in range(self.board_height-3):
            for c in range(self.board_width-3):
                window = [board[r+i][c+i] for i in range(WINNING_TOKEN_COUNT)]
                score += self.evaluate_window(window, piece)

        for r in range(self.board_height-3):
            for c in range(self.board_width-3):
                window = [board[r+3-i][c+i] for i in range(WINNING_TOKEN_COUNT)]
                score += self.evaluate_window(window, piece)

        return score

    def is_terminal_node(self,board):
        return self.winning_move(board, self.opponent_number) or self.winning_move(board, self.ai_number) or len(self.get_valid_locations(board)) == 0

    def minimax(self,board, depth, alpha, beta, maximizingPlayer):
        valid_locations = self.get_valid_locations(board)
        is_terminal = self.is_terminal_node(board)
        if depth == 0 or is_terminal:
            if is_terminal:
                if self.winning_move(board, self.ai_number):
                    return (None, 100000000000000)
                elif self.winning_move(board, self.opponent_number):
                    return (None, -10000000000000)
                else: # Game is over, no more valid moves
                    return (None, 0)
            else: # Depth is zero
                return (None, self.score_position(board, self.ai_number))
        if maximizingPlayer:
            value = -math.inf
            column = random.choice(valid_locations)
            print('valid_locations:',valid_locations, 'column:',column)
            for col in valid_locations:
                row = self.get_next_open_row(board, col)
                b_copy = board.copy()
                self.drop_piece(b_copy, row, col, self.ai_number)
                new_score = self.minimax(b_copy, depth-1, alpha, beta, False)[1]
                
                if new_score > value:
                    value = new_score
                    column = col
                alpha = max(alpha, value)
                
                if alpha >= beta:
                    print("alpha:",alpha, "beta:",beta)
                    break
            print("column:",column,"value:", value, "alpha:",alpha, "beta:",beta)
            return column, value

        else: # Minimizing player
            value = math.inf
            column = random.choice(valid_locations)
            for col in valid_locations:
                row = self.get_next_open_row(board, col)
                b_copy = board.copy()
                self.drop_piece(b_copy, row, col, self.opponent_number)
                new_score = self.minimax(b_copy, depth-1, alpha, beta, True)[1]
                if new_score < value:
                    value = new_score
                    column = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return column, value

    def get_valid_locations(self,board):
        valid_locations = []
        
        for col in range(self.board_width):
            
            if self.is_valid_location(board, col):
                valid_locations.append(col)
        return valid_locations

    def is_valid_location(self, board, col):
        return board[0][col] == 0

    def get_next_open_row(self, board, col):
        for r in range(self.board_height):
            if board[self.board_height-1-r][col] == 0:
                return r
    def drop_piece(self, board, row, col, piece):
	    board[row][col] = piece