from GameEnvironment.connectFour import ConnectFour
from MiniMaxAlgorithm.minimaxAgent import MinimaxAgent
import math

def get_player_selection():
    while True:
        try:
            selection = input('Select column [0-6]: ')
            selection = int(selection)
            return selection
        except Exception as e:
            print(e)

'''
This code file gives an example of how to use the ConnectFour class to allow 2 agents to play against each other
'''
if __name__ == '__main__':
    agent_number = 1    # Set for agent to be player 1 or player 2
   
    # Play an infinite number of games: End with ctrl + c
    while True:
        connect_four = ConnectFour(agent_number) # Intitalise game with AGENT as player 1
        # connect_four = ConnectFour(2) # Intitialise game with AGENT as player 2
        minimaxAgent = MinimaxAgent(agent_number)
        game_result = 0
        while not connect_four.is_done:
            available_actions = connect_four.get_available_actions() # A value of -1 indicates a column that is not valid
            turn_number = connect_four.get_turn_number()
            game_state = connect_four.get_state()   # 2D numpy array: 1 = player 1 tokens, 2 = player 2 tokens
            # PLAYER 1 TURN (AI)
            if turn_number % 2 == 1:
                print("available_actions:",available_actions)
                selection,minimax_score = minimaxAgent.minimax(board=game_state,depth=5, alpha=-math.inf, beta=math.inf, maximizingPlayer=True)
                print("selection: ", selection)
                # if selection==None:break
                if minimaxAgent.is_valid_location(game_state, selection).any():
                    # row = minimaxAgent.get_next_open_row(game_state, selection)

                    print("ai plays:", selection)
                    game_result = connect_four.play_turn(selection)
                
            # PLAYER 2 TURN
            elif turn_number % 2 == 0:
                connect_four.print_board()
                selection = get_player_selection()
                game_result = connect_four.play_turn(selection)
        print(f'[Game Finished] Result for Agent: {str(game_result)}')
        print()

    

