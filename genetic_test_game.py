from GameEnvironment.connectFour import ConnectFour
from Agents.randomAgent import RandomAgent
from GeneticAlgorithm.geneticAgent import GeneticAgent

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
    # Play an infinite number of games: End with ctrl + c
    feature_weights = [0.467, 0.397, 0.044, 0.239, 0.592, 0.554, 0.783, 0.496, 0.561, 0.418, 0.033]
    agent_number = 1    # Set for agent to be player 1 or player 2
    agent = GeneticAgent(agent_number, feature_weights)
    count = 0
    max_games = 1000
    accumulated_outcomes = []
    to_continue = True
    while count < max_games and to_continue:
        connect_four = ConnectFour(agent_number)
        random_agent = RandomAgent()
        game_result = 0
        while not connect_four.is_done:
            available_actions = connect_four.get_available_actions() # A value of -1 indicates a column that is not valid
            turn_number = connect_four.get_turn_number()
            game_state = connect_four.get_state()   # 2D numpy array: 1 = player 1 tokens, 2 = player 2 tokens
            # PLAYER 1 TURN
            if turn_number % 2 == 1:
                # --------- MOVE SELECTION --------- #
                # connect_four.print_board()
                # selection = get_player_selection()
                # selection = random_agent.select_random_column(available_actions)      # Random agent
                selection = agent.selectAction(game_state, available_actions)
                # ----------- PLAY MOVE ----------- #
                game_result = connect_four.play_turn(selection)
            # PLAYER 2 TURN
            elif turn_number % 2 == 0:
                # --------- MOVE SELECTION --------- #
                connect_four.print_board()
                selection = get_player_selection()
                # selection = random_agent.select_random_column(available_actions)      # Random agent
                # selection = agent.selectAction(game_state, available_actions)
                # ----------- PLAY MOVE ----------- #
                game_result = connect_four.play_turn(selection)
        print(f'[Game Finished] Result for Agent: {str(game_result)}')
        count += 1
        accumulated_outcomes.append(game_result)
        print()
    print('ALL GAMES PLAYED - FINAL OUTCOME')
    num_wins = 0
    num_loss = 0
    num_draw = 0
    for outcome in accumulated_outcomes:
        if outcome == 1:
            num_wins += 1
        elif outcome == 0:
            num_draw += 1
        elif outcome == -1:
            num_loss += 1
    print(f'WINS: {str(num_wins)}')
    print(f'LOSS: {str(num_loss)}')
    print(f'DRAW: {str(num_draw)}')
            

