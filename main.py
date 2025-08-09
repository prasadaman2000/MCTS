import game
import player
import numpy as np

# board = np.zeros((6, 7))

# board = np.array([[0,1,0,0,],[0,0,0,0],[0,0,0,1]])

# state = game.GameState(board, 6)
# state.print_state()
# terminal = state.is_terminal()
# empty = ''
# print(f"state {'is' if terminal[0] else 'is not'} terminal{f' winner is {terminal[1] if terminal[0] else empty}'}")

# print(state.get_valid_actions())

# valid = True
# terminal = False
# col = 0
# empty = ''
# while valid and not terminal:
#     new_state, valid = state.next_state(col % 6, (col % 2) + 1)
#     if new_state:
#         terminal = new_state.is_terminal()
#         print(f"state {'is' if terminal[0] else 'is not'} terminal{f' winner is {terminal[1] if terminal[0] else empty}'}")
#         new_state.print_state()
#         state = new_state
#         col += 1
#         terminal = terminal[0]


p1 = player.MCTSTrainer(1)
p2 = player.MCTSTrainer(2)
p1.load("player1.json")
p2.load("player2.json")
g = game.ConnectFour((6,7), p1, p2)

for iter in range(100):
    for _ in range(1000):
        winner = g.play()
        p1.back_propogate(winner[0])
        p2.back_propogate(winner[0])
        print(f"winner: {winner[0]}")

    p1.dump(f"player1_ckpt{iter % 10}.json")
    p2.dump(f"player2_ckpt{iter % 10}.json")