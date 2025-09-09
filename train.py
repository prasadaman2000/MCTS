import game
import player
import mcts_players
import numpy as np
import argparse
import time

NUM_EVALS = 100

parser = argparse.ArgumentParser(description="MCTS trainer for connect 4")
parser.add_argument("--if1", help="Input file for p1")
parser.add_argument("--if2", help="Input file for p2")
parser.add_argument("--num_checkpoints", default=100, type=int, help="number of checkpoints to collect")
parser.add_argument("--num_steps_per_checkpoint", type=int, default=1000, help="number of steps between checkpoints")

args = parser.parse_args()


p1 = mcts_players.DeepMCTSTrainer(1, r=0.1, s=250)
p2 = mcts_players.DeepMCTSTrainer(2, r=0.1, s=250)
p1.load(args.if1)
p2.load(args.if2)

r_player = player.RandomPlayer()

g = game.ConnectFour((6,7), p1, p2)

g_1 = game.ConnectFour((6,7), p1, r_player)
g_2 = game.ConnectFour((6,7), r_player, p2)

for iter in range(args.num_checkpoints):
    print(f"starting checkpoint {iter + 1}/{args.num_checkpoints}")
    p1_new_count = 0
    p2_new_count = 0
    train_game_times = []
    for i in range(args.num_steps_per_checkpoint):
        start = time.time()
        winner = g.play(eval=False)
        end = time.time()
        train_game_times.append(end - start)
        print(f"finished game {i + 1}/{args.num_steps_per_checkpoint}", end="\r")
    p1.train_model()
    p2.train_model()
    print(f"finished game {args.num_steps_per_checkpoint} games. average game time: {np.mean(train_game_times)}")
    
    num_wins_1 = 0
    num_wins_2 = 0
    eval_game_times = []
    eval_wins = []
    for i in range(NUM_EVALS):
        print(f"Playing eval game {i + 1}", end='\r')
        start = time.time()
        winner_1, state = g_1.play(eval=True)
        end = time.time()
        eval_game_times.append(end - start)

        start = time.time()
        winner_2, state = g_2.play(eval=True)
        end = time.time()
        eval_game_times.append(end - start)

        if winner_1 == 1:
            num_wins_1 += 1
        if winner_2 == 2:
            num_wins_2 += 1
    print(f"Player 1 won {num_wins_1}/{NUM_EVALS} against random")
    print(f"Player 2 won {num_wins_2}/{NUM_EVALS} against random")
    print(f"average eval game time: {np.mean(eval_game_times)}")

    with open("wins.csv", "a") as f:
        f.write(f"{num_wins_1},{num_wins_2}\n")


    print(f"dumping to file: player*_ckpt{iter % 10}")
    p1.dump(f"player1_ckpt{iter % 10}.pth")
    p2.dump(f"player2_ckpt{iter % 10}.pth")
