import game
import player
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="MCTS trainer for connect 4")
parser.add_argument("--if1", help="Input file for p1")
parser.add_argument("--if2", help="Input file for p2")
parser.add_argument("--num_checkpoints", default=100, type=int, help="number of checkpoints to collect")
parser.add_argument("--num_steps_per_checkpoint", type=int, default=1000, help="number of steps between checkpoints")

args = parser.parse_args()


p1 = player.MCTSTrainer(1)
p2 = player.MCTSTrainer(2)
p1.load(args.if1)
p2.load(args.if2)
g = game.ConnectFour((6,7), p1, p2)

for iter in range(args.num_checkpoints):
    for i in range(args.num_steps_per_checkpoint):
        winner = g.play()
        p1.back_propogate(winner[0])
        p2.back_propogate(winner[0])
        print(f"finished game {i}/1000", end='\r')
    print(f"dumping to file: player*_ckpt{iter % 10}")
    p1.dump(f"player1_ckpt{iter % 10}.json")
    p2.dump(f"player2_ckpt{iter % 10}.json")