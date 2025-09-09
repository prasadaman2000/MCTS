import game
import player
import mcts_players
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Connect 4 game")
parser.add_argument("--if1", help="Input file/mode for p1")
parser.add_argument("--if2", help="Input file/mode for p2")

args = parser.parse_args()

if args.if1 == "keyboard":
    p1 = player.Player()
else:
    p1 = mcts_players.DeepMCTSTrainer(1, r=0.1, s=1000)
    p1.load(args.if1)

if args.if2 == "keyboard":
    p2 = player.Player()
else:
    p2 = mcts_players.DeepMCTSTrainer(1, r=0.1, s=1000)
    p2.load(args.if2)


g = game.ConnectFour((6,7), p1, p2)

winner, state = g.play(eval=True)
print(f"Winner is {winner}")
state.print_state()