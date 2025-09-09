import game
import random

def serialize_object(obj):
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

class Player:
    def get_move(self, state: "game.GameState", **kwargs):
        state.print_state()
        return int(input(f"choose a move from {state.get_valid_actions()}: "))

class RandomPlayer(Player):
    def get_move(self, state: "game.GameState", **kwargs):
        return random.sample(state.get_valid_actions(), 1)[0]

