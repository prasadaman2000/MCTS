import game
import random
from collections import defaultdict
import json

def serialize_object(obj):
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

class Player:
    def get_move(self, state: "game.GameState"):
        state.print_state()
        return int(input(f"choose a move from {state.get_valid_actions()}: "))
    
class RandomPlayer(Player):
    def get_move(self, state: "game.GameState"):
        state.print_state()
        return random.sample(state.get_valid_actions(), 1)[0]

class StateActionProbabilities:
    def __init__(self, state):
        self.state = state
        self.action_seen = defaultdict(int)
        self.action_won = defaultdict(int)

    def __str__(self):
        return f"state: {self.state}, action_seen: {self.action_seen}, action_won: {self.action_won}"

class MCTSTrainer(Player):
    def __init__(self, id):
        self.id = id
        self.states = {}
        self.cur_states_and_actions = []
    
    def get_move(self, state: "game.GameState"):
        move = random.sample(state.get_valid_actions(), 1)[0]
        self.cur_states_and_actions.append((str(state.board), move))
        return move

    def back_propogate(self, winner):
        won = winner == self.id
        for state, action in self.cur_states_and_actions:
            if state not in self.states:
                self.states[state] = StateActionProbabilities(state)
            self.states[state].action_seen[action] += 1
            if won:
                self.states[state].action_won[action] += 1
        self.cur_states_and_actions = []

    def __str__(self):
        to_ret = f"Player {self.id} state:\n"
        for state, val in self.states.items():
            to_ret += str(val) + "\n"
        return to_ret

    def dump(self, fname):
        with open(fname, "w") as f:
            json.dump(self.__dict__, f, default=serialize_object, indent=2)

    def load(self, fname):
        try:
            with open(fname, "r") as f:
                loaded = json.load(f)
            for k, v in loaded['states'].items():
                sapb = StateActionProbabilities(k)
                for action, count in v['action_seen'].items():
                    sapb.action_seen[int(action)] = count
                for action, count in v['action_won'].items():
                    sapb.action_won[int(action)] = count
                self.states[k] = sapb
            print(f"Loaded state from {fname}")
        except:
            print(f"could not load file {fname}. starting fresh")