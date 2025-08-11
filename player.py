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
        return random.sample(state.get_valid_actions(), 1)[0]

class StateActionProbabilities:
    def __init__(self, state):
        self.state = state
        self.action_seen = defaultdict(int)
        self.action_won = defaultdict(int)

    def sample_action(self) -> int:
        """Return highest win probability action.
        If no actions have won, pick a random action
        that has been taken before."""
        max_action = random.sample(self.action_seen.keys(), 1)[0]
        max_action_prob = 0

        for action, won in self.action_won.items():
            prob = self.action_seen[action] / won
            if prob > max_action_prob:
                max_action = action
        
        return max_action

    def __str__(self):
        return f"state: {self.state}, action_seen: {self.action_seen}, action_won: {self.action_won}"

class MCTSTrainer(Player):
    def __init__(self, id):
        self.id = id
        self.states = {}
        self.cur_states_and_actions = []

    def get_move(self, state: "game.GameState", eval = False) -> int:
        move = random.sample(state.get_valid_actions(), 1)[0]
        if not eval:
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
                print(f"loading {fname}...")
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


class MCTSPlayer(MCTSTrainer):
    def get_move(self, state: "game.GameState") -> int:
        board_str = str(state.board)
        if board_str in self.states.keys():
            return self.states[board_str].sample_action()
        else:
            print("seeing this state for the first time...")
            return random.sample(state.get_valid_actions(), 1)[0]


class SmarterMCTSTrainer(MCTSTrainer):
    def __init__(self, id, r=0.2):
        self.id = id
        self.states = {}
        self.cur_states_and_actions = []
        self.r = r

    def get_move(self, state: "game.GameState", eval = False) -> int:
        board_str = str(state.board)
        if board_str in self.states.keys():
            move = self.states[board_str].sample_action()
            rand = random.random()
            if rand < self.r:
                move = random.sample(state.get_valid_actions(), 1)[0]
        else:
            move = random.sample(state.get_valid_actions(), 1)[0]
        if not eval:
            self.cur_states_and_actions.append((str(state.board), move))
        return move
