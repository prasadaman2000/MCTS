import game
import random
from collections import defaultdict
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

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

class StateActionProbabilities:
    def __init__(self, state):
        self.state = state
        self.action_seen = defaultdict(int)
        self.action_won = defaultdict(int)

    def num_samples(self) -> int:
        return np.sum([v for _,v in self.action_seen.items()])

    def sample_action(self) -> int:
        """Return highest win probability action.
        If no actions have won, pick a random action
        that has been taken before."""
        max_action = random.sample(self.action_seen.keys(), 1)[0]
        max_action_prob = 0

        for action, won in self.action_won.items():
            prob = won / self.action_seen[action]
            if prob > max_action_prob:
                max_action = action
                max_action_prob = prob
        
        return max_action

    def get_probabilities(self):
        return {
            action: won / self.action_seen[action] for action, won in self.action_won.items()
        }

    def probabilities_as_np(self, arr_len) -> np.ndarray:
        probabilities = self.get_probabilities()
        prob_arr = np.zeros((arr_len,))
        for move, probability in probabilities.items():
            prob_arr[move] = probability
        return prob_arr

    def __str__(self):
        return f"state: {self.state}, action_seen: {self.action_seen}, action_probs: {self.get_probabilities()}"

class MCTSTrainer(Player):
    def __init__(self, id):
        self.id = id
        self.states = {}
        self.cur_states_and_actions = []

    def get_move(self, state: "game.GameState", **kwargs) -> int:
        move = random.sample(state.get_valid_actions(), 1)[0]
        if not eval:
            self.cur_states_and_actions.append((str(state.board), move))
        return move

    def back_propogate(self, winner) -> int:
        won = winner == self.id
        new_count = 0
        for state, action in self.cur_states_and_actions:
            if state not in self.states:
                new_count += 1
                self.states[state] = StateActionProbabilities(state)
            self.states[state].action_seen[action] += 1
            if won:
                self.states[state].action_won[action] += 1
        self.cur_states_and_actions = []
        return new_count

    def __str__(self):
        to_ret = f"Player {self.id} state:\n"
        for state, val in self.states.items():
            to_ret += str(val) + "\n"
        return to_ret

    def dump(self, fname):
        with open(fname, "w") as f:
            json.dump(self.__dict__, f, default=serialize_object)

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
        except Exception as e:
            print(f"could not load file {fname} ({e}). starting fresh")


class MCTSPlayer(MCTSTrainer):
    def get_move(self, state: "game.GameState", **kwargs) -> int:
        board_str = str(state.board)
        if board_str in self.states.keys():
            return self.states[board_str].sample_action()
        else:
            print("seeing this state for the first time...")
            return random.sample(state.get_valid_actions(), 1)[0]

class SmarterMCTSTrainer(MCTSTrainer):
    def __init__(self, id, r=0.2, s=50):
        self.id = id
        self.states = {}
        self.cur_states_and_actions = []
        self.r = r
        self.num_simulations_per_move = s
        self.move_times = []

    def get_other_player_id(self):
        return 1 if self.id == 2 else 2

    def get_move(self, init_state: "game.GameState", **kwargs) -> int:
        if kwargs["eval"]:
            board_str = str(init_state.board)
            if board_str in self.states.keys():
                return self.states[board_str].sample_action()

        start = time.time()
        for _ in range(self.num_simulations_per_move):
            cur_state = init_state
            self.cur_states_and_actions = []
            my_move = True
            while not cur_state.is_terminal()[0]:
                board_str = str(cur_state.board)
                if board_str in self.states.keys():
                    move = self.states[board_str].sample_action()
                    rand = random.random()
                    if rand < self.r:
                        move = random.sample(cur_state.get_valid_actions(), 1)[0]
                else:
                    move = random.sample(cur_state.get_valid_actions(), 1)[0]
                if my_move:
                    self.cur_states_and_actions.append((board_str, move))
                    cur_state, _ = cur_state.next_state(move, self.id)
                else:
                    cur_state, _ = cur_state.next_state(move, self.get_other_player_id())
                my_move = not my_move
            _, winner = cur_state.is_terminal()
            self.back_propogate(winner)
        
        board_str = str(init_state.board)
        move = self.states[board_str].sample_action()
        # print("sampling from")
        # print(self.states[board_str])
        rand = random.random()
        end = time.time()
        self.move_times.append(end - start)
        if rand < self.r:
            # print("random move")
            return random.sample(init_state.get_valid_actions(), 1)[0]
        return move


class DeepMCTSTrainer(MCTSTrainer):
    class SimpleConvNN(nn.Module):
        def __init__(self, action_set_size):
            super(DeepMCTSTrainer.SimpleConvNN, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(2,2))
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(36, 36)
            self.fc2 = nn.Linear(36, action_set_size)

        def forward(self, image):
            x = self.conv1(image)
            x = F.max_pool2d(torch.relu(x), (2,2)) 
            F.dropout2d(x, training=self.training)
            conv_result = self.flatten(x)
            x = self.fc1(conv_result)
            x = F.dropout(x, training=self.training)
            x = torch.relu(x)
            x = self.fc2(conv_result)
            return x

    def __init__(self, id, r=0.2, s=50, board_size=(7,6)):
        self.id = id
        self.board_size = board_size
        self.states = {}
        self.cur_states_and_actions = []
        self.states_to_train = set()
        self.r = r
        self.num_simulations_per_move = s
        self.move_times = []
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model = self.SimpleConvNN(self.board_size[0]).to(self.device) #TODO modify this
        self.optimizer = optim.Adagrad(self.model.parameters(), lr=0.01)


    def get_other_player_id(self):
        return 1 if self.id == 2 else 2

    def get_move(self, init_state: "game.GameState", **kwargs) -> int:
        if kwargs["eval"]:
            board_str = str(init_state.board)
            state_arr = np.fromstring(board_str.replace("\n", "").replace("[", "").replace("]", ""), sep=' ')
            X_torch = torch.tensor(state_arr, dtype=torch.float)
            X_torch = X_torch.reshape((1,1,self.board_size[0],self.board_size[1])).to(self.device)
            self.model.train(False)
            win_probabilities = self.model(X_torch)[0]
            print(win_probabilities)
            valid_actions = init_state.get_valid_actions()
            max_prob_action = None
            max_prob = -1000
            for action in valid_actions:
                if win_probabilities[action] > max_prob:
                    max_prob = win_probabilities[action]
                    max_prob_action = action
            return max_prob_action

        start = time.time()
        for _ in range(self.num_simulations_per_move):
            cur_state = init_state
            self.cur_states_and_actions = []
            my_move = True
            while not cur_state.is_terminal()[0]:
                board_str = str(cur_state.board)
                if board_str in self.states.keys():
                    move = self.states[board_str].sample_action()
                    rand = random.random()
                    if rand < self.r:
                        move = random.sample(cur_state.get_valid_actions(), 1)[0]
                else:
                    move = random.sample(cur_state.get_valid_actions(), 1)[0]
                if my_move:
                    self.states_to_train.add(board_str)
                    self.cur_states_and_actions.append((board_str, move))
                    cur_state, _ = cur_state.next_state(move, self.id)
                else:
                    cur_state, _ = cur_state.next_state(move, self.get_other_player_id())
                my_move = not my_move
            _, winner = cur_state.is_terminal()
            self.back_propogate(winner)
        
        board_str = str(init_state.board)
        move = self.states[board_str].sample_action()
        rand = random.random()
        end = time.time()
        self.move_times.append(end - start)
        if rand < self.r:
            return random.sample(init_state.get_valid_actions(), 1)[0]
        return move

    def train_model(self):
        X_s = []
        y_s = []
        for board_str in self.states_to_train:
            if self.states[board_str].num_samples() < 10:
                continue
            y_s.append(self.states[board_str].probabilities_as_np(self.board_size[0]))
            state_arr = np.fromstring(board_str.replace("\n", "").replace("[", "").replace("]", ""), sep=' ')
            X_s.append(state_arr)
        
        print(f"Training on {len(X_s)} datapoints")
        
        y_s = np.array(y_s)
        X_s = np.array(X_s)

        X_torch = torch.tensor(X_s, dtype=torch.float)
        X_torch = X_torch.reshape((-1,1,self.board_size[0],self.board_size[1])).to(self.device)
        y_torch = torch.tensor(y_s, dtype=torch.float).reshape((-1,self.board_size[0])).to(self.device)

        self.model.train(True)

        loss = nn.MSELoss()
        batch_size = 128
        num_epochs = 100
        all_losses = []
        for epoch in range(num_epochs):
            idx_start = 0
            epoch_loss = []
            step_count = 0
            while idx_start < len(X_torch):
                idx_end = min(len(X_torch), idx_start + batch_size)
                X_batch = X_torch[idx_start:idx_end]
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                l = loss(outputs, y_torch[idx_start:idx_end])
                if not torch.any(torch.isnan(l)):
                    l.backward()
                    self.optimizer.step()
                    epoch_loss.append(l.item())
                step_count += 1
                idx_start += batch_size
            print(f"finished epoch {epoch + 1} with average loss {np.mean(epoch_loss)}", end = "\r")
            all_losses.append(np.mean(epoch_loss))
        print(f"\nfinished training model for player{self.id} with final loss {all_losses[-1]}")
        self.states_to_train.clear()

    def dump(self, fname):
        torch.save(self.model.state_dict(), fname)

    def load(self, fname):
        try:
           if fname:
            model_state_dict = torch.load(fname)
            self.model.load_state_dict(model_state_dict)
            print("Loaded model from path")
        except Exception as e:
            print(f"could not load file {fname} ({e}). starting fresh")
