import random
import numpy as np
import player

class GameState:
    def __init__(self, board: np.ndarray, n: int):
        shape = board.shape
        self.board = board
        self.rows = shape[1]
        self.cols = shape[0]
        self.n = n

    def next_state(self, played_col, player) -> tuple["GameState", bool]:
        "Returns (new state, is_valid)"
        if player != 1 and player != 2:
            print("error, invalid player")
            return None, False

        if self.board[played_col][-1] != 0:
            print(f"error, column {played_col} full")
            return None, False

        new_board = self.board.copy()
        for row, row_val in enumerate(new_board[played_col]):
            if row_val == 0:
                new_board[played_col][row] = player
                break
        
        return GameState(new_board, self.n), True
    
    def _check_n_in_direction(self, col, row, dir) -> tuple[bool, int]:
        dx, dy = dir
        player = self.board[col][row]
        num_in_row = 0
        while col < self.cols and row < self.rows:
            if self.board[col][row] != player:
                return False, 0
            num_in_row += 1
            if num_in_row == self.n:
                return True, player
            col += dx
            row += dy
        return False, 0
    
    def is_terminal(self) -> tuple[bool, int]:
        directions = [(1,0), (0,1), (1, -1), (1, 1)]
        has_zero = False
        for col in range(self.cols):
            for row in range(self.cols):
                if self.board[col][row] == 0:
                    has_zero = True
                    continue
                for dir in directions:
                    result = self._check_n_in_direction(col, row, dir)
                    if result[0]:
                        return result
        if not has_zero:
            return True, -1
        return False, 0

    def get_valid_actions(self) -> list[int]:
        return [col for col in range(self.cols) if self.board[col][-1] == 0]
    
    def print_state(self):
        print("#####")
        print(self.board.T)
    
class ConnectFour:
    def __init__(self, dims: tuple[int, int], player1: player.Player, player2: player.Player, n: int = 4):
        self.player1 = player1
        self.player2 = player2
        self.dims = dims
        self.n = n

    def play(self) -> tuple[int, GameState]:
        board = np.zeros(self.dims, dtype=np.int16)
        cur_state = GameState(board, self.n)

        while True:
            valid = False
            while not valid:
                p1_move = self.player1.get_move(cur_state)
                cur_state, valid = cur_state.next_state(p1_move, 1)
            is_terminal, winner = cur_state.is_terminal()
            if is_terminal:
                return winner, cur_state
            valid = False
            while not valid:
                p2_move = self.player2.get_move(cur_state)
                cur_state, valid = cur_state.next_state(p2_move, 2)
            is_terminal, winner = cur_state.is_terminal()
            if is_terminal:
                return winner, cur_state
