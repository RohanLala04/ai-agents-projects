import copy
import numpy as np

INPUT_FILE = 'input.txt'
OUTPUT_FILE = 'output.txt'
STEP_FILE = 'step_num.txt'
SIZE = 5
EMPTY = 0
BLACK = 1
WHITE = 2
KOMI = 2.5
DX = [1, 0, -1, 0]
DY = [0, 1, 0, -1]


class AlphaBetaPlayer:
    def __init__(self, color, prev_board, curr_board):
        self.color = color
        self.opponent_color = self.get_opponent(self.color)
        self.prev_board = prev_board
        self.curr_board = curr_board

    def search(self, depth, branching, step_num):
        best_move, best_score = self.maximize(self.curr_board, self.color, depth, 0, branching,
                                               -np.inf, np.inf, None, step_num, False)
        write_output(best_move)

    def maximize(self, board, color, depth, curr_depth, branching, alpha, beta, prev_move,
                 step_num, second_pass):
        if curr_depth == depth or step_num + curr_depth == 24:
            return self.evaluate(board, color)
        if second_pass:
            return self.evaluate(board, color)
        
        second_pass = False
        best_score = -np.inf
        best_move = None
        moves = self.get_valid_moves(board, color)
        moves.append((-1, -1))
        
        if prev_move == (-1, -1):
            second_pass = True
        
        for move in moves[:branching]:
            opp_color = self.get_opponent(color)
            if move == (-1, -1):
                next_board = copy.deepcopy(board)
            else:
                next_board = self.apply_move(board, color, move)
            
            score = self.minimize(next_board, opp_color, depth, curr_depth + 1,
                                 branching, alpha, beta, move, step_num, second_pass)
            
            if best_score < score:
                best_score = score
                best_move = move
            
            if best_score >= beta:
                if curr_depth == 0:
                    return best_move, best_score
                else:
                    return best_score
            
            alpha = max(alpha, best_score)
        
        if curr_depth == 0:
            return best_move, best_score
        else:
            return best_score

    def minimize(self, board, color, depth, curr_depth, branching, alpha, beta, prev_move,
                 step_num, second_pass):
        if curr_depth == depth:
            return self.evaluate(board, color)
        if step_num + curr_depth == 24 or second_pass:
            return self.evaluate(board, self.color)
        
        second_pass = False
        best_score = np.inf
        moves = self.get_valid_moves(board, color)
        moves.append((-1, -1))
        
        if prev_move == (-1, -1):
            second_pass = True
        
        for move in moves[:branching]:
            opp_color = self.get_opponent(color)
            if move == (-1, -1):
                next_board = copy.deepcopy(board)
            else:
                next_board = self.apply_move(board, color, move)
            
            score = self.maximize(next_board, opp_color, depth, curr_depth + 1,
                                 branching, alpha, beta, move, step_num, second_pass)
            
            if score < best_score:
                best_score = score
            
            if best_score <= alpha:
                return best_score
            
            beta = min(beta, best_score)
        
        return best_score

    def evaluate(self, board, color):
        opp_color = self.get_opponent(color)
        my_count = 0
        my_liberties = set()
        opp_count = 0
        opp_liberties = set()
        
        for i in range(SIZE):
            for j in range(SIZE):
                if board[i][j] == color:
                    my_count += 1
                elif board[i][j] == opp_color:
                    opp_count += 1
                else:
                    for k in range(len(DX)):
                        ni = i + DX[k]
                        nj = j + DY[k]
                        if 0 <= ni < SIZE and 0 <= nj < SIZE:
                            if board[ni][nj] == color:
                                my_liberties.add((i, j))
                            elif board[ni][nj] == opp_color:
                                opp_liberties.add((i, j))

        my_edge = 0
        opp_edge = 0
        for j in range(SIZE):
            if board[0][j] == color or board[SIZE - 1][j] == color:
                my_edge += 1
            if board[0][j] == opp_color or board[SIZE - 1][j] == opp_color:
                opp_edge += 1

        for j in range(1, SIZE - 1):
            if board[j][0] == color or board[j][SIZE - 1] == color:
                my_edge += 1
            if board[j][0] == opp_color or board[j][SIZE - 1] == opp_color:
                opp_edge += 1

        center_empty = 0
        for i in range(1, SIZE - 1):
            for j in range(1, SIZE - 1):
                if board[i][j] == EMPTY:
                    center_empty += 1

        liberty_diff = min(max((len(my_liberties) - len(opp_liberties)), -8), 8)
        euler_term = -4 * self.compute_euler(board, color)
        stone_diff = 5 * (my_count - opp_count)
        edge_penalty = -9 * my_edge * (center_empty / 9)
        
        score = liberty_diff + euler_term + stone_diff + edge_penalty
        
        if self.color == WHITE:
            score += KOMI

        return score

    def apply_move(self, board, color, move):
        new_board = copy.deepcopy(board)
        new_board[move[0]][move[1]] = color
        
        for k in range(len(DX)):
            ni = move[0] + DX[k]
            nj = move[1] + DY[k]
            if 0 <= ni < SIZE and 0 <= nj < SIZE:
                opp_color = self.get_opponent(color)
                if new_board[ni][nj] == opp_color:
                    stack = [(ni, nj)]
                    visited = set()
                    should_remove = True
                    
                    while stack:
                        curr = stack.pop()
                        visited.add(curr)
                        
                        for d in range(len(DX)):
                            next_i = curr[0] + DX[d]
                            next_j = curr[1] + DY[d]
                            if 0 <= next_i < SIZE and 0 <= next_j < SIZE:
                                if (next_i, next_j) in visited:
                                    continue
                                elif new_board[next_i][next_j] == EMPTY:
                                    should_remove = False
                                    break
                                elif new_board[next_i][next_j] == opp_color and (next_i, next_j) not in visited:
                                    stack.append((next_i, next_j))

                    if should_remove:
                        for stone in visited:
                            new_board[stone[0]][stone[1]] = EMPTY
        
        return new_board

    def compute_euler(self, board, color):
        opp_color = self.get_opponent(color)
        padded = np.zeros((SIZE + 2, SIZE + 2), dtype=int)
        
        for i in range(SIZE):
            for j in range(SIZE):
                padded[i + 1][j + 1] = board[i][j]

        q1_my = 0
        q2_my = 0
        q3_my = 0
        q1_opp = 0
        q2_opp = 0
        q3_opp = 0

        for i in range(SIZE):
            for j in range(SIZE):
                sub = padded[i: i + 2, j: j + 2]
                q1_my += self.count_q1(sub, color)
                q2_my += self.count_q2(sub, color)
                q3_my += self.count_q3(sub, color)
                q1_opp += self.count_q1(sub, opp_color)
                q2_opp += self.count_q2(sub, opp_color)
                q3_opp += self.count_q3(sub, opp_color)

        return (q1_my - q3_my + 2 * q2_my - (q1_opp - q3_opp + 2 * q2_opp)) / 4

    def count_q1(self, sub, color):
        if ((sub[0][0] == color and sub[0][1] != color and sub[1][0] != color and sub[1][1] != color) or
            (sub[0][0] != color and sub[0][1] == color and sub[1][0] != color and sub[1][1] != color) or
            (sub[0][0] != color and sub[0][1] != color and sub[1][0] == color and sub[1][1] != color) or
            (sub[0][0] != color and sub[0][1] != color and sub[1][0] != color and sub[1][1] == color)):
            return 1
        else:
            return 0

    def count_q2(self, sub, color):
        if ((sub[0][0] == color and sub[0][1] != color and sub[1][0] != color and sub[1][1] == color) or
            (sub[0][0] != color and sub[0][1] == color and sub[1][0] == color and sub[1][1] != color)):
            return 1
        else:
            return 0

    def count_q3(self, sub, color):
        if ((sub[0][0] == color and sub[0][1] == color and sub[1][0] == color and sub[1][1] != color) or
            (sub[0][0] != color and sub[0][1] == color and sub[1][0] == color and sub[1][1] == color) or
            (sub[0][0] == color and sub[0][1] != color and sub[1][0] == color and sub[1][1] == color) or
            (sub[0][0] == color and sub[0][1] == color and sub[1][0] != color and sub[1][1] == color)):
            return 1
        else:
            return 0

    def get_valid_moves(self, board, color):
        moves = {'capturing': [], 'regular': [], 'edge': []}
        
        for i in range(SIZE):
            for j in range(SIZE):
                if board[i][j] == EMPTY:
                    if self.has_liberty(board, i, j, color):
                        if not self.is_ko(i, j):
                            if i == 0 or j == 0 or i == SIZE - 1 or j == SIZE - 1:
                                moves['edge'].append((i, j))
                            else:
                                moves['regular'].append((i, j))
                    else:
                        for k in range(len(DX)):
                            ni = i + DX[k]
                            nj = j + DY[k]
                            if 0 <= ni < SIZE and 0 <= nj < SIZE:
                                opp_color = self.get_opponent(color)
                                if board[ni][nj] == opp_color:
                                    test_board = copy.deepcopy(board)
                                    test_board[i][j] = color
                                    if not self.has_liberty(test_board, ni, nj, opp_color):
                                        if not self.is_ko(i, j):
                                            moves['capturing'].append((i, j))
                                        break

        result = []
        result.extend(moves['capturing'])
        result.extend(moves['regular'])
        result.extend(moves['edge'])
        return result

    def has_liberty(self, board, i, j, color):
        stack = [(i, j)]
        visited = set()
        
        while stack:
            curr = stack.pop()
            visited.add(curr)
            
            for k in range(len(DX)):
                ni = curr[0] + DX[k]
                nj = curr[1] + DY[k]
                if 0 <= ni < SIZE and 0 <= nj < SIZE:
                    if (ni, nj) in visited:
                        continue
                    elif board[ni][nj] == EMPTY:
                        return True
                    elif board[ni][nj] == color and (ni, nj) not in visited:
                        stack.append((ni, nj))
        
        return False

    def get_opponent(self, color):
        return WHITE if color == BLACK else BLACK

    def is_ko(self, i, j):
        if self.prev_board[i][j] != self.color:
            return False
        
        test_board = copy.deepcopy(self.curr_board)
        test_board[i][j] = self.color
        opp_move = self.find_opponent_move()
        
        if opp_move:
            opp_i, opp_j = opp_move
            for k in range(len(DX)):
                ni = i + DX[k]
                nj = j + DY[k]
                if ni == opp_i and nj == opp_j:
                    if not self.has_liberty(test_board, ni, nj, self.opponent_color):
                        self.remove_group(test_board, ni, nj, self.opponent_color)
        
        return np.array_equal(test_board, self.prev_board)

    def find_opponent_move(self):
        if np.array_equal(self.curr_board, self.prev_board):
            return None
        
        for i in range(SIZE):
            for j in range(SIZE):
                if self.curr_board[i][j] != self.prev_board[i][j] and self.curr_board[i][j] != EMPTY:
                    return i, j
        return None

    def remove_group(self, board, i, j, color):
        stack = [(i, j)]
        visited = set()
        
        while stack:
            curr = stack.pop()
            visited.add(curr)
            board[curr[0]][curr[1]] = EMPTY
            
            for k in range(len(DX)):
                ni = curr[0] + DX[k]
                nj = curr[1] + DY[k]
                if 0 <= ni < SIZE and 0 <= nj < SIZE:
                    if (ni, nj) in visited:
                        continue
                    elif board[ni][nj] == color:
                        stack.append((ni, nj))
        
        return board


def read_input(filename=INPUT_FILE):
    with open(filename) as f:
        lines = [line.strip() for line in f.readlines()]
        color = int(lines[0])
        prev_board = np.zeros((SIZE, SIZE), dtype=int)
        curr_board = np.zeros((SIZE, SIZE), dtype=int)
        
        for i in range(1, 6):
            for j in range(len(lines[i])):
                prev_board[i - 1][j] = int(lines[i][j])
        
        for i in range(6, 11):
            for j in range(len(lines[i])):
                curr_board[i - 6][j] = int(lines[i][j])
        
        return color, prev_board, curr_board


def write_output(move):
    with open(OUTPUT_FILE, 'w') as f:
        if move is None or move == (-1, -1):
            f.write('PASS')
        else:
            f.write(f'{move[0]},{move[1]}')


def calculate_step_number(prev_board, curr_board):
    prev_empty = True
    curr_empty = True
    
    for i in range(SIZE - 1):
        for j in range(SIZE - 1):
            if prev_board[i][j] != EMPTY:
                prev_empty = False
                curr_empty = False
                break
            elif curr_board[i][j] != EMPTY:
                curr_empty = False

    if prev_empty and curr_empty:
        step_num = 0
    elif prev_empty and not curr_empty:
        step_num = 1
    else:
        with open(STEP_FILE) as f:
            step_num = int(f.readline())
            step_num += 2

    with open(STEP_FILE, 'w') as f:
        f.write(f'{step_num}')

    return step_num


if __name__ == '__main__':
    color, prev_board, curr_board = read_input()
    step_num = calculate_step_number(prev_board, curr_board)
    player = AlphaBetaPlayer(color, prev_board, curr_board)
    player.search(4, 20, step_num)