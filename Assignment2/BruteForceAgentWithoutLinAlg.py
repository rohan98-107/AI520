from MineSweeper import *
from collections import deque
from copy import deepcopy
import sys
import time
from Agent import *
from LinAlg import *

class brute_force_no_lin_alg_agent(agent):
    def __init__(self, game, useMineCount, order):
        agent.__init__(self,game,order)
        self.useMineCount = useMineCount

    def confirm_full_consistency(self, board, cells):
        dim = self.game.dim
        for x,y in cells:
            clue = self.game.board[x][y]
            numSafeNbrs = 0
            numMineNbrs = 0
            numHiddenNbrs = 0
            numTotalNbrs = 0
            for i, j in dirs:
                nx  = x + i
                ny = y + j
                if 0 <= nx < dim and 0 <= ny < dim:
                    numTotalNbrs += 1
                    if (board[nx][ny] == MINE) or (board[nx][ny] == DETONATED):
                        numMineNbrs += 1
                    elif board[nx][ny] == SAFE:
                        numSafeNbrs += 1
                    elif board[nx][ny] == HIDDEN:
                        numHiddenNbrs += 1
            if numMineNbrs != clue:
                return False
        return True

    def confirm_consistency(self, board, x, y):
        clue = self.game.board[x][y]
        dim = self.game.dim
        numSafeNbrs = 0
        numMineNbrs = 0
        numHiddenNbrs = 0
        numTotalNbrs = 0
        for i, j in dirs:
            nx  = x + i
            ny = y + j
            if 0 <= nx < dim and 0 <= ny < dim:
                numTotalNbrs += 1
                if (board[nx][ny] == MINE) or (board[nx][ny] == DETONATED):
                    numMineNbrs += 1
                elif board[nx][ny] == SAFE:
                    numSafeNbrs += 1
                elif board[nx][ny] == HIDDEN:
                    numHiddenNbrs += 1
        if numMineNbrs <= clue:
            return True
        return False

    def probability_method(self):
        # print(self.playerKnowledge)
        dim = self.game.dim
        consistency_cells = []
        config_cells = list()
        for x in range(dim):
            for y in range(dim):
                if self.playerKnowledge[x][y] == SAFE:
                    numSafeNbrs, numMineNbrs, numHiddenNbrs, numTotalNbrs = self.getCellNeighborData(x, y)
                    if numHiddenNbrs > 0:
                        consistency_cells.append((x,y))
                        children = set(self.get_hidden_neighbors(x,y))
                        # print(children)
                        intersects = []
                        for i,s in enumerate(config_cells):
                            if len(children.intersection(s)) > 0:
                                intersects.append(i)
                        if len(intersects) > 0:
                            for i in intersects:
                                children.update(config_cells[i])
                            for i in intersects[::-1]:
                                del config_cells[i]
                        config_cells.append(children)
        #                 print(config_cells)
        # print(config_cells)

        if len(consistency_cells) == 0:
            return self.get_next_random(set())

        config_cells = [sorted(list(y), key = lambda x: x[0] * dim + x[1]) for y in config_cells]

        configs = []
        to_remove = []
        for i,s in enumerate(config_cells):
            if len(s) > 20:
                to_remove.append(i)
                # print("len(s)={}, ignoring".format(len(s)))
                continue
            start_search_time = time.time()
            configs.append(self.get_configs(s, consistency_cells))
            if self.logging and time.time() - start_search_time > 10:
                print("len(s)={}, took {} seconds to compute".format(len(s), round(time.time() - start_search_time,2)))
        # print([len(s) for s in config_cells])
        for i in to_remove[::-1]:
            del config_cells[i]
        # print([len(s) for s in config_cells])
        # print(len(config_cells))
        # print(len(configs))
        if len(configs) == 0:
            return self.get_next_random(set())


        min_mine_count = sum([ 0 if len(s) == 0 else min([len(config) for config in s]) for s in configs])
        for s in configs:
            min_for_set =  0 if len(s) == 0 else min([len(config) for config in s])
            to_remove = []
            for i,config in enumerate(s):
                if len(config) + min_mine_count - min_for_set > self.game.num_mines-self.numFlaggedMines-self.numDetonatedMines:
                    to_remove.append(i)
            for i in to_remove[::-1]:
                del s[i]

        # print("found configs:")
        probabilities = dict()
        for i,s in enumerate(configs):
            if len(s) > 0:
                counts = {x:0 for x in config_cells[i]}
                for config in s:
                    for coordinates in config:
                        counts[coordinates] += 1
                for k , v in counts.items():
                    probabilities[k] = v / len(s)
            else:
                for cell in config_cells[i]:
                    probabilities[cell] = 0

        best_cell = None
        best_probability = len(configs)

        for cell, probability in probabilities.items():
            if probability <= best_probability:
                best_cell = cell
                best_probability = probability

        other_cells = self.numHiddenCells() - len(config_cells)
        mines_left = self.game.num_mines-self.numFlaggedMines-self.numDetonatedMines
        max_mines_in_hidden_cells_not_in_config_Cells = mines_left - min_mine_count
        other_cell_max_probability = 1 if other_cells == 0 else max_mines_in_hidden_cells_not_in_config_Cells / other_cells

        if best_probability < other_cell_max_probability:
            assert best_cell is not None
            return best_cell
        else:
            to_exclude = []
            for l in config_cells:
                to_exclude.extend(l)
            return self.get_next_random(set(to_exclude))

    def get_configs(self, configCells, consistency_cells):
        configs = [(deepcopy(self.playerKnowledge), i, [])  for i in range(len(configCells))]
        out = []
        while configs:
            next_set = []
            for board, index, currentConfig in configs:
                current_board = deepcopy(board)
                current_config = deepcopy(currentConfig)
                dim = self.game.dim

                if index >= len(configCells) or (self.useMineCount and len(current_config) == self.game.num_mines-self.numFlaggedMines-self.numDetonatedMines):
                    if self.confirm_full_consistency(current_board, consistency_cells):
                        out.append(current_config)
                    continue
                cell = configCells[index]
                current_config.append(cell)
                x = cell[0]
                y = cell[1]
                current_board[x,y] = MINE
                valid = True
                for dx, dy in dirs:
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < dim and 0 <= ny < dim and self.playerKnowledge[nx][ny] == SAFE and not self.confirm_consistency(current_board, nx, ny):
                        valid = False
                if valid:
                    for i in range(index + 1, len(configCells) + 1):
                        next_set.append((current_board,i,current_config))

            configs = next_set
        return out

def baselineVsBruteNoLinAlgComparisonGameDriver(dim, density, trials, useMineCount = False):
    print("baseline vs brute no lin alg, dim {}, density {}, trials {}, useMineCount={}".format(dim,density,trials,useMineCount))
    num_mines = int(density*(dim**2))
    baseline_cumulative_time = 0
    baseline_cumulative_rate = 0
    brute_cumulative_time = 0
    brute_cumulative_rate = 0
    for i in range(trials):
        game = MineSweeper(dim, num_mines)
        order = [i for i in range(dim**2)]
        random.shuffle(order)
        brute_agent = brute_force_no_lin_alg_agent(game, useMineCount, order)
        baselineAgent = agent(game, order)
        brute_agent.solve()
        baselineAgent.solve()
        baseline_cumulative_time+=baselineAgent.totalSolveTime
        baseline_cumulative_rate+=baselineAgent.mineFlagRate*100
        brute_cumulative_time+=brute_agent.totalSolveTime
        brute_cumulative_rate+=brute_agent.mineFlagRate*100
        if i % 10 == 9:
            print('\n\n\n\n\nFinished {} trials:\n\tBaseline average was {} seconds detcting {}% of mines\n\tBrute finished in {} seconds detcting {}% of mines'\
                  .format((i+1), baseline_cumulative_time/(i+1), baseline_cumulative_rate/(i+1), brute_cumulative_time/(i+1), brute_cumulative_rate/(i+1)))


def linalgVsBruteNoLinAlgComparisonGameDriver(dim, density, trials, useMineCount = False):
    print("lin alg vs brute no lin alg, dim {}, density {}, trials {}, useMineCount={}".format(dim,density,trials,useMineCount))
    num_mines = int(density*(dim**2))
    la_cumulative_time = 0
    la_cumulative_rate = 0
    brute_cumulative_time = 0
    brute_cumulative_rate = 0
    for i in range(trials):
        game = MineSweeper(dim, num_mines)
        order = [i for i in range(dim**2)]
        random.shuffle(order)
        brute_agent = brute_force_no_lin_alg_agent(game, useMineCount, order)
        la_agent = lin_alg_agent(game, useMineCount, order)
        brute_agent.solve()
        la_agent.solve()
        la_cumulative_time+=la_agent.totalSolveTime
        la_cumulative_rate+=la_agent.mineFlagRate*100
        brute_cumulative_time+=brute_agent.totalSolveTime
        brute_cumulative_rate+=brute_agent.mineFlagRate*100
        if i % 10 == 9:
            print('\n\n\n\n\nFinished {} trials:\n\tLin alg average was {} seconds detcting {}% of mines\n\tBrute finished in {} seconds detcting {}% of mines'\
                  .format((i+1), la_cumulative_time/(i+1), la_cumulative_rate/(i+1), brute_cumulative_time/(i+1), brute_cumulative_rate/(i+1)))
