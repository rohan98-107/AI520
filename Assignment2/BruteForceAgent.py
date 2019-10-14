from MineSweeper import *
from collections import deque
from copy import deepcopy
import sys
import time
from Agent import *
from LinAlg import *

class brute_force_agent(lin_alg_agent):
    def __init__(self, game, useMineCount, order):
        lin_alg_agent.__init__(self,game,useMineCount,order)
        self.current_iter_configs = []

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
        # print()
        # print("{},{}, with clue:{}".format(x,y,clue))
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
        # print("num safe: {}, num mines: {}, num hidden: {}, total: {}".format(numSafeNbrs, numMineNbrs, numHiddenNbrs, numTotalNbrs))
        if numMineNbrs <= clue:
            return True
        # print("rejected for consistency")
        return False

    def probability_method(self):
        # print("probability_method")
        # print()
        # print("board")
        # print(np.array(self.game.board))
        # print()
        # print("knowledge")
        # print(self.playerKnowledge)
        # print()
        self.current_iter_configs = []
        dim = self.game.dim
        consistency_cells = []
        config_cells = set()
        for x in range(dim):
            for y in range(dim):
                if self.playerKnowledge[x][y] == SAFE:
                    numSafeNbrs, numMineNbrs, numHiddenNbrs, numTotalNbrs = self.getCellNeighborData(x, y)
                    if numHiddenNbrs > 0:
                        consistency_cells.append((x,y))
                    for nx, ny in self.get_hidden_neighbors(x,y):
                        config_cells.add((nx,ny))

        config_cells = sorted(list(config_cells), key = lambda x: x[0] * dim + x[1])


        if len(config_cells) <= 20:
            start_search_time = time.time()
            self.get_configs(config_cells, consistency_cells)
            # print("len(config_cells)={}, took {} seconds to compute".format(len(config_cells), round(time.time() - start_search_time,2)))

        if len(consistency_cells) == 0 or len(self.current_iter_configs) == 0:
            return self.get_next_random(set())

        counts = {x:0 for x in config_cells}

        min_mine_count = len(config_cells)
        # print("found configs:")
        for config in self.current_iter_configs:
            # print(config)
            for coordinates in config:
                counts[coordinates] += 1
            min_mine_count = min(min_mine_count, len(config))

        best_cell = None
        best_count = len(self.current_iter_configs)

        for cell, count in counts.items():
            if count <= best_count:
                best_count = count
                best_cell = cell

        # print()
        # print("best cell: {},{}".format(best_cell[0], best_cell[1]))
        # print()
        other_cells = self.numHiddenCells() - len(config_cells)
        mines_left = self.game.num_mines-self.numFlaggedMines-self.numDetonatedMines
        max_mines_in_hidden_cells_not_in_config_Cells = mines_left - min_mine_count
        other_cell_max_probability = 1 if other_cells == 0 else max_mines_in_hidden_cells_not_in_config_Cells / other_cells
        if best_count / len(self.current_iter_configs) < other_cell_max_probability:
            assert best_cell is not None
            return best_cell
        else:
            return self.get_next_random(set(config_cells))
            return (x,y)

    def get_configs(self, configCells, consistency_cells):
        configs = [(deepcopy(self.playerKnowledge), i, [])  for i in range(len(configCells))]
        # print("config cells:")
        # print(configCells)
        # print()
        while configs:
            # print("current iter configs:")
            # for _,i,config in configs:
            #     print(config + [configCells[i] if i < len(configCells) else None])
            next_set = []
            for board, index, currentConfig in configs:
                current_board = deepcopy(board)
                current_config = deepcopy(currentConfig)
                dim = self.game.dim

                if index >= len(configCells) or (self.useMineCount and len(current_config) == self.game.num_mines-self.numFlaggedMines-self.numDetonatedMines):
                    if self.confirm_full_consistency(current_board, consistency_cells):
                        # print("found config")
                        # print(current_config)
                        self.current_iter_configs.append(current_config)
                    continue
                cell = configCells[index]
                current_config.append(cell)
                x = cell[0]
                y = cell[1]
                current_board[x,y] = MINE
                # print()
                # for row in self.game.board:
                #     print(row)
                # print()
                # print(current_board)
                valid = True
                for dx, dy in dirs:
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < dim and 0 <= ny < dim and self.playerKnowledge[nx][ny] == SAFE and not self.confirm_consistency(current_board, nx, ny):
                        # print("config rejected due to inconsistency at cell {},{}".format(nx,ny))
                        # print(current_config)
                        valid = False
                if valid:
                    for i in range(index + 1, len(configCells) + 1):
                        next_set.append((current_board,i,current_config))

            configs = next_set

def baselineVsBruteComparisonGameDriver(dim, density, trials):
    num_mines = int(density*(dim**2))
    baseline_cumulative_time = 0
    baseline_cumulative_rate = 0
    brute_cumulative_time = 0
    brute_cumulative_rate = 0
    for i in range(trials):
        game = MineSweeper(dim, num_mines)
        order = [i for i in range(dim**2)]
        random.shuffle(order)
        brute_agent = brute_force_agent(game, True, order)
        baselineAgent = agent(game, order)
        brute_agent.solve()
        baselineAgent.solve()
        print('Trial {}:\n\tBaseline finished in {} seconds detcting {}% of mines\n\tBrute finished in {} seconds detcting {}% of mines'\
              .format(i+1, baselineAgent.totalSolveTime, baselineAgent.mineFlagRate*100, brute_agent.totalSolveTime, brute_agent.mineFlagRate*100))
        baseline_cumulative_time+=baselineAgent.totalSolveTime
        baseline_cumulative_rate+=baselineAgent.mineFlagRate*100
        brute_cumulative_time+=brute_agent.totalSolveTime
        brute_cumulative_rate+=brute_agent.mineFlagRate*100
    print('\n\n\n\n\nFinished {} trials:\n\tBaseline average was {} seconds detcting {}% of mines\n\tBrute finished in {} seconds detcting {}% of mines'\
          .format(i+1, baseline_cumulative_time/trials, baseline_cumulative_rate/trials, brute_cumulative_time/trials, brute_cumulative_rate/trials))

def linalgVsBruteComparisonGameDriver(dim, density, trials):
    num_mines = int(density*(dim**2))
    la_cumulative_time = 0
    la_cumulative_rate = 0
    brute_cumulative_time = 0
    brute_cumulative_rate = 0
    for i in range(trials):
        game = MineSweeper(dim, num_mines)
        order = [i for i in range(dim**2)]
        random.shuffle(order)
        brute_agent = brute_force_agent(game, True, order)
        la_agent = lin_alg_agent(game, True, order)
        brute_agent.solve()
        la_agent.solve()
        # print('Trial {}:\n\tLin alg finished in {} seconds detcting {}% of mines\n\tBrute finished in {} seconds detcting {}% of mines'\
        #       .format(i+1, la_agent.totalSolveTime, la_agent.mineFlagRate*100, brute_agent.totalSolveTime, brute_agent.mineFlagRate*100))
        la_cumulative_time+=la_agent.totalSolveTime
        la_cumulative_rate+=la_agent.mineFlagRate*100
        brute_cumulative_time+=brute_agent.totalSolveTime
        brute_cumulative_rate+=brute_agent.mineFlagRate*100
    print('\n\n\n\n\nFinished {} trials:\n\tLin alg average was {} seconds detcting {}% of mines\n\tBrute finished in {} seconds detcting {}% of mines'\
          .format(i+1, la_cumulative_time/trials, la_cumulative_rate/trials, brute_cumulative_time/trials, brute_cumulative_rate/trials))


dim = 50
density = 0.25
trialFileName = 'brute_force'

# baselineVsBruteComparisonGameDriver(dim,density,50)
linalgVsBruteComparisonGameDriver(dim,density,20)
linalgVsBruteComparisonGameDriver(dim,density,20)
linalgVsBruteComparisonGameDriver(dim,density,20)
linalgVsBruteComparisonGameDriver(dim,density,20)
linalgVsBruteComparisonGameDriver(dim,density,20)
