from MineSweeper import *
from collections import deque
from copy import deepcopy
import sys
import time
from Agent import *
from LinAlg import *

# agent using improved probabilities (called brute force as it computes individual cell)
# probabilities by brute force computing all configurations of mines
# this is a child class of lin_alg_agent so we gain the overridden enqueueBestPossibleCells
# method for our deterministic solutions. To provide the optimizations here we override
# probability_method as we are making more informed probabalistic choices

class brute_force_agent(lin_alg_agent):
    def __init__(self, game, useMineCount, order):
        lin_alg_agent.__init__(self,game,useMineCount,order)

    # helper method to confirm that a given board is consistent with KB
    # board is a possible configurations mines for a given component (see below algorithm for explanation of component)
    # cells is a list of all cells we have clues on that touch the component
    # For the potential board to be full consistent, for all the cells we must have
    # mine neighbors perfectly match the clue
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

    # this is consitency checker for in the middle of creating a possible configuration
    # of mines for a component. So we check temporary consistency by see if we add a mine
    # to a cell then we don't exceed the number of allowable mine neighbors for any of the cell's
    # (x,y) neighbors
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

    # overriding probability_method to use brute force probabilities
    # broad strokes implementation:
    # First create a list of components, a component is a set of hidden cells that are nighboring cells we have clues on
    # and can thus generate a probability it is a mine, two such hidden cells are in the same component if they both neighbor
    # the same safe cell, this is transitive. So basically a component is a set of cells that are potentially mines, and one being a
    # mine affects whther the other cells in the component are mines
    # This list of components of hidden cells is config_cells below
    # We also store the associated safe cells for each component in consistency_cells
    # cap is the maximum size for a component that we are willing to compute probabilities for, as larger components take exponentially longer to compute
    def probability_method(self, cap = 20):
        if self.logging:
            print("couldn't find definitively safe location, beginning computing brute force probabilities")
            print()
            print("game:")
            for row in self.game.board:
                print(row)
            print()
            print("knowledge:")
            print(self.playerKnowledge)
            print()
        dim = self.game.dim
        # list of safe cells separated by corresponding component, we will need to use confirm_full_consistency using the set of safe cells
        # to verufy a confuration of mines' consistency
        consistency_cells = list()
        # list of hidden cells separated by corresponding component, called config_cells as
        # each component will be used to generate potential configurations of mines
        config_cells = list()
        for x in range(dim):
            for y in range(dim):
                # if the cell has a clue
                if self.playerKnowledge[x][y] == SAFE:
                    numSafeNbrs, numMineNbrs, numHiddenNbrs, numTotalNbrs = self.getCellNeighborData(x, y)
                    # make sure there are hidden neighbors to try making mines
                    if numHiddenNbrs > 0:
                        # this cell we be needed for the consistency_cells component for all its children (neighbors)
                        children_consistency = {(x,y)}
                        children = set(self.get_hidden_neighbors(x,y))
                        intersects = []
                        # find all other components that have use any of the same hidden cells
                        for i,s in enumerate(config_cells):
                            if len(children.intersection(s)) > 0:
                                intersects.append(i)
                        # if any such components were found, then we add those components to the one we are currently creating,
                        # as they should actually be the same component. Then delete the old version separated versions those
                        # components.
                        if len(intersects) > 0:
                            for i in intersects:
                                children.update(config_cells[i])
                                children_consistency.update(consistency_cells[i])
                            for i in intersects[::-1]:
                                del config_cells[i]
                                del consistency_cells[i]
                        # Then we add our current component to our lists.
                        config_cells.append(children)
                        consistency_cells.append(children_consistency)

        # if we found no components then we just use pure random as we cannot compute probabilities
        if len(consistency_cells) == 0:
            if self.logging:
                print("couldn't find any cells to use for probabilities, returning pure random")
                print()
            return self.get_next_random(set())

        # just turn the config_cells into a list for simplicity
        config_cells = [sorted(list(y), key = lambda x: x[0] * dim + x[1]) for y in config_cells]

        # configs is potential configurations for a given component
        configs = []
        # we will remove some components for being too large to compute
        to_remove = []
        if self.logging:
            print("separated into components, now printing each component and possible sets of mines within component:")
            print()

        for i,s in enumerate(config_cells):
            if self.logging:
                print("for component:")
                print(s)
            # if a component is too large, note that we should remove it
            if len(s) > cap:
                to_remove.append(i)
                if self.logging:
                    print("component size > cap of {}, too computationally intensive to compute, ignoring".format(cap))
                    print()
                continue
            # otherwise get the list of possible mine configurations for the config, passing the corresponding
            # safe cell list (consistency_cells[i]) so we can ensure the configurations are possible
            start_search_time = time.time()
            current_configs = self.get_configs(s, consistency_cells[i])
            configs.append(current_configs)
            if self.logging:
                print("took {} seconds to compute following mine configurations".format( round(time.time() - start_search_time,2)))
                for c in current_configs:
                    print(c)
                print()

        # remove the components that are too large
        for i in to_remove[::-1]:
            del config_cells[i]
            del consistency_cells[i]

        if len(configs) == 0:
            if self.logging:
                print("no configurations found, using random cell")
                print()
            return self.get_next_random(set())

        if self.logging:
            print("mapping each cell to percentage of times it occurs in it's component's possible mine configurations")
            print()

        # minimum number of mines present among all configs
        min_mine_count = sum([ 0 if len(s) == 0 else min([len(config) for config in s]) for s in configs])
        if self.useMineCount:
            for s in configs:
                min_for_set =  0 if len(s) == 0 else min([len(config) for config in s])
                to_remove = []
                for i,config in enumerate(s):
                    # for each config for the component, if the number of mines in the config + the minimum number
                    # of mines in other configs (min_mine_count - min_for_set) exceeds the number of mines in the game,
                    # then remove the config as it is impossible
                    if len(config) + min_mine_count - min_for_set > self.game.num_mines-self.numFlaggedMines-self.numDetonatedMines:
                        to_remove.append(i)
                for i in to_remove[::-1]:
                    del s[i]

        # map each cell in a component to the percentage of configs of the component it appears in
        # this is it's probability of being a mine
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

        # get cell with lowest probability
        best_cell = None
        best_probability = len(configs)

        for cell, probability in probabilities.items():
            if probability <= best_probability:
                best_cell = cell
                best_probability = probability

        # the probability that a cell not in any component is a mine is uniform among all such cells
        # it is at most the maximum numnber of mines not in any component divided by the number of such cells
        other_cells = self.numHiddenCells() - sum([len(component) for component in config_cells])
        mines_left = self.game.num_mines-self.numFlaggedMines-self.numDetonatedMines
        max_mines_in_hidden_cells_not_in_config_Cells = mines_left - min_mine_count
        other_cell_max_probability = 1 if other_cells == 0 else max_mines_in_hidden_cells_not_in_config_Cells / other_cells

        # we can only compute and useother cell probabilities if using mine count
        # when we can use it, them if other cells have a lower probability, then instead
        # just return the next random excluding all hidden cells in any component
        # otherwise just return the best cell found above
        if self.useMineCount and best_probability > other_cell_max_probability:

            if self.logging:
                print("found best cell ({},{}) that was in {}% of it's component's configurations, but probability for other {} cells outside fringe is at most {}, so using random"\
                .format(best_cell[0],best_cell[1],round(best_probability,2),other_cells,round(other_cell_max_probability,2)))
                print()
            to_exclude = []
            for l in config_cells:
                to_exclude.extend(l)
            return self.get_next_random(set(to_exclude))
        else:
            if self.logging:
                print("found best cell ({},{}) that was in {}% of it's component's configurations"\
                    .format(best_cell[0],best_cell[1],round(best_probability,2)))
                print()
            assert best_cell is not None
            return best_cell


    # potential mine configuration finder
    def get_configs(self, configCells, consistency_cells):
        # configs is our current list of potential configs, we have a tuple with three elements
        # 1. a copy othe KB that will be modified to add mines
        # 2. an index of the next cell in the component to set as a mine, this never decreases to force
        #    an order on how we visit configurations, and removes looping. If index >= len(component) then we terminate
        # 3. a list of cells set to mines in the component to be returned if the config is valid
        configs = [(deepcopy(self.playerKnowledge), i, [])  for i in range(len(configCells))]
        # list of all valid configs
        out = []
        while configs:
            # essentially doing a bfs over configurations, instead of a queue we take all the nodes at a given depth
            # and then add all their children to next_set, and then use that for next iteration
            next_set = []
            for board, index, currentConfig in configs:
                current_board = deepcopy(board)
                current_config = deepcopy(currentConfig)
                dim = self.game.dim

                # terminate if index is out of bounds, or if using mine count and there are too many mines
                if index >= len(configCells) or (self.useMineCount and len(current_config) == self.game.num_mines-self.numFlaggedMines-self.numDetonatedMines):
                    # if the config is consitent, add it to our output list
                    if self.confirm_full_consistency(current_board, consistency_cells):
                        out.append(current_config)
                    continue

                # get the next cell we want to set as a mine and do so
                cell = configCells[index]
                current_config.append(cell)
                x = cell[0]
                y = cell[1]
                current_board[x,y] = MINE

                # initialize the config as valid, but set valid false if any neighbor has been made inconsistent
                valid = True
                for dx, dy in dirs:
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < dim and 0 <= ny < dim and self.playerKnowledge[nx][ny] == SAFE and not self.confirm_consistency(current_board, nx, ny):
                        valid = False
                # if every neighbor was found to be valid then add each of the node (the config's) children to the next list
                # it's children are adding any of the cells after index as a mine, or terminating (index = len(configCells))
                if valid:
                    for i in range(index + 1, len(configCells) + 1):
                        next_set.append((current_board,i,current_config))

            configs = next_set
        return out

def linearAlgebraWithBruteGameDriver(dim, density, logFileName, useMineCount = False):
    sys.stdout = open('{}_log.txt'.format(logFileName), 'w')

    num_mines = int(density*(dim**2))

    print('\n\n***** GAME STARTING *****\n\n{} by {} board with {} mines\n\nSolving with LINEAR ALGEBRA + BRUTE strategy\n\n'\
          .format(dim, dim, num_mines))
    order = [i for i in range(dim**2)]
    random.shuffle(order)
    game = MineSweeper(dim, num_mines)
    # game.saveBoard('{}_init_board'.format(logFileName))

    agent = brute_force_agent(game, useMineCount, order)
    agent.enableLogging()
    agent.solve()

    print('\n\n***** GAME OVER *****\n\nGame ended in {} seconds\n\nSafely detected (without detonating) {}% of mines'\
          .format(agent.totalSolveTime, agent.mineFlagRate*100))

def baselineVsBruteWithLinAlgComparisonGameDriver(dim, density, trials, useMineCount = False):
    print("baseline vs brute with lin alg, dim {}, density {}, trials {}, useMineCount={}".format(dim,density,trials,useMineCount))
    num_mines = int(density*(dim**2))
    baseline_cumulative_time = 0
    baseline_cumulative_rate = 0
    brute_cumulative_time = 0
    brute_cumulative_rate = 0
    for i in range(trials):
        game = MineSweeper(dim, num_mines)
        order = [i for i in range(dim**2)]
        random.shuffle(order)
        brute_agent = brute_force_agent(game, useMineCount, order)
        baselineAgent = agent(game, order)
        brute_agent.solve()
        baselineAgent.solve()
        baseline_cumulative_time+=baselineAgent.totalSolveTime
        baseline_cumulative_rate+=baselineAgent.mineFlagRate*100
        brute_cumulative_time+=brute_agent.totalSolveTime
        brute_cumulative_rate+=brute_agent.mineFlagRate*100
        if i % 10 == 9:
            print('\n\n\n\n\nFinished {} trials:\n\tBaseline average was {} seconds detcting {}% of mines\n\tBrute + lin alg finished in {} seconds detcting {}% of mines'\
                  .format(i+1, baseline_cumulative_time/(i+1), baseline_cumulative_rate/(i+1), brute_cumulative_time/(i+1), brute_cumulative_rate/(i+1)))

def linalgVsBruteWithLinAlgComparisonGameDriver(dim, density, trials, useMineCount = False):
    print("lin alg vs brute with lin alg, dim {}, density {}, trials {}, useMineCount={}".format(dim,density,trials,useMineCount))
    num_mines = int(density*(dim**2))
    la_cumulative_time = 0
    la_cumulative_rate = 0
    brute_cumulative_time = 0
    brute_cumulative_rate = 0
    for i in range(trials):
        game = MineSweeper(dim, num_mines)
        order = [i for i in range(dim**2)]
        random.shuffle(order)
        brute_agent = brute_force_agent(game, useMineCount, order)
        la_agent = lin_alg_agent(game, useMineCount, order)
        brute_agent.solve()
        la_agent.solve()
        la_cumulative_time+=la_agent.totalSolveTime
        la_cumulative_rate+=la_agent.mineFlagRate*100
        brute_cumulative_time+=brute_agent.totalSolveTime
        brute_cumulative_rate+=brute_agent.mineFlagRate*100
        if i % 10 == 9:
            print('\n\n\n\n\nFinished {} trials:\n\tLin alg average was {} seconds detcting {}% of mines\n\tBrute + lin alg finished in {} seconds detcting {}% of mines'\
                .format(i+1, la_cumulative_time/(i+1), la_cumulative_rate/(i+1), brute_cumulative_time/(i+1), brute_cumulative_rate/(i+1)))
