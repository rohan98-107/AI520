# Assignment 2 - Minesweeper
# Rohan Rele, Alex Eng, Aakash Raman, and Adarsh Patel
# Baseline agent code

from MineSweeper import *
from collections import deque
from copy import deepcopy
import sys
import time
import random

class agent:
    def __init__(self, game, order):
        # copy game into agent object
        self.game = deepcopy(game)
        # track player knowledge per cell: initially all hidden
        self.playerKnowledge = np.array([[HIDDEN] * self.game.dim for _ in range(self.game.dim)])

        # keep track of these throughout game to yield final mineFlagRate
        self.numFlaggedMines = 0
        self.numDetonatedMines = 0

        # these are the two metrics we can judge performance with
        self.mineFlagRate = 0
        self.totalSolveTime = 0
        self.logging = False

        self.order = order
        self.current_in_order = 0

        self.uncertaintyType = 'none'

    def enableLogging(self):
        self.logging = True

    def solve(self):
        start = time.time()

        # use a queue of next cells (x,y) to visit
        q = deque()
        # randomly add a hidden cell to the queue; right now the best possible happens to be random choice
        self.enqueueBestPossibleCells(False, q)

        while len(q) != 0:
            (x, y) = q.popleft()

            # if it's a mine, mark as detonated
            if self.game.board[x][y] == MINE:
                self.playerKnowledge[x][y] = DETONATED
                self.numDetonatedMines += 1
                if self.logging:
                    print('\nBOOM! Mine detonated at ({}, {}).\n\n'.format(x, y))

            # otherwise mark as safe, get clue & metadata, and infer safety/non-safety of nbrs if possible
            else:
                self.playerKnowledge[x][y] = SAFE
                clue = self.getClue(x, y)
                if clue != -1:


                    if self.logging:
                        print('\nCell ({}, {}) safely revealed with clue: {}.'.format(x, y, clue))
                    numSafeNbrs, numMineNbrs, numHiddenNbrs, numTotalNbrs = self.getCellNeighborData(x, y)
                    if self.logging:
                        print('\t# safe neighbors: {}\n\t# mine neighbors: {}\n\t# hidden neighbors: {}\n\t# total neighbors: {}\n\n'\
                          .format(numSafeNbrs, numMineNbrs, numHiddenNbrs, numTotalNbrs))

                    # if no neighbors are hidden, then we can't deduce any more new info about them
                    if numHiddenNbrs == 0:
                        if self.logging:
                            print('All neighbors of ({}, {}) are already revealed; nothing to infer.\n'.format(x, y))
                        pass

                    # case when all hidden nbrs must be mines: mark as mine and fall through to adding a random hidden cell
                    elif (clue - numMineNbrs) == numHiddenNbrs:
                        if self.logging:
                            print('All neighbors of ({}, {}) must be mines.\n'.format(x, y))
                        for i, j in dirs:
                            if 0 <= x + i < self.game.dim and 0 <= y + j < self.game.dim:
                                if self.playerKnowledge[x + i][y + j] == HIDDEN:
                                    #if self.logging:
                                    #    print(self.playerKnowledge)
                                    if self.uncertaintyType == 'none':
                                        assert self.game.board[x + i][y + j] == MINE
                                    self.playerKnowledge[x + i][y + j] = MINE
                                    self.numFlaggedMines += 1
                                    if self.logging:
                                        print('\tNeighbor ({}, {}) flagged as a mine.\n'.format(x + i, y + j))

                    # case when all hidden nbrs must be safe: mark as safe, add safe cell(s) to q,
                    # and DON'T fall through to adding a random hidden cell
                    elif (numTotalNbrs - clue - numSafeNbrs) == numHiddenNbrs:
                        if self.logging:
                            print('All neighbors of ({}, {}) must be safe.\n'.format(x, y))
                        for i, j in dirs:
                            if 0 <= x + i < self.game.dim and 0 <= y + j < self.game.dim:
                                if self.playerKnowledge[x + i][y + j] == HIDDEN:
                                    if self.uncertaintyType == 'none':
                                        assert self.game.board[x + i][y + j] != MINE
                                    self.playerKnowledge[x + i][y + j] = SAFE
                                    q.append((x+i, y+j))
                                    if self.logging:
                                        print('\tNeighbor ({}, {}) flagged as safe and enqueued for next visitation.\n'.format(x + i, y + j))

            # if we detonated a mine or we inferred all nbrs were mines, then enqueue the next best possible cell(s)
            # i.e. re-crunch KB and try to mark new safes (and mines), enqueue safes; else randomly choose
            # if no hidden cells remaining, do nothing
            if self.numHiddenCells() != 0 and len(q) == 0:
                if self.logging:
                    print('Revealing cell ({}, {}) led to no conclusive next move (either DETONATED or all neighbors MINES).\n'.format(x, y))
                    print('Will attempt to re-deduce & enqueue new safe cell(s) from all of current knowledge,\n')
                    print('or add random if none available.\n')
                self.enqueueBestPossibleCells(True, q)

            if self.logging:
                print('-' * 40)

        # store performance metrics
        self.totalSolveTime = time.time() - start
        self.mineFlagRate = self.numFlaggedMines / self.game.num_mines

        if self.uncertaintyType == 'none':
            dim = self.game.dim
            for x in range(dim):
                for y in range(dim):
                    if self.game.board[x][y] == MINE:
                        if not (self.playerKnowledge[x][y] == MINE or self.playerKnowledge[x][y] == DETONATED):
                            print(np.array(self.game.board))
                            print(self.playerKnowledge)
                            print("error at {},{}".format(x,y))
                        assert self.playerKnowledge[x][y] == MINE or self.playerKnowledge[x][y] == DETONATED
                    else:
                        if self.playerKnowledge[x][y] != SAFE:
                            print(np.array(self.game.board))
                            print(self.playerKnowledge)
                            print("error at {},{}".format(x,y))
                        assert self.playerKnowledge[x][y] == SAFE



    # utility function to grab relevant metadata given game, (x,y) coordinates
    def getCellNeighborData(self, x, y):
        numSafeNbrs = 0;    numMineNbrs = 0;    numHiddenNbrs = 0;  numTotalNbrs = 0
        for i, j in dirs:
            if 0 <= x + i < self.game.dim and 0 <= y + j < self.game.dim:
                numTotalNbrs += 1
                if (self.playerKnowledge[x + i][y + j] == MINE) or (self.playerKnowledge[x + i][y + j] == DETONATED):
                    numMineNbrs += 1
                elif self.playerKnowledge[x + i][y + j] == SAFE:
                    numSafeNbrs += 1
                elif self.playerKnowledge[x + i][y + j] == HIDDEN:
                    numHiddenNbrs += 1
        return numSafeNbrs, numMineNbrs, numHiddenNbrs, numTotalNbrs


    #gets all neighbors surrounding (x,y) that we haven't tested yet
    def get_hidden_neighbors(self, x, y):
        neighbors = []
        for i, j in dirs:
            if 0 <= x + i < self.game.dim and 0 <= y + j < self.game.dim and self.playerKnowledge[x + i][y + j] == HIDDEN:
                neighbors.append((x+i,y+j))
        return neighbors

    # utility function to return a random x,y from the game's currently hidden cells iff no new safe cells can be recovered
    # i.e. re-perform (recrunch) inference on all cells; if new safe cells found, enqueue them;
    # else randomly choose
    def enqueueBestPossibleCells(self, recrunch, q):
        safesFound = False;
        if recrunch:
            for x in range(self.game.dim):
                for y in range(self.game.dim):
                    # if it's not safe, then there's no need to check all its neighbors (nothing to deduce if mine/hidden)
                    if self.playerKnowledge[x][y] != SAFE:
                        pass
                    # otherwise do inference using neighbors
                    else:
                        clue = self.getClue(x, y)
                        if clue != -1:
                            numSafeNbrs, numMineNbrs, numHiddenNbrs, numTotalNbrs = self.getCellNeighborData(x, y)

                            # if no neighbors are hidden, then we can't deduce any more new info about them
                            if numHiddenNbrs == 0:
                                pass
                            # case when all hidden nbrs must be mines: mark as such
                            elif (clue - numMineNbrs) == numHiddenNbrs:
                                if self.logging:
                                    print('\tRe-processing KB found that: All neighbors of ({}, {}) must be mines.\n'.format(x, y))
                                for i, j in dirs:
                                    if 0 <= x + i < self.game.dim and 0 <= y + j < self.game.dim:
                                        if self.playerKnowledge[x + i][y + j] == HIDDEN:
                                            if self.uncertaintyType == 'none':
                                                assert self.game.board[x + i][y + j] == MINE
                                            self.playerKnowledge[x + i][y + j] = MINE
                                            self.numFlaggedMines += 1
                                            if self.logging:
                                                print('\t\tNeighbor ({}, {}) flagged as a mine.\n'.format(x + i, y + j))
                            # case when all hidden nbrs must be safe: mark as safe, add safe cell(s) to q
                            elif (numTotalNbrs - clue - numSafeNbrs) == numHiddenNbrs:
                                if self.logging:
                                    print('\tRe-processing KB found that: All neighbors of ({}, {}) must be safe.\n'.format(x, y))
                                for i, j in dirs:
                                    if 0 <= x + i < self.game.dim and 0 <= y + j < self.game.dim:
                                        if self.playerKnowledge[x + i][y + j] == HIDDEN:
                                            if self.uncertaintyType == 'none':
                                                assert self.game.board[x + i][y + j] != MINE
                                            self.playerKnowledge[x + i][y + j] = SAFE
                                            q.append((x+i, y+j))
                                            if self.logging:
                                                print('\t\tNeighbor ({}, {}) flagged as safe and enqueued for next visitation.\n'.format(x + i, y + j))
                                safesFound = True


        # don't need to recrunch for the first enqueue
        if (recrunch is False) or (safesFound is False):
            if recrunch is True and self.logging:
                print('\tRe-processing did not find new safe cells; proceeding to randomly select hidden cell.\n')
            tuple = self.probability_method()
            if tuple:
                q.append(tuple)

    def probability_method(self):
        tuple = self.get_next_random(set())
        return tuple

    def int_to_cell(self, x):
        dim = self.game.dim
        return (x//dim, x % dim)

    def get_next_random(self, to_exclude):
        dim = self.game.dim
        while self.current_in_order < dim ** 2:
            cell = self.int_to_cell(self.order[self.current_in_order])
            if self.playerKnowledge[cell[0]][cell[1]] == HIDDEN:
                break
            self.current_in_order += 1
        if self.current_in_order == dim ** 2:
            return None
        cell_to_consider = self.int_to_cell(self.order[self.current_in_order])
        if cell not in to_exclude:
            self.current_in_order += 1
            return cell
        for i in range(self.current_in_order + 1, dim**2):
            cell_to_consider = self.int_to_cell(self.order[self.current_in_order])
            if cell not in to_exclude:
                return cell
        cell = self.int_to_cell(self.order[self.current_in_order])
        self.current_in_order += 1
        return cell

    # utility function to return the number of hidden cells remaining
    def numHiddenCells(self):
        return (self.playerKnowledge == HIDDEN).sum()

   # utility function to return clue (or estimate for uncertain cases)
    def getClue(self, x, y, p=0.02):
        if self.uncertaintyType == "none":
            return self.game.board[x][y]
        elif self.uncertaintyType == "randomReveal":
            if random.random() < p:
                return self.game.board[x][y]
            else:
                return -1
        else:
            numSafeNbrs, numMineNbrs, numHiddenNbrs, numTotalNbrs = self.getCellNeighborData(x, y)
            if self.uncertaintyType == 'optimistic':
                if numMineNbrs < self.game.board[x][y]:
                    return random.randint(numMineNbrs, self.game.board[x][y])
                elif numMineNbrs == self.game.board[x][y]:
                    return self.game.board[x][y]
                else:
                    return -1
            elif self.uncertaintyType == 'cautious':
                if self.game.board[x][y] < numMineNbrs + numHiddenNbrs:
                    return random.randint(self.game.board[x][y], numMineNbrs + numHiddenNbrs)
                elif self.game.board[x][y] == numMineNbrs + numHiddenNbrs:
                    return self.game.board[x][y]
                else:
                    return -1
            else:
                return -1



    def printPlayerKnowledge(self):
        # Color Legend:
        # green (0): safely identified safe cells
        # orange/pink (-6): safely identified (undetonated) mine cells
        # red (-8): detonated mine cells

        imgsize = int(self.game.dim / 10)
        fontsize = 12.5 / (self.game.dim)
        dpi = 500

        plt.figure(figsize=(imgsize, imgsize), dpi=dpi)
        sns.heatmap(self.playerKnowledge, vmin=-8, vmax=0, linewidth=0.01, linecolor='lightgray',
                    annot=True, annot_kws={"size": fontsize},
                    square=True, cbar=False,
                    xticklabels=False,
                    yticklabels=False,
                    cmap='RdYlGn')
        plt.show()

    def savePlayerKnowledge(self, filename):
        imgsize = int(self.game.dim / 10)
        fontsize = 55 / (self.game.dim)
        dpi = 1000

        plt.figure(figsize=(imgsize, imgsize), dpi=dpi)
        sns.heatmap(self.playerKnowledge, vmin=-8, vmax=0, linewidth=0.01, linecolor='lightgray',
                    annot=True, annot_kws={"size": fontsize},
                    square=True, cbar=False,
                    xticklabels=False,
                    yticklabels=False,
                    cmap='RdYlGn')
        plt.savefig('{}.png'.format(filename), dpi=dpi)


# utility function to run game, save initial & solved boards, and print play-by-play to log txt file
# game metrics: mine safe detection rate and solve time calculated here; outputted to log
def baselineGameDriver(dim, density, logFileName):
    sys.stdout = open('{}_log.txt'.format(logFileName), 'w')

    num_mines = int(density*(dim**2))

    print('\n\n***** GAME STARTING *****\n\n{} by {} board with {} mines\n\nSolving with BASELINE strategy\n\n'\
          .format(dim, dim, num_mines))

    game = MineSweeper(dim, num_mines)
    game.saveBoard('{}_init_board'.format(logFileName))
    order = [i for i in range(dim**2)]
    random.shuffle(order)
    baselineAgent = agent(game, order)
    baselineAgent.enableLogging()
    baselineAgent.solve()

    print('\n\n***** GAME OVER *****\n\nGame ended in {} seconds\n\nSafely detected (without detonating) {}% of mines'\
          .format(baselineAgent.totalSolveTime, baselineAgent.mineFlagRate*100))

    baselineAgent.savePlayerKnowledge('{}_solved_board'.format(logFileName))

# utility function to run baseline with various uncertainty drivers
def baselineUncertaintyDriver(dim, density):
    num_mines = int(density * (dim ** 2))
    game = MineSweeper(dim, num_mines)

    order = [i for i in range(dim ** 2)]
    random.shuffle(order)

    solveTimes = []
    solveRates = []

    # to change randomReveal probability, change param p=? in getClue(...)
    uncertainties = ['none', 'randomReveal', 'optimistic', 'cautious']

    for u in uncertainties:
        baselineAgent = agent(deepcopy(game), order)
        baselineAgent.uncertaintyType = u
        #baselineAgent.enableLogging()
        baselineAgent.solve()
        solveTimes.append(baselineAgent.totalSolveTime)
        solveRates.append(baselineAgent.mineFlagRate)

    for u, v, w in zip(uncertainties, solveTimes, solveRates):
        print('\n\nUncertainty type {} solved dim {} board w/ density {}\nSolve time: {}\nMine flag rate: {}\n' \
              .format(u, dim, density, v, w))

#baselineUncertaintyDriver(50, 0.4)

baselineGameDriver(15, 0.3, 'baselineTrialRun')


