# Assignment 2 - Minesweeper
# Rohan Rele, Alex Eng, Aakash Raman, and Adarsh Patel
# Baseline agent code

from MineSweeper import *
from collections import deque
from copy import deepcopy
import sys
import time

class agent:
    def __init__(self, game):
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

    def solveBaseline(self):
        start = time.time()

        # use a queue of next cells (x,y) to visit
        q = deque()
        # randomly add a hidden cell to the queue; right now the best possible happens to be random choice
        self.enqueueBestPossibleCells(False, q)

        while len(q) != 0:
            (x, y) = q.popleft()

            # if it's a mine, mark as detonated
            if self.game.board[x][y] == MINE:
                self.playerKnowledge[x][y] = DETONATED; self.numDetonatedMines += 1
                print('\nBOOM! Mine detonated at ({}, {}).\n\n'.format(x, y))

            # otherwise mark as safe, get clue & metadata, and infer safety/non-safety of nbrs if possible
            else:
                self.playerKnowledge[x][y] = SAFE
                clue = self.game.board[x][y]
                print('\nCell ({}, {}) safely revealed with clue: {}.'.format(x, y, clue))
                numSafeNbrs, numMineNbrs, numHiddenNbrs, numTotalNbrs = self.getCellNeighborData(x, y)
                print('\t# safe neighbors: {}\n\t# mine neighbors: {}\n\t# hidden neighbors: {}\n\t# total neighbors: {}\n\n'\
                      .format(numSafeNbrs, numMineNbrs, numHiddenNbrs, numTotalNbrs))

                # if no neighbors are hidden, then we can't deduce any more new info about them
                if numHiddenNbrs == 0:
                    print('All neighbors of ({}, {}) are already revealed; nothing to infer.\n'.format(x, y))
                    pass

                # case when all hidden nbrs must be mines: mark as mine and fall through to adding a random hidden cell
                elif (clue - numMineNbrs) == numHiddenNbrs:
                    print('All neighbors of ({}, {}) must be mines.\n'.format(x, y))
                    for i, j in dirs:
                        if 0 <= x + i < self.game.dim and 0 <= y + j < self.game.dim:
                            if self.playerKnowledge[x + i][y + j] == HIDDEN:
                                self.playerKnowledge[x + i][y + j] = MINE;  self.numFlaggedMines += 1
                                print('\tNeighbor ({}, {}) flagged as a mine.\n'.format(x + i, y + j))

                # case when all hidden nbrs must be safe: mark as safe, add safe cell(s) to q,
                # and DON'T fall through to adding a random hidden cell
                elif (8 - clue - numSafeNbrs) == numHiddenNbrs:
                    print('All neighbors of ({}, {}) must be safe.\n'.format(x, y))
                    for i, j in dirs:
                        if 0 <= x + i < self.game.dim and 0 <= y + j < self.game.dim:
                            if self.playerKnowledge[x + i][y + j] == HIDDEN:
                                self.playerKnowledge[x + i][y + j] = SAFE
                                q.append((x+i, y+j))
                                print('\tNeighbor ({}, {}) flagged as safe and enqueued for next visitation.\n'.format(x + i, y + j))

            # if we detonated a mine or we inferred all nbrs were mines, then enqueue the next best possible cell(s)
            # i.e. re-crunch KB and try to mark new safes (and mines), enqueue safes; else randomly choose
            # if no hidden cells remaining, do nothing
            if self.numHiddenCells() != 0 and len(q) == 0:
                print('Revealing cell ({}, {}) led to no conclusive next move (either DETONATED or all neighbors MINES).\n'.format(x, y))
                print('Will attempt to re-deduce & enqueue new safe cell(s) from all of current knowledge,\n')
                print('or add random if none available.\n')
                self.enqueueBestPossibleCells(True, q)

            print('-' * 40)

        # store performance metrics
        self.totalSolveTime = time.time() - start
        self.mineFlagRate = self.numFlaggedMines / self.game.num_mines

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
                        clue = self.game.board[x][y]
                        numSafeNbrs, numMineNbrs, numHiddenNbrs, numTotalNbrs = self.getCellNeighborData(x, y)

                        # if no neighbors are hidden, then we can't deduce any more new info about them
                        if numHiddenNbrs == 0:
                            pass
                        # case when all hidden nbrs must be mines: mark as such
                        elif (clue - numMineNbrs) == numHiddenNbrs:
                            print('\tRe-processing KB found that: All neighbors of ({}, {}) must be mines.\n'.format(x, y))
                            for i, j in dirs:
                                if 0 <= x + i < self.game.dim and 0 <= y + j < self.game.dim:
                                    if self.playerKnowledge[x + i][y + j] == HIDDEN:
                                        self.playerKnowledge[x + i][y + j] = MINE; self.numFlaggedMines += 1
                                        print('\t\tNeighbor ({}, {}) flagged as a mine.\n'.format(x + i, y + j))
                        # case when all hidden nbrs must be safe: mark as safe, add safe cell(s) to q
                        elif (8 - clue - numSafeNbrs) == numHiddenNbrs:
                            print('\tRe-processing KB found that: All neighbors of ({}, {}) must be safe.\n'.format(x, y))
                            for i, j in dirs:
                                if 0 <= x + i < self.game.dim and 0 <= y + j < self.game.dim:
                                    if self.playerKnowledge[x + i][y + j] == HIDDEN:
                                        self.playerKnowledge[x + i][y + j] = SAFE
                                        q.append((x+i, y+j))
                                        print('\t\tNeighbor ({}, {}) flagged as safe and enqueued for next visitation.\n'.format(x + i, y + j))
                            safesFound = True


        # don't need to recrunch for the first enqueue
        if (recrunch is False) or (safesFound is False):
            if safesFound is False:
                print('\tRe-processing did not find new safe cells; proceeding to randomly select hidden cell.\n')
            (x_tuple, y_tuple) = np.where(self.playerKnowledge == HIDDEN)
            i = np.random.randint(len(x_tuple))
            x = x_tuple[i]; y = y_tuple[i]
            q.append((x, y))

    # utility function to return the number of hidden cells remaining
    def numHiddenCells(self):
        return (self.playerKnowledge == HIDDEN).sum()

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
        fontsize = 75 / (self.game.dim)
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

    baselineAgent = agent(game)
    baselineAgent.solveBaseline()

    print('\n\n***** GAME OVER *****\n\nGame ended in {} seconds\n\nSafely detected (without detonating) {}% of mines'\
          .format(baselineAgent.totalSolveTime, baselineAgent.mineFlagRate*100))

    baselineAgent.savePlayerKnowledge('{}_solved_board'.format(logFileName))


# test game driver
dim = 50
density = 0.25
trialFileName = 'test'

baselineGameDriver(dim, density, trialFileName)
