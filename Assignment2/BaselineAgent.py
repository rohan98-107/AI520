# Assignment 2 - Minesweeper
# Rohan Rele, Alex Eng, Aakash Raman, and Adarsh Patel
# Baseline agent code

from MineSweeper import *
from collections import deque

def solveBaseline(game):
    # use a queue of next cells (x,y) to visit
    q = deque()
    # randomly add a hidden cell to the queue
    q.appendleft(addRandomHiddenCell(game))

    while len(q) != 0:
        (x, y) = q.popleft()

        # if it's a mine, mark as detonated
        if game.board[x][y] == MINE:
            game.playerKnowledge[x][y] = DETONATED
            print('\nBOOM! Mine detonated at {}, {}\n\n'.format(x, y))
        # otherwise mark as safe, get clue & metadata, and infer safety/non-safety of nbrs if possible
        else:
            game.playerKnowledge[x][y] = SAFE
            clue = game.board[x][y]
            print('\nCell {}, {} safely revealed with clue: {}'.format(x, y, clue))
            numSafeNbrs, numMineNbrs, numHiddenNbrs, numTotalNbrs = getCellNeighborData(game, x, y)
            print('\t# safe neighbors: {}\n\t# mine neighbors: {}\n\t# hidden neighbors: {}\n\t# total neighbors: {}\n\n'\
                  .format(numSafeNbrs, numMineNbrs, numHiddenNbrs, numTotalNbrs))

            # if no neighbors are hidden, then we can't deduce any more new info about them
            if numHiddenNbrs == 0:
                pass
            # case when all hidden nbrs must be mines: mark as mine and fall through to adding a random hidden cell
            elif (clue - numMineNbrs) == numHiddenNbrs:
                print('All neighbors of {}, {} must be mines.\n'.format(x, y))
                for i, j in dirs:
                    if 0 <= x + i < game.dim and 0 <= y + j < game.dim:
                        if game.playerKnowledge[x+i][y+j] == HIDDEN:
                            game.playerKnowledge[x+i][y+j] = MINE
                            print('Neighbor {}, {} flagged as a mine.\n'.format(x+i, y+j))
            # case when all hidden nbrs must be safe: mark as safe, add safe cell(s),
            # and DON'T fall through to adding a random hidden cell
            elif (8 - clue - numSafeNbrs) == numHiddenNbrs:
                print('All neighbors of {}, {} must be safe.\n'.format(x, y))
                for i, j in dirs:
                    if 0 <= x + i < game.dim and 0 <= y + j < game.dim:
                        if game.playerKnowledge[x+i][y+j] == HIDDEN:
                            game.playerKnowledge[x+i][y+j] = SAFE
                            q.appendleft((x+i, y+j))
                            print('Neighbor {}, {} flagged as safe and enqueued for next visitation.\n'.format(x + i, y + j))
                            continue

        # add random hidden cell iff we detonated a mine or we detecting all nbrs were mines
        # if no hidden cells remaining, do nothing and keep clearing the queue
        if numHiddenCells(game) != 0:
            q.appendleft(addRandomHiddenCell(game))
            print('Revealing cell {}, {} led to no conclusive next move. Will randomly reveal a cell next.\n'.format(x, y))

    return game


# utility function to grab relevant metadata given game, (x,y) coordinates
def getCellNeighborData(game, x, y):
    numSafeNbrs = 0;    numMineNbrs = 0;    numHiddenNbrs = 0;      numTotalNbrs = 0
    for i, j in dirs:
        if 0 <= x + i < game.dim and 0 <= y + j < game.dim:
            numTotalNbrs += 1
            if (game.playerKnowledge[x+i][y+j] == MINE) or (game.playerKnowledge[x+i][y+j] == DETONATED):
                numMineNbrs += 1
            elif game.playerKnowledge[x+i][y+j] == SAFE:
                numSafeNbrs += 1
            elif game.playerKnowledge[x+i][y+j] == HIDDEN:
                numHiddenNbrs += 1
    return numSafeNbrs, numMineNbrs, numHiddenNbrs, numTotalNbrs

# utility function to return a random x,y from the game's currently hidden cells
def addRandomHiddenCell(game):
    (x, y) = np.where(game.playerKnowledge == HIDDEN)
    i = np.random.randint(len(x))
    return (x[i], y[i])

# utility function to return the number of hidden cells remaining
def numHiddenCells(game):
    return (game.playerKnowledge == HIDDEN).sum()

# test game
dim = 20
density = 0.25
num_mines = int(density*(dim**2))

game = MineSweeper(dim, num_mines)
game.saveBoard('test_init_board')

solvedGame = solveBaseline(game)
num_undetonated_mines = (solvedGame.playerKnowledge == MINE).sum()

print('\n\nGAME OVER\n\nSafely detected (without detonating) {}% of mines'.format((num_undetonated_mines / num_mines)*100))

solvedGame.savePlayerKnowledge('test_solved_board')